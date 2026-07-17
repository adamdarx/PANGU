// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include <string>
#include <vector>

#include "riemann_solver/source_term.h"
#include "task_list/task_list.h"

using namespace parthenon::driver::prelude;

Simulator::Simulator(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {
  pin->CheckRequired("parthenon/mesh", "ix1_bc");
  pin->CheckRequired("parthenon/mesh", "ox1_bc");
  pin->CheckRequired("parthenon/mesh", "ix2_bc");
  pin->CheckRequired("parthenon/mesh", "ox2_bc");

  pin->CheckDesired("core", "cfl_number");
  pin->CheckDesired("core", "adiabatic_index");
}

TaskCollection Simulator::MakeTaskCollection(BlockList_t &blocks,
                                             const int stage) {
  using namespace parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  const Real beta = integrator->beta[stage % 2];
  const Real dt = integrator->dt;
  const auto &stage_name = integrator->stage_name;

  auto partitions = pmesh->GetDefaultBlockPartitions();
  int num_partitions = partitions.size();

  // Region 0: Start asynchronous MPI receives (overlaps with physics below).
  TaskRegion &mpi_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = mpi_region[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);

    const auto any = parthenon::BoundaryType::any;

    tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mc1);
    tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mc0);
  }

  // Region 1: All physics operations on a single partition-level task list
  // (MeshData + MeshBlockPack fused across blocks).
  TaskRegion &physics_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = physics_region[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc_init = pmesh->mesh_data.Add(stage_name[0], mbase);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
    auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);

    // ---- Flux computation (on mc0) ----
    auto fix_prim = tl.AddTask(none, FixPrimitive, mc0.get());
    auto calc_cons =
        tl.AddTask(fix_prim, CalculateConservative, mc0.get());
    auto calc_flux = tl.AddTask(calc_cons, CalculateFluxes, mc0.get());
    const auto enable_B_ct = pmesh->packages.Get("core")->Param<bool>("enable_B");
    auto ct_task = calc_flux;
    if (enable_B_ct) ct_task = tl.AddTask(calc_flux, ConstraintedTransport, mc0.get());

    // ---- Flux divergence + RK update (mc0  mdudt  mc1) ----
    auto set_flx = parthenon::AddFluxCorrectionTasks(
        ct_task, tl, mc0, pmesh->multilevel);
    auto flux_div = tl.AddTask(set_flx, FluxDivergence<MeshData<Real>>,
                               mc0.get(), mdudt.get());
    auto update = tl.AddTask(flux_div, UpdateIndependentData<MeshData<Real>>,
                             mc_init.get(), mdudt.get(), beta * dt, mc1.get());

    parthenon::AddBoundaryExchangeTasks(update, tl, mc1, pmesh->multilevel);

    // ---- Source + recovery (on mc1) ----
    auto source_task =
        tl.AddTask(update, AddGeometricSource, mc1.get(), beta * dt);
    auto recover_task = tl.AddTask(source_task, Recovery, mc1.get());
    const auto enable_heating =
        pmesh->packages.Get("core")->Param<bool>("enable_heating");
    auto post_recover = recover_task;
    if (enable_heating)
      post_recover = tl.AddTask(recover_task, ApplyElectronHeating, mc1.get());
    // The robuster (Kastaun) recovery path has no flag field and needs no
    // FixRecovery pass.
    const auto &recovery_name =
        pmesh->packages.Get("core")->Param<std::string>("recovery");
    if (recovery_name == "chainer")
      tl.AddTask(post_recover, FixRecovery, mc1.get());
  }

  // Region 2: Boundary conditions, fill derived, timestep estimation.
  TaskRegion &post_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = post_region[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);

    auto set_bc =
        tl.AddTask(none, parthenon::ApplyBoundaryConditionsMD, mc1);
    auto fill_derived = tl.AddTask(
        set_bc, parthenon::Update::FillDerived<MeshData<Real>>, mc1.get());
    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          fill_derived, EstimateTimestep<MeshData<Real>>, mc1.get());
      if (pmesh->adaptive) {
        auto tag_refine = tl.AddTask(
            fill_derived, parthenon::Refinement::Tag<MeshData<Real>>,
            mc1.get());
      }
    }
  }

  return tc;
}

parthenon::Packages_t ProcessPackages(
    std::unique_ptr<parthenon::ParameterInput> &pin) {
  Packages_t packages;
  auto package_core = core::Initialize(pin.get());
  auto package_metric = metric::Initialize(pin.get());
  packages.Add(package_core);
  packages.Add(package_metric);

  return packages;
}
