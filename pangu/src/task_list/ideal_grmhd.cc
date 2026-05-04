// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include <memory>
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
  pin->CheckDesired("core", "q_factor_floor");
  pin->CheckDesired("core", "q_factor_ceiling");
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
  TaskRegion &single_tasklist_per_pack_region2 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region2[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
    auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);

    const auto any = parthenon::BoundaryType::any;

    tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mc1);
    tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mc0);
  }

  auto num_task_lists_executed_independently = blocks.size();
  TaskRegion &async_region1 =
      tc.AddRegion(num_task_lists_executed_independently);

  assert(blocks.size() == async_region1.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region1[i];

    auto &geom_sc = pmb->meshblock_data.Get(stage_name[0]);
    auto &sc0 = pmb->meshblock_data.Get(stage_name[stage - 1]);
    auto &dudt = pmb->meshblock_data.Get("dUdt");

    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
    auto fix_prim = tl.AddTask(none, FixPrimitive, sc0, geom_sc);
    auto calc_cons =
        tl.AddTask(fix_prim, CalculateConservative, sc0, geom_sc);
    auto calc_flux = tl.AddTask(calc_cons, CalculateFluxes, sc0, geom_sc);
    auto ct_task = tl.AddTask(calc_flux, ConstraintedTransport, sc0);
  }

  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc_init = pmesh->mesh_data.Add(stage_name[0], mbase);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
    auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);

    auto set_flx =
        parthenon::AddFluxCorrectionTasks(none, tl, mc0, pmesh->multilevel);
    auto flux_div = tl.AddTask(set_flx, FluxDivergence<MeshData<Real>>,
                               mc0.get(), mdudt.get());
    auto update = tl.AddTask(flux_div, UpdateIndependentData<MeshData<Real>>,
                             mc_init.get(), mdudt.get(), beta * dt, mc1.get());

    parthenon::AddBoundaryExchangeTasks(update, tl, mc1, pmesh->multilevel);
  }

  TaskRegion &async_region2 =
      tc.AddRegion(num_task_lists_executed_independently);
  assert(blocks.size() == async_region2.size());

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region2[i];

    auto &geom_sc = pmb->meshblock_data.Get(stage_name[0]);
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    auto source_task =
        tl.AddTask(none, AddGeometricSource, sc1, beta * dt, geom_sc);
    auto recover_task = tl.AddTask(source_task, Recovery, sc1, geom_sc);
    auto electron_heat = tl.AddTask(recover_task, ApplyElectronHeating, sc1, geom_sc);
    auto fix_rec = tl.AddTask(electron_heat, FixRecovery, sc1);
  }

  TaskRegion &async_region3 =
      tc.AddRegion(num_task_lists_executed_independently);
  assert(blocks.size() == async_region3.size());
  
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region3[i];
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    auto set_bc = tl.AddTask(none, parthenon::ApplyBoundaryConditions, sc1);
    auto fill_derived_op =
      parthenon::Update::FillDerived<MeshBlockData<Real>>;
    auto fill_derived = tl.AddTask(
      set_bc, fill_derived_op, sc1.get());
    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          fill_derived, EstimateTimestep<MeshBlockData<Real>>, sc1.get());
      if (pmesh->adaptive) {
        auto tag_refine = tl.AddTask(
            fill_derived, parthenon::Refinement::Tag<MeshBlockData<Real>>,
            sc1.get());
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
