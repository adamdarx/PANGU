#include "z4c/evolution/tasks.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "driver/task_assembly.h"
#include "interface/update.hpp"
#include "mhd/mhd_package.h"
#include "z4c/evolution/package.h"
#include "z4c/coupling/sync_grhd.h"
#include "z4c/coupling/sync_grmhd.h"

namespace pangu::nr {

using namespace parthenon::driver::prelude;

namespace {

bool CanUsePackedSyncOutflow(Mesh* mesh) {
  return mesh->ndim == 3 && !mesh->multilevel &&
         std::all_of(mesh->mesh_bc_names.begin(), mesh->mesh_bc_names.end(),
                     [](const std::string& name) { return name == "nr_outflow"; });
}

TaskStatus CarrySyncStageStateMeshTask(MeshData<Real>* current, MeshData<Real>* next) {
  // AthenaK advances its dynamical-spacetime state in place: the RK kernels
  // replace active cells, while AMR ghost cells that are not owned by an
  // explicit boundary message retain their valid value from the preceding
  // stage.  Parthenon's low-storage driver rotates between stage containers,
  // so those same edge/corner ghosts would otherwise be uninitialized in
  // `next`.  Preserve the in-place semantics before the active-cell kernels
  // overwrite the new stage.  This is required when a WENO stencil is as wide
  // as a fine MeshBlock and CT consequently consumes codimension-2 ghosts.
  const auto& packages = current->GetParentPointer()->packages;
  const bool has_mhd = packages.AllPackages().count("mhd") != 0;
  const bool has_hydro = packages.AllPackages().count("hydro") != 0;
  std::vector<std::string> cell_variables{"nr.z4c"};
  if (has_mhd)
    cell_variables.emplace_back("mhd.cons");
  else if (has_hydro)
    cell_variables.emplace_back("hydro.cons");
  const auto source_cell = current->PackVariables(cell_variables);
  const auto target_cell = next->PackVariables(cell_variables);
  const auto entire_ib = current->GetBoundsI(IndexDomain::entire);
  const auto entire_jb = current->GetBoundsJ(IndexDomain::entire);
  const auto entire_kb = current->GetBoundsK(IndexDomain::entire);
  const auto interior_ib = current->GetBoundsI(IndexDomain::interior);
  const auto interior_jb = current->GetBoundsJ(IndexDomain::interior);
  const auto interior_kb = current->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU carry synchronized cell ghosts", parthenon::DevExecSpace(), 0,
      current->NumBlocks() - 1, 0, source_cell.GetDim(4) - 1, entire_kb.s, entire_kb.e, entire_jb.s,
      entire_jb.e, entire_ib.s, entire_ib.e,
      KOKKOS_LAMBDA(const int block, const int component, const int k, const int j, const int i) {
        const bool interior = i >= interior_ib.s && i <= interior_ib.e && j >= interior_jb.s &&
                              j <= interior_jb.e && k >= interior_kb.s && k <= interior_kb.e;
        if (!interior)
          target_cell(block, component, k, j, i) = source_cell(block, component, k, j, i);
      });

  if (has_mhd) {
    using TE = parthenon::TopologicalElement;
    const auto source_face = current->PackVariables(std::vector<std::string>{"mhd.b_face"});
    const auto target_face = next->PackVariables(std::vector<std::string>{"mhd.b_face"});
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU carry synchronized B1 ghosts", parthenon::DevExecSpace(), 0,
        current->NumBlocks() - 1, entire_kb.s, entire_kb.e, entire_jb.s, entire_jb.e, entire_ib.s,
        entire_ib.e + 1, KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
          const bool active = i >= interior_ib.s && i <= interior_ib.e + 1 && j >= interior_jb.s &&
                              j <= interior_jb.e && k >= interior_kb.s && k <= interior_kb.e;
          if (!active)
            target_face(block, TE::F1, 0, k, j, i) = source_face(block, TE::F1, 0, k, j, i);
        });
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU carry synchronized B2 ghosts", parthenon::DevExecSpace(), 0,
        current->NumBlocks() - 1, entire_kb.s, entire_kb.e, entire_jb.s, entire_jb.e + 1,
        entire_ib.s, entire_ib.e,
        KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
          const bool active = i >= interior_ib.s && i <= interior_ib.e && j >= interior_jb.s &&
                              j <= interior_jb.e + 1 && k >= interior_kb.s && k <= interior_kb.e;
          if (!active)
            target_face(block, TE::F2, 0, k, j, i) = source_face(block, TE::F2, 0, k, j, i);
        });
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU carry synchronized B3 ghosts", parthenon::DevExecSpace(), 0,
        current->NumBlocks() - 1, entire_kb.s, entire_kb.e + 1, entire_jb.s, entire_jb.e,
        entire_ib.s, entire_ib.e,
        KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
          const bool active = i >= interior_ib.s && i <= interior_ib.e && j >= interior_jb.s &&
                              j <= interior_jb.e && k >= interior_kb.s && k <= interior_kb.e + 1;
          if (!active)
            target_face(block, TE::F3, 0, k, j, i) = source_face(block, TE::F3, 0, k, j, i);
        });
  }
  return TaskStatus::complete;
}

void ConfigureSyncIntegrator(Mesh* mesh, parthenon::LowStorageIntegrator& integrator) {
  PARTHENON_REQUIRE(mesh->ndim == 3, "sync Z4c evolution requires a three-dimensional mesh");
  if (integrator.GetName() != "rk4")
    return;
  // AthenaK's four-stage RK4(4)[2S] differs from Parthenon's default
  // five-stage SSPRK(5,4).  The sync-Z4c reference path keeps Parthenon's
  // storage object but installs the reference coefficients and explicit
  // auxiliary-state accumulation used by AthenaK.
  integrator.nstages = 4;
  integrator.nbuffers = 2;
  integrator.gam0 = {0.0, 0.121098479554482, -3.843833699660025, 0.546370891121863};
  integrator.gam1 = {1.0, 0.721781678111411, 2.121209265338722, 0.198653035682705};
  integrator.beta = {1.193743905974738, 0.099279895495783, 1.131678018054042, 0.310665766509336};
  integrator.delta = {1.0, 0.217683334308543, 1.065841341361089, 0.0};
  integrator.c = {0.0, 1.193743905974738, 0.517227474715325, 1.0};
  integrator.buffer_name = {"base", "1", "base"};
  integrator.stage_name = {"base", "1", "2", "3", "base"};
}

void RunPostStep(Mesh* mesh, const parthenon::SimTime& time) {
  AdvancePunctureTracker(mesh, time.dt);
  RunPostStepDiagnostics(mesh, time.time + time.dt, time.ncycle + 1);
}

TaskCollection BuildSyncStage(driver::StageBuildContext& context) {
  auto* mesh = context.mesh;
  auto* integrator = context.integrator;
  const auto& tm = *context.time;
  const int stage = context.stage;
  const auto& stage_name = integrator->stage_name;
  const auto partitions = mesh->GetDefaultBlockPartitions();
  const auto beta = integrator->beta[stage - 1];
  const auto gam0 = integrator->gam0[stage - 1];
  const auto gam1 = integrator->gam1[stage - 1];
  const auto dt = integrator->dt;
  // ConfigureSyncIntegrator replaced the rk4 coefficients with the four-stage
  // low-storage scheme that needs the explicit accumulator below.
  const bool low_storage_rk4 = integrator->GetName() == "rk4";
  const bool vacuum = mesh->packages.Get("numerical_relativity")->Param<bool>("vacuum");
  TaskCollection collection;
  TaskID none(0);

  // Z4c and Valencia GRHD/GRMHD share one RK-stage graph. Both systems read the
  // same current ADM fields; their conservative updates may proceed in
  // parallel, then synchronize before the common boundary exchange. The
  // next-stage order is ADM reconstruction -> C2P -> Tmunu -> constraints.
  const bool sync_grhd = mesh->packages.AllPackages().count("hydro") != 0;
  const bool sync_grmhd = mesh->packages.AllPackages().count("mhd") != 0;
  const bool sync_matter = sync_grhd || sync_grmhd;
  auto& region = collection.AddRegion(partitions.size());
  for (std::size_t partition = 0; partition < partitions.size(); ++partition) {
    auto& tasks = region[partition];
    auto& base = mesh->mesh_data.Add("base", partitions[partition]);
    auto& current = mesh->mesh_data.Add(stage_name[stage - 1], base);
    auto& next = mesh->mesh_data.Add(stage_name[stage], base);
    std::shared_ptr<MeshData<Real>> emf_boundary;
    if (sync_grmhd)
      emf_boundary = mesh->mesh_data.AddShallow("sync_mhd_emf_" + stage_name[stage - 1], current,
                                                std::vector<std::string>{"mhd.edge_emf"});
    auto* rk_reference = base.get();
    // A Parthenon restart persists independent cell/face interiors, not the
    // multilevel ghost representation.  Re-canonicalize the current AMR
    // state once at the beginning of every complete RK step so a run that
    // resumes at this step is algebraically identical to an uninterrupted
    // run.  AthenaK likewise enters every dynamical-GRMHD stage from a
    // completed restrict/send/receive/prolongate boundary state.
    TaskID current_ready = none;
    if (stage == 1 && mesh->multilevel) {
      const auto current_receive = tasks.AddTask(
          none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, current);
      current_ready = parthenon::AddBoundaryExchangeTasks(current_receive, tasks, current, true);
      current_ready = tasks.AddTask(current_ready, parthenon::Update::FillDerived<MeshData<Real>>,
                                    current.get());
    }
    TaskID accumulated = current_ready;
    if (low_storage_rk4) {
      auto& accumulator = mesh->mesh_data.Add("nr_rk4_accumulator", base);
      accumulated = tasks.AddTask(current_ready, AccumulateRKStateMeshTask, current.get(),
                                  accumulator.get(), integrator->delta[stage - 1], stage == 1);
      rk_reference = accumulator.get();
    }
    const auto receive =
        tasks.AddTask(none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, next);
    TaskID emf_receive = none;
    if (sync_grmhd)
      emf_receive = tasks.AddTask(
          none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, emf_boundary);
    TaskID hydro_flux = none;
    TaskID hydro_flux_receive = none;
    TaskID matter_diagnostics_reset = none;
    if (sync_grmhd)
      matter_diagnostics_reset =
          tasks.AddTask(current_ready, ResetSyncGRMHDDiagnosticsMeshTask, current.get());
    if (sync_grhd) {
      hydro_flux = tasks.AddTask(current_ready, CalculateSyncGRHDFluxesMeshTask, current.get());
    } else if (sync_grmhd) {
      hydro_flux =
          tasks.AddTask(matter_diagnostics_reset, CalculateSyncGRMHDFluxesMeshTask, current.get());
    }
    if (sync_matter && mesh->multilevel)
      hydro_flux_receive = tasks.AddTask(none, parthenon::StartReceiveFluxCorrections, current);
    // The current stage already carries Tmunu from initialization or from
    // the preceding stage's C2P. Rebuilding it here duplicated AthenaK's
    // once-per-stage stress-energy evaluation without changing the state.
    const auto matter = accumulated;
    const auto carried =
        tasks.AddTask(current_ready, CarrySyncStageStateMeshTask, current.get(), next.get());
    const auto update = tasks.AddTask(matter | carried, StageUpdateMeshTask, current.get(),
                                      rk_reference, gam0, gam1, beta * dt, tm.time, next.get());
    const auto floor = tasks.AddTask(update, FloorChiMeshTask, next.get());
    TaskID hydro_update = none;
    TaskID magnetic_update = none;
    if (sync_matter) {
      auto flux_ready = hydro_flux;
      if (sync_grmhd) {
        const auto corrected = tasks.AddTask(hydro_flux, ApplySyncGRMHDFluxCorrectionMeshTask,
                                             current.get(), base.get(), gam0, gam1, beta * dt);
        const auto corner = tasks.AddTask(corrected, mhd::BuildCornerEMFMeshTask, current.get());
        flux_ready = parthenon::AddBoundaryExchangeTasks(
            corner | emf_receive, tasks, emf_boundary, mesh->multilevel,
            [](TaskID dependency, TaskList*, std::shared_ptr<MeshData<Real>>, bool) {
              return dependency;
            });
      }
      const auto flux_corrections = parthenon::AddFluxCorrectionTasks(
          flux_ready | hydro_flux_receive, tasks, current, mesh->multilevel);
      hydro_update = tasks.AddTask(flux_corrections | carried, driver::UpdateCellConservedMeshTask,
                                   current.get(), base.get(), gam0, gam1, beta * dt, next.get());
      hydro_update = sync_grhd ? tasks.AddTask(hydro_update | matter, ApplySyncGRHDSourcesMeshTask,
                                               current.get(), next.get(), beta * dt)
                               : tasks.AddTask(hydro_update | matter, ApplySyncGRMHDSourcesMeshTask,
                                               current.get(), next.get(), beta * dt);
      if (sync_grmhd)
        magnetic_update =
            tasks.AddTask(flux_corrections | carried, mhd::UpdateFaceFieldsMeshTask,
                          current.get(), base.get(), gam0, gam1, beta * dt, next.get());
    }
    const auto boundary_dependency = floor | hydro_update | magnetic_update | receive;
    const auto boundaries =
        CanUsePackedSyncOutflow(mesh)
            ? parthenon::AddBoundaryExchangeTasks(
                  boundary_dependency, tasks, next, false,
                  parthenon::BValOnMDFunc_t(ApplyPackedSyncOutflowBoundariesMeshTask))
            : parthenon::AddBoundaryExchangeTasks(boundary_dependency, tasks, next,
                                                  mesh->multilevel);
    auto projected = boundaries;
    // AthenaK projects the Z4c algebraic constraints after every RK stage
    // when dynamical matter is present, because the next-stage C2P and
    // stress-energy construction immediately consume the reconstructed ADM
    // fields.  Vacuum Z4c retains its reference behavior of projecting only
    // after the final stage.
    if (sync_matter || stage == integrator->nstages)
      projected = tasks.AddTask(boundaries, EnforceAlgebraicConstraintsMeshTask, next.get());
    const Real next_stage_time =
        stage < integrator->nstages ? tm.time + integrator->c[stage] * dt : tm.time + dt;
    TaskID ready = projected;
    if (sync_matter) {
      const auto adm_fields = tasks.AddTask(projected, Z4cToADMFieldsMeshTask, next.get());
      auto protected_state = adm_fields;
      if (sync_grmhd)
        protected_state =
            tasks.AddTask(adm_fields, ApplySyncGRMHDPunctureProtectionMeshTask, next.get());
      const auto primitive =
          sync_grhd
              ? tasks.AddTask(protected_state, SyncGRHDConservedToPrimitiveMeshTask, next.get())
              : tasks.AddTask(protected_state, SyncGRMHDConservedToPrimitiveMeshTask, next.get());
      const auto next_matter =
          tasks.AddTask(primitive, BuildStressEnergyMeshTask, next.get(), next_stage_time);
      // Constraints are diagnostics of the completed RK state and are not
      // inputs to the following stage's ADM/C2P chain. AthenaK evaluates
      // them once per full step rather than once per RK stage.
      ready = stage == integrator->nstages
                  ? tasks.AddTask(next_matter, ComputeConstraintsMeshTask, next.get())
                  : next_matter;
    } else {
      auto next_matter = projected;
      if (!vacuum)
        next_matter =
            tasks.AddTask(projected, BuildStressEnergyMeshTask, next.get(), next_stage_time);
      ready = tasks.AddTask(next_matter, Z4cToADMStageMeshTask, next.get());
    }
    if (stage == integrator->nstages) {
      const auto timestep =
          tasks.AddTask(ready, parthenon::Update::EstimateTimestep<MeshData<Real>>, next.get());
      if (mesh->adaptive)
        tasks.AddTask(timestep, parthenon::Refinement::Tag<MeshData<Real>>, next.get());
    }
  }
  return collection;
}

} // namespace

driver::StageContribution SyncStageContribution() {
  return {"sync_spacetime",
          100,
          [](const driver::StageBuildContext&) { return true; },
          BuildSyncStage,
          ConfigureSyncIntegrator,
          RunPostStep};
}

} // namespace pangu::nr
