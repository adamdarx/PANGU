#include "radiation/task_contribution.h"

#include "driver/task_assembly.h"
#include "hydro/hydro_package.h"
#include "radiation/transport_package.h"

namespace pangu::radiation {

using namespace parthenon::driver::prelude;

TaskCollection ContributeTransportStage(driver::StageBuildContext& context) {
  auto* mesh = context.mesh;
  auto* integrator = context.integrator;
  const auto& time = *context.time;
  const int stage = context.stage;
  const auto& stage_name = integrator->stage_name;
  const auto partitions = mesh->GetDefaultBlockPartitions();
  const driver::StageContext stage_context{
      stage, integrator->nstages, integrator->beta[stage - 1],
      integrator->gam0[stage - 1], integrator->gam1[stage - 1], integrator->dt,
      time.time + integrator->c[stage - 1] * integrator->dt};
  const auto beta_dt = stage_context.StageDt();
  const auto gamma0 = stage_context.gamma0;
  const auto gamma1 = stage_context.gamma1;
  const bool fixed_fluid = mesh->packages.Get("radiation")->Param<bool>("fixed_fluid");

  TaskCollection collection;
  TaskID none(0);
  auto& region = collection.AddRegion(partitions.size());
  for (std::size_t partition = 0; partition < partitions.size(); ++partition) {
    auto& tasks = region[partition];
    auto& base = mesh->mesh_data.Add("base", partitions[partition]);
    auto& current = mesh->mesh_data.Add(stage_name[stage - 1], base);
    auto& next = mesh->mesh_data.Add(stage_name[stage], base);
    const auto receive_start = tasks.AddTask(
        none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, next);
    TaskID flux_receive = none;
    if (mesh->multilevel)
      flux_receive = tasks.AddTask(none, parthenon::StartReceiveFluxCorrections, current);
    const auto radiation_flux =
        tasks.AddTask(none, CalculateTransportFluxesMeshTask, current.get());
    TaskID flux_corrections;
    if (fixed_fluid && !mesh->multilevel) {
      flux_corrections = radiation_flux;
    } else {
      const auto hydro_flux =
          tasks.AddTask(none, hydro::CalculateFluxesMeshTask, current.get());
      auto hydro_corrected = hydro_flux;
      if (context.flux_correction != nullptr)
        hydro_corrected = tasks.AddTask(
            hydro_flux, hydro::ApplyFirstOrderFluxCorrectionMeshTask, current.get(),
            base.get(), gamma0, gamma1, beta_dt);
      flux_corrections = parthenon::AddFluxCorrectionTasks(
          hydro_corrected | radiation_flux | flux_receive, tasks, current,
          mesh->multilevel);
    }
    const auto radiation_update = tasks.AddTask(
        flux_corrections, UpdateTransportMeshTask, current.get(), base.get(), gamma0,
        gamma1, beta_dt, next.get());
    const auto radiation_source =
        tasks.AddTask(radiation_update, ApplyBeamSourceMeshTask, next.get(), beta_dt);
    TaskID coupling_primitives;
    if (fixed_fluid) {
      coupling_primitives = tasks.AddTask(
          radiation_source, hydro::CarryStateMeshTask, current.get(), next.get());
    } else {
      const auto hydro_update = tasks.AddTask(
          flux_corrections, driver::UpdateCellConservedMeshTask, current.get(), base.get(),
          gamma0, gamma1, beta_dt, next.get());
      const auto stage_primitives = tasks.AddTask(
          hydro_update, driver::CopyCellDerivedMeshTask, current.get(), next.get());
      const auto hydro_sources =
          tasks.AddTask(stage_primitives, hydro::ApplySourcesMeshTask, next.get(), beta_dt);
      coupling_primitives = tasks.AddTask(
          hydro_sources | radiation_source,
          parthenon::Update::FillDerived<parthenon::MeshData<parthenon::Real>>, next.get());
    }
    const auto coupling = tasks.AddTask(
        coupling_primitives, CoupleTransportToFluidMeshTask, next.get(), beta_dt);
    driver::AddStandardMeshFinalization(coupling, receive_start, tasks, next,
                                        stage_context);
  }
  return collection;
}

driver::StageContribution TransportStageContribution() {
  return {"radiation_transport", 100,
          [](const driver::StageBuildContext&) { return true; },
          ContributeTransportStage};
}

} // namespace pangu::radiation
