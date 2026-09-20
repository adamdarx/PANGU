#include "hydro/task_contribution.h"

#include "driver/task_assembly.h"
#include "geometry_assembly.h"
#include "hydro/hydro_package.h"
#include "riemann/registry.h"

namespace pangu::hydro {

using namespace parthenon::driver::prelude;

namespace {

bool SupportsFixedGRHydro(const driver::StageBuildContext& context) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized())
    return false;

  auto* mesh = context.mesh;
  const auto& packages = mesh->packages;
  if (mesh->adaptive || packages.AllPackages().count("hydro") == 0 ||
      packages.AllPackages().count("mhd") != 0)
    return false;

  const auto hydro = packages.Get("hydro");
  if (hydro->Param<int>("physics_mode") != 2 || hydro->Param<int>("nscalars") != 0)
    return false;
  const auto solver = static_cast<riemann::Solver>(hydro->Param<int>("riemann"));
  if (solver != riemann::Solver::llf && solver != riemann::Solver::hlle)
    return false;

  return true;
}

bool SupportsPackedNewtonianHydro(const driver::StageBuildContext& context) {
  auto* mesh = context.mesh;
  const auto& packages = mesh->packages;
  if (mesh->adaptive || packages.AllPackages().count("hydro") == 0 ||
      packages.AllPackages().count("mhd") != 0)
    return false;
  const auto hydro = packages.Get("hydro");
  if (hydro->Param<int>("physics_mode") != 0 ||
      static_cast<riemann::Solver>(hydro->Param<int>("riemann")) == riemann::Solver::none ||
      hydro->Param<bool>("fofc") ||
      hydro->Param<Real>("accel1") != 0.0 || hydro->Param<Real>("accel2") != 0.0 ||
      hydro->Param<Real>("accel3") != 0.0)
    return false;
  const auto sources = packages.Get("source_terms");
  return sources->Param<Real>("cooling_rate") == 0.0 &&
         sources->Param<Real>("heating_rate") == 0.0 &&
         sources->Param<Real>("point_mass") == 0.0 &&
         !sources->Param<bool>("constant_acceleration") &&
         !sources->Param<bool>("ism_cooling") &&
         !sources->Param<bool>("relativistic_cooling");
}

bool SupportsHydro(const driver::StageBuildContext& context) {
  return SupportsFixedGRHydro(context) || SupportsPackedNewtonianHydro(context);
}

TaskCollection BuildFixedGRHydroStage(driver::StageBuildContext& context) {
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

  TaskCollection collection;
  TaskID none(0);
  auto& region = collection.AddRegion(partitions.size());
  for (std::size_t partition = 0; partition < partitions.size(); ++partition) {
    auto& tasks = region[partition];
    auto& base = mesh->mesh_data.Add("base", partitions[partition]);
    auto& current = mesh->mesh_data.Add(stage_name[stage - 1], base);
    auto& next = mesh->mesh_data.Add(stage_name[stage], base);
    mesh->mesh_data.Add("dUdt", base);

    const auto receive_start = tasks.AddTask(
        none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, next);
    TaskID flux_receive(0);
    if (mesh->multilevel)
      flux_receive = tasks.AddTask(none, parthenon::StartReceiveFluxCorrections, current);
    const auto flux = tasks.AddTask(none, CalculateFluxesMeshTask, current.get());
    const auto corrected = tasks.AddTask(
        flux, ApplyFirstOrderFluxCorrectionMeshTask, current.get(), base.get(),
        stage_context.gamma0, stage_context.gamma1, stage_context.StageDt());
    const auto flux_corrections = parthenon::AddFluxCorrectionTasks(
        corrected | flux_receive, tasks, current, mesh->multilevel);
    const auto update = tasks.AddTask(
        flux_corrections, driver::UpdateCellConservedMeshTask, current.get(), base.get(),
        stage_context.gamma0, stage_context.gamma1, stage_context.StageDt(), next.get());
    const auto stage_primitives = tasks.AddTask(
        update, driver::CopyCellDerivedMeshTask, current.get(), next.get());
    const auto sources = tasks.AddTask(stage_primitives, ApplySourcesMeshTask, next.get(),
                                       stage_context.StageDt());
    driver::AddStandardMeshFinalization(sources, receive_start, tasks, next,
                                        stage_context);
  }
  return collection;
}

TaskCollection BuildPackedNewtonianHydroStage(driver::StageBuildContext& context) {
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
    const auto flux = tasks.AddTask(none, CalculateFluxesMeshTask, current.get());
    const auto update = tasks.AddTask(
        flux, driver::UpdateCellConservedMeshTask, current.get(), base.get(),
        stage_context.gamma0, stage_context.gamma1, stage_context.StageDt(), next.get());
    const auto boundaries = driver::AddUniformBoundaryExchangeTasks(
        update | receive_start, tasks, next);
    const auto derived = tasks.AddTask(
        boundaries, parthenon::Update::FillDerived<MeshData<Real>>, next.get());
    driver::AddFinalStageTasks(derived, tasks, next.get(), stage_context);
  }
  return collection;
}

} // namespace

TaskCollection BuildHydroStage(driver::StageBuildContext& context) {
  return context.mesh->packages.Get("hydro")->Param<int>("physics_mode") == 0
             ? BuildPackedNewtonianHydroStage(context)
             : BuildFixedGRHydroStage(context);
}

driver::StageContribution HydroStageContribution() {
  return {"hydro", 50, SupportsHydro, BuildHydroStage};
}

} // namespace pangu::hydro
