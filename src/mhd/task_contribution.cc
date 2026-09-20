#include "mhd/task_contribution.h"

#include <string>
#include <vector>

#include "driver/task_assembly.h"
#include "geometry_assembly.h"
#include "mhd/mhd_package.h"
#include "driver/stage_extension.h"
#include "relativity/relativistic_mhd.h"
#include "riemann/registry.h"
#include "srcterms/source_terms.h"

namespace pangu::mhd {

using namespace parthenon::driver::prelude;

namespace {

bool SupportsFixedGRMHD(const driver::StageBuildContext& context) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized())
    return false;

  auto* mesh = context.mesh;
  const auto& packages = mesh->packages;
  if (mesh->adaptive || packages.AllPackages().count("mhd") == 0 ||
      packages.AllPackages().count("hydro") != 0 ||
      mesh->GetDefaultBlockPartitions().size() != 1)
    return false;
  const auto mhd = packages.Get("mhd");
  if (mhd->Param<int>("physics_mode") != 2 || mhd->Param<int>("nscalars") != 0)
    return false;
  const auto solver = static_cast<riemann::Solver>(mhd->Param<int>("riemann"));
  if (solver != riemann::Solver::llf && solver != riemann::Solver::hlle)
    return false;
  return !srcterms::HasActiveSources(packages);
}

// Non-GR MHD on one uniform partition without stage sources, FOFC, or kinematic evolution.
bool SupportsSingleRegionMHD(const driver::StageBuildContext& context) {
  auto* mesh = context.mesh;
  const auto& packages = mesh->packages;
  if (mesh->adaptive || packages.AllPackages().count("mhd") == 0 ||
      mesh->GetDefaultBlockPartitions().size() != 1 || srcterms::HasActiveSources(packages))
    return false;
  const auto mhd = packages.Get("mhd");
  return mhd->Param<int>("physics_mode") != 2 &&
         static_cast<riemann::Solver>(mhd->Param<int>("riemann")) != riemann::Solver::none &&
         !mhd->Param<bool>("fofc");
}

bool SupportsMHD(const driver::StageBuildContext& context) {
  return SupportsFixedGRMHD(context) || SupportsSingleRegionMHD(context);
}

TaskID AddRegisteredStageSources(TaskID dependency, TaskID face_update,
                                 TaskList& tasks, MeshData<Real>* next,
                                 const driver::StageContext& stage_context,
                                 const Packages_t& packages) {
  auto result = dependency;
  for (const auto* extension : driver::FindStageExtensions<driver::StageSourceExtension>(
           packages, driver::stage_source_extension_key)) {
    result = extension->add_source_task(
        result, face_update, tasks, next, stage_context.source_time,
        stage_context.StageDt(), stage_context.IsFinal());
  }
  return result;
}

TaskCollection BuildFixedGRMHDStage(driver::StageBuildContext& context) {
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
  const auto* passive = driver::FindStageExtension<driver::PassiveStageExtension>(
      mesh->packages, driver::passive_stage_extension_key);

  TaskCollection collection;
  TaskID none(0);
  auto& region = collection.AddRegion(1);
  auto& tasks = region[0];
  auto& base = mesh->mesh_data.Add("base", partitions[0]);
  auto& current = mesh->mesh_data.Add(stage_name[stage - 1], base);
  auto& next = mesh->mesh_data.Add(stage_name[stage], base);
  auto& emf_boundary = mesh->mesh_data.AddShallow(
      "mhd_emf_" + stage_name[stage - 1], current,
      std::vector<std::string>{"mhd.edge_emf"});
  std::shared_ptr<MeshData<Real>> passive_boundary;
  if (passive != nullptr && passive->applies_heating)
    passive_boundary = mesh->mesh_data.AddShallow(
        "passive_heated_" + stage_name[stage], next,
        std::vector<std::string>{std::string(passive->conserved_field)});

  const auto receive_start = tasks.AddTask(
      none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, next);
  const auto emf_receive_start = tasks.AddTask(
      none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, emf_boundary);
  TaskID flux_receive(0);
  if (mesh->multilevel)
    flux_receive = tasks.AddTask(none, parthenon::StartReceiveFluxCorrections, current);
  const auto flux = passive != nullptr
                        ? tasks.AddTask(none, relativity::CalculateMHDFluxesWithPassiveMeshTask,
                                        current.get(), std::string(passive->conserved_field))
                        : tasks.AddTask(none, relativity::CalculateMHDFluxesMeshTask,
                                        current.get());
  auto corrected = flux;
  if (context.flux_correction != nullptr) {
    corrected = passive != nullptr
                    ? tasks.AddTask(flux, passive->apply_flux_correction, current.get(),
                                    base.get(), stage_context.gamma0,
                                    stage_context.gamma1, stage_context.StageDt(),
                                    std::string(passive->conserved_field))
                    : tasks.AddTask(flux, relativity::ApplyMHDFluxCorrectionMeshTask,
                                    current.get(), base.get(), stage_context.gamma0,
                                    stage_context.gamma1, stage_context.StageDt());
  }
  TaskID passive_flux(0);
  if (passive != nullptr)
    passive_flux = tasks.AddTask(corrected, passive->calculate_fluxes, current.get());
  const auto corner_emf = tasks.AddTask(corrected, BuildCornerEMFMeshTask, current.get());
  const auto emf_boundaries = parthenon::AddBoundaryExchangeTasks(
      corner_emf | emf_receive_start, tasks, emf_boundary, mesh->multilevel,
      [](TaskID dependency, TaskList*, std::shared_ptr<MeshData<Real>>, bool) {
        return dependency;
      });
  const auto flux_corrections = parthenon::AddFluxCorrectionTasks(
      emf_boundaries | passive_flux | flux_receive, tasks, current, mesh->multilevel);
  const auto cell_update = tasks.AddTask(
      flux_corrections, driver::UpdateCellConservedMeshTask, current.get(), base.get(),
      stage_context.gamma0, stage_context.gamma1, stage_context.StageDt(), next.get());
  TaskID passive_update(0);
  if (passive != nullptr)
    passive_update = tasks.AddTask(
        flux_corrections, passive->update_conserved, current.get(), base.get(),
        stage_context.gamma0, stage_context.gamma1, stage_context.StageDt(), next.get());
  const auto face_update = tasks.AddTask(
      flux_corrections, UpdateFaceFieldsMeshTask, current.get(), base.get(),
      stage_context.gamma0, stage_context.gamma1, stage_context.StageDt(), next.get());
  const auto stage_primitives = tasks.AddTask(
      cell_update, driver::CopyCellDerivedMeshTask, current.get(), next.get());
  const auto fluid_sources = tasks.AddTask(stage_primitives, ApplySourcesMeshTask,
                                           next.get(), stage_context.StageDt());
  const auto sources = AddRegisteredStageSources(
      fluid_sources, face_update, tasks, next.get(), stage_context, mesh->packages);
  const auto boundaries = parthenon::AddBoundaryExchangeTasks(
      sources | face_update | passive_update | receive_start, tasks, next,
      mesh->multilevel);
  auto derived = tasks.AddTask(boundaries, ConservedToPrimitiveMeshTask, next.get());
  if (passive != nullptr)
    derived = tasks.AddTask(derived, passive->conserved_to_primitive, next.get());
  if (passive != nullptr && passive->applies_heating) {
    derived = tasks.AddTask(derived, passive->apply_heating, current.get(), next.get());
    derived = tasks.AddTask(derived, passive->primitive_to_conserved, next.get());
    const auto passive_receive = tasks.AddTask(
        derived, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>,
        passive_boundary);
    const auto passive_boundaries = parthenon::AddBoundaryExchangeTasks(
        derived | passive_receive, tasks, passive_boundary, mesh->multilevel,
        [](TaskID dependency, TaskList*, std::shared_ptr<MeshData<Real>>, bool) {
          return dependency;
        });
    derived = tasks.AddTask(passive_boundaries, passive->apply_physical_boundaries,
                            next.get());
    derived = tasks.AddTask(derived, passive->conserved_to_primitive, next.get());
  }
  driver::AddFinalStageTasks(derived, tasks, next.get(), stage_context);
  return collection;
}

TaskCollection BuildSingleRegionMHDStage(driver::StageBuildContext& context) {
  auto* mesh = context.mesh;
  auto* integrator = context.integrator;
  auto& blocks = *context.blocks;
  const int stage = context.stage;
  const auto& stage_name = integrator->stage_name;
  const auto partitions = mesh->GetDefaultBlockPartitions();
  const auto beta_dt = integrator->beta[stage - 1] * integrator->dt;
  const auto gamma0 = integrator->gam0[stage - 1];
  const auto gamma1 = integrator->gam1[stage - 1];

  // One dependency graph over the base MeshData partition.  Block-local flux
  // and primitive kernels remain tasks in this graph; the conservative and CT
  // updates operate on the complete partition without TaskRegion barriers.
  TaskCollection collection;
  TaskID none(0);
  auto& region = collection.AddRegion(1);
  auto& tasks = region[0];
  auto& base = mesh->mesh_data.Add("base", partitions[0]);
  auto& current = mesh->mesh_data.Add(stage_name[stage - 1], base);
  auto& next = mesh->mesh_data.Add(stage_name[stage], base);
  const auto receive_start =
      tasks.AddTask(none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, next);
  auto fluxes = TaskID(0);
  for (std::size_t block = 0; block < blocks.size(); ++block) {
    auto& current_block = blocks[block]->meshblock_data.Get(stage_name[stage - 1]);
    fluxes = fluxes | tasks.AddTask(none, context.flux, current_block);
  }
  const auto cell_update = tasks.AddTask(fluxes, driver::UpdateCellConservedMeshTask,
                                         current.get(), base.get(), gamma0, gamma1, beta_dt,
                                         next.get());
  const auto face_update = tasks.AddTask(fluxes, UpdateFaceFieldsMeshTask,
                                         current.get(), base.get(), gamma0, gamma1, beta_dt,
                                         next.get());
  const auto boundaries = driver::AddUniformBoundaryExchangeTasks(
      cell_update | face_update | receive_start, tasks, next);
  for (std::size_t block = 0; block < blocks.size(); ++block) {
    auto& next_block = blocks[block]->meshblock_data.Get(stage_name[stage]);
    const auto derived = tasks.AddTask(
        boundaries, parthenon::Update::FillDerived<MeshBlockData<Real>>, next_block.get());
    if (stage == integrator->nstages)
      tasks.AddTask(derived, parthenon::Update::EstimateTimestep<MeshBlockData<Real>>,
                    next_block.get());
  }
  return collection;
}

TaskCollection BuildMHDStage(driver::StageBuildContext& context) {
  return SupportsFixedGRMHD(context) ? BuildFixedGRMHDStage(context)
                                     : BuildSingleRegionMHDStage(context);
}

} // namespace

driver::StageContribution MHDStageContribution() {
  return {"mhd", 50, SupportsMHD, BuildMHDStage};
}

} // namespace pangu::mhd
