#include "driver/task_assembly.h"
#include "driver/contributor.h"
#include "driver/stage_extension.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "amr_criteria/refinement_package.hpp"
#include "app/problem_registry.h"
#include "bvals/comms/bvals_in_one.hpp"
#include "geometry_assembly.h"
#include "interface/update.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "pangu/plugin_api/diffusion.h"

namespace pangu::driver {
using namespace parthenon::driver::prelude;

namespace {

std::vector<const StageContribution*> RegisteredContributions(Mesh* mesh) {
  std::vector<const StageContribution*> contributions;
  for (const auto& [name, package] : mesh->packages.AllPackages()) {
    if (package->AllParams().hasKey(std::string(stage_contribution_key)))
      contributions.push_back(
          &package->Param<StageContribution>(std::string(stage_contribution_key)));
  }
  return contributions;
}

} // namespace

TaskID AddUniformBoundaryExchangeTasks(TaskID dependency, TaskList& tasks,
                                       std::shared_ptr<MeshData<Real>>& data) {
  // This is the uniform-grid specialization of Parthenon's standard boundary
  // task sequence.  The generic SendBoundBufs path requests restriction even
  // when every neighbor is on the same level.  Use the framework's native
  // no-restriction send while retaining its receive, unpack, and physical-BC
  // implementations.
  const auto send = tasks.AddTask(
      dependency, parthenon::SendBoundBufsNoRestrict<parthenon::BoundaryType::any>, data);
  const auto receive =
      tasks.AddTask(dependency, parthenon::ReceiveBoundBufs<parthenon::BoundaryType::any>, data);
  const auto set = tasks.AddTask(receive, parthenon::SetBounds<parthenon::BoundaryType::any>, data);
  const auto boundaries =
      tasks.AddTask(set, parthenon::ApplyBoundaryConditionsOnCoarseOrFineMD, data, false);
  return send | boundaries;
}

std::vector<std::string> CellConservedVariables(Mesh* mesh) {
  const auto* fluid =
      FindStageExtension<FluidStageExtension>(mesh->packages, fluid_stage_extension_key);
  if (fluid != nullptr && !fluid->conserved.empty())
    return fluid->conserved;
  const parthenon::Metadata::FlagCollection flags{parthenon::Metadata::Independent,
                                                  parthenon::Metadata::Cell,
                                                  parthenon::Metadata::WithFluxes};
  return mesh->GetVariableNames(flags);
}

TaskStatus UpdateCellConservedMeshTask(MeshData<Real>* current, MeshData<Real>* base,
                                       const Real gam0, const Real gam1, const Real beta_dt,
                                       MeshData<Real>* next) {
  // AthenaK accumulates the three Cartesian flux differences after dividing
  // each by its cell width, then performs the complete RK expression in one
  // assignment.  Parthenon's generic divergence uses face areas/cell volumes
  // and PANGU formerly split the RK average and update into two kernels.  Those
  // forms are algebraically equivalent, but their roundoff is observable in
  // floor-dominated GRMHD inversions.  Preserve AthenaK's expression tree.
  auto* mesh = current->GetParentPointer();
  if (const auto* custom = FindStageExtension<ConservedUpdateExtension>(
          mesh->packages, conserved_update_extension_key))
    return custom->update(current, base, gam0, gam1, beta_dt, next);
  const auto variables = CellConservedVariables(mesh);
  const auto state = current->PackVariablesAndFluxes(variables);
  const auto initial = base->PackVariables(variables);
  const auto output = next->PackVariables(variables);
  const auto ib = current->GetBoundsI(IndexDomain::interior);
  const auto jb = current->GetBoundsJ(IndexDomain::interior);
  const auto kb = current->GetBoundsK(IndexDomain::interior);
  const int ndim = state.GetNdim();
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU cell update", parthenon::DevExecSpace(), 0,
      state.GetDim(5) - 1, 0, state.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
        if (state.IsAllocated(m, n) && initial.IsAllocated(m, n) && output.IsAllocated(m, n)) {
          const auto& coords = state.GetCoords(m);
          const auto& block_state = state(m);
          Real divergence = (block_state.flux(parthenon::X1DIR, n, k, j, i + 1) -
                             block_state.flux(parthenon::X1DIR, n, k, j, i)) /
                            coords.Dxc<parthenon::X1DIR>(k, j, i);
          if (ndim >= 2)
            divergence += (block_state.flux(parthenon::X2DIR, n, k, j + 1, i) -
                           block_state.flux(parthenon::X2DIR, n, k, j, i)) /
                          coords.Dxc<parthenon::X2DIR>(k, j, i);
          if (ndim >= 3)
            divergence += (block_state.flux(parthenon::X3DIR, n, k + 1, j, i) -
                           block_state.flux(parthenon::X3DIR, n, k, j, i)) /
                          coords.Dxc<parthenon::X3DIR>(k, j, i);
          output(m, n, k, j, i) =
              gam0 * state(m, n, k, j, i) + gam1 * initial(m, n, k, j, i) - beta_dt * divergence;
        }
      });
  return TaskStatus::complete;
}

TaskStatus CopyCellDerivedBlockTask(MeshBlockData<Real>* current, MeshBlockData<Real>* next) {
  // Source terms in AthenaK are evaluated from the primitive state at the
  // beginning of the RK stage while they update the stage-output conserved
  // state.  PANGU's stage buffers are periodic (the final buffer is `base`), so
  // their derived fields do not in general belong to `current`.  Copy only the
  // cell-derived state here; conserved and face-independent variables in
  // `next` must retain the just-computed RK update.
  const auto* fluid = FindStageExtension<FluidStageExtension>(
      current->GetBlockPointer()->packages, fluid_stage_extension_key);
  if (fluid == nullptr)
    return TaskStatus::complete;
  return parthenon::Update::CopyData(fluid->stage_derived, current, next);
}

TaskStatus CopyCellDerivedMeshTask(MeshData<Real>* current, MeshData<Real>* next) {
  const auto* fluid = FindStageExtension<FluidStageExtension>(
      current->GetParentPointer()->packages, fluid_stage_extension_key);
  if (fluid == nullptr)
    return TaskStatus::complete;
  return parthenon::Update::CopyData(fluid->stage_derived, current, next);
}

TaskID AddFinalStageTasks(TaskID dependency, TaskList& tasks, MeshData<Real>* next,
                          const StageContext& stage) {
  if (!stage.IsFinal())
    return dependency;
  const auto timestep =
      tasks.AddTask(dependency, parthenon::Update::EstimateTimestep<MeshData<Real>>, next);
  if (next->GetParentPointer()->adaptive)
    return tasks.AddTask(timestep, parthenon::Refinement::Tag<MeshData<Real>>, next);
  return timestep;
}

TaskID AddStandardMeshFinalization(TaskID dependency, TaskID receive_start, TaskList& tasks,
                                   std::shared_ptr<MeshData<Real>>& next,
                                   const StageContext& stage) {
  auto* mesh = next->GetParentPointer();
  const auto boundaries =
      parthenon::AddBoundaryExchangeTasks(dependency | receive_start, tasks, next, mesh->multilevel);
  const auto derived =
      tasks.AddTask(boundaries, parthenon::Update::FillDerived<MeshData<Real>>, next.get());
  return AddFinalStageTasks(derived, tasks, next.get(), stage);
}

TaskAssembly::TaskAssembly(ParameterInput* pin, Mesh* mesh,
                           parthenon::LowStorageIntegrator& integrator)
    : flux_task_(app::GetBlockFluxTask(pin->GetString("parthenon/job", "problem_id"))),
      flux_correction_task_(
          app::GetBlockFluxCorrectionTask(pin->GetString("parthenon/job", "problem_id"))),
      source_task_(app::GetBlockSourceTask(pin->GetString("parthenon/job", "problem_id"))) {
  pin->CheckRequired("parthenon/mesh", "ix1_bc");
  pin->CheckRequired("parthenon/mesh", "ox1_bc");
  pin->CheckDesired("parthenon/mesh", "refinement");
  pin->CheckDesired("parthenon/mesh", "numlevel");
  for (const auto* contribution : RegisteredContributions(mesh)) {
    if (contribution->configure != nullptr)
      contribution->configure(mesh, integrator);
  }
}

void TaskAssembly::PostStep(Mesh* mesh, const parthenon::SimTime& time) const {
  for (const auto* contribution : RegisteredContributions(mesh)) {
    if (contribution->post_step != nullptr)
      contribution->post_step(mesh, time);
  }
}

TaskCollection TaskAssembly::Build(Mesh* pmesh,
                                   parthenon::LowStorageIntegrator* integrator,
                                   const parthenon::SimTime& tm, BlockList_t& blocks,
                                   const int stage) {
  using namespace parthenon::Update;
  TaskCollection collection;
  TaskID none(0);

  const StageContext stage_context{stage,
                                   integrator->nstages,
                                   integrator->beta[stage - 1],
                                   integrator->gam0[stage - 1],
                                   integrator->gam1[stage - 1],
                                   integrator->dt,
                                   tm.time + integrator->c[stage - 1] * integrator->dt};
  const auto beta = stage_context.beta;
  const auto gam0 = stage_context.gamma0;
  const auto gam1 = stage_context.gamma1;
  const auto dt = stage_context.dt;
  const auto source_time = stage_context.source_time;
  const auto& stage_name = integrator->stage_name;
  const auto partitions = pmesh->GetDefaultBlockPartitions();

  StageBuildContext context{pmesh, integrator, &tm, &blocks, stage, flux_task_, flux_correction_task_};
  if (const auto* contribution = SelectStageContribution(RegisteredContributions(pmesh), context))
    return contribution->build(context);

  // A synchronized spacetime evolves only through its registered contribution.
  PARTHENON_REQUIRE(!geometry::ConfiguredGeometryIsSynchronized(),
                    "synchronized geometry requires a registered stage contribution");

  const auto& packages = pmesh->packages;
  const auto* fluid = FindStageExtension<FluidStageExtension>(packages, fluid_stage_extension_key);
  const auto* passive =
      FindStageExtension<PassiveStageExtension>(packages, passive_stage_extension_key);
  const auto stage_sources =
      FindStageExtensions<StageSourceExtension>(packages, stage_source_extension_key);
  const bool heats_passive = passive != nullptr && passive->applies_heating;

  auto& receive_region = collection.AddRegion(partitions.size());
  for (std::size_t partition = 0; partition < partitions.size(); ++partition) {
    auto& tasks = receive_region[partition];
    auto& base = pmesh->mesh_data.Add("base", partitions[partition]);
    auto& current = pmesh->mesh_data.Add(stage_name[stage - 1], base);
    auto& next = pmesh->mesh_data.Add(stage_name[stage], base);
    pmesh->mesh_data.Add("dUdt", base);
    tasks.AddTask(none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, next);
    if (pmesh->multilevel)
      tasks.AddTask(none, parthenon::StartReceiveFluxCorrections, current);
  }

  auto& flux_region = collection.AddRegion(blocks.size());
  assert(flux_region.size() == blocks.size());
  for (std::size_t block = 0; block < blocks.size(); ++block) {
    auto& tasks = flux_region[block];
    auto& current = blocks[block]->meshblock_data.Get(stage_name[stage - 1]);
    auto& base = blocks[block]->meshblock_data.Get("base");
    auto flux = passive != nullptr
                    ? tasks.AddTask(none, passive->calculate_fluid_fluxes_block, current,
                                    std::string(passive->conserved_field))
                    : tasks.AddTask(none, flux_task_, current);
    if (flux_correction_task_ != nullptr) {
      flux = passive != nullptr
                 ? tasks.AddTask(flux, passive->apply_fluid_flux_correction_block, current, base,
                                 gam0, gam1, beta * dt, std::string(passive->conserved_field))
                 : tasks.AddTask(flux, flux_correction_task_, current, base, gam0, gam1, beta * dt);
    }
    if (passive != nullptr)
      flux = tasks.AddTask(flux, passive->calculate_fluxes_block, current);
    for (const auto* diffusion : FindStageExtensions<plugin_api::DiffusionOperator>(
             packages, plugin_api::diffusion_operator_key)) {
      flux = tasks.AddTask(flux, diffusion->add_fluxes, current);
    }
  }

  auto& update_region = collection.AddRegion(partitions.size());
  for (std::size_t partition = 0; partition < partitions.size(); ++partition) {
    auto& tasks = update_region[partition];
    auto& base = pmesh->mesh_data.Add("base", partitions[partition]);
    auto& current = pmesh->mesh_data.Add(stage_name[stage - 1], base);
    auto& next = pmesh->mesh_data.Add(stage_name[stage], base);

    const auto flux_corrections =
        parthenon::AddFluxCorrectionTasks(none, tasks, current, pmesh->multilevel);
    tasks.AddTask(flux_corrections, UpdateCellConservedMeshTask, current.get(), base.get(), gam0,
                  gam1, beta * dt, next.get());
    if (passive != nullptr)
      tasks.AddTask(flux_corrections, passive->update_conserved, current.get(), base.get(), gam0,
                    gam1, beta * dt, next.get());
    if (fluid != nullptr && fluid->update_face_fields != nullptr)
      tasks.AddTask(flux_corrections, fluid->update_face_fields, current.get(), base.get(), gam0,
                    gam1, beta * dt, next.get());
    // AthenaK applies stage source terms before sending the updated conserved and
    // face-centred state or invoking physical/user boundaries.  The receive was
    // posted in receive_region above; its matching exchange is deliberately
    // deferred until after source_region.
  }

  if (fluid != nullptr && fluid->has_stage_sources(packages)) {
    auto& source_region = collection.AddRegion(blocks.size());
    assert(source_region.size() == blocks.size());
    for (std::size_t block = 0; block < blocks.size(); ++block) {
      auto& tasks = source_region[block];
      auto& current = blocks[block]->meshblock_data.Get(stage_name[stage - 1]);
      auto& next = blocks[block]->meshblock_data.Get(stage_name[stage]);
      const auto stage_primitives =
          tasks.AddTask(none, CopyCellDerivedBlockTask, current.get(), next.get());
      if (source_task_ != nullptr)
        tasks.AddTask(stage_primitives, source_task_, next, beta * dt);
    }
  }

  if (!stage_sources.empty()) {
    // Registered stage sources run as packed MeshData tasks.  TaskCollection
    // regions serialize them after every block-local fluid source and before the
    // updated state is exchanged at MeshBlock boundaries.
    auto& source_extension_region = collection.AddRegion(partitions.size());
    for (std::size_t partition = 0; partition < partitions.size(); ++partition) {
      auto& tasks = source_extension_region[partition];
      auto& base = pmesh->mesh_data.Add("base", partitions[partition]);
      auto& next = pmesh->mesh_data.Add(stage_name[stage], base);
      for (const auto* source : stage_sources)
        source->add_source_task(none, none, tasks, next.get(), source_time, beta * dt,
                                stage == integrator->nstages);
    }
  }

  // Exchange the fully updated stage state once sources have been applied.  This
  // consumes the receives posted at the beginning of the stage and matches
  // AthenaK's SendU/SendB -> physical boundary ordering.
  auto& post_exchange_region = collection.AddRegion(partitions.size());
  for (std::size_t partition = 0; partition < partitions.size(); ++partition) {
    auto& tasks = post_exchange_region[partition];
    auto& base = pmesh->mesh_data.Add("base", partitions[partition]);
    auto& next = pmesh->mesh_data.Add(stage_name[stage], base);
    if (pmesh->multilevel)
      parthenon::AddBoundaryExchangeTasks(none, tasks, next, true);
    else
      AddUniformBoundaryExchangeTasks(none, tasks, next);
  }

  auto& finish_region = collection.AddRegion(blocks.size());
  assert(finish_region.size() == blocks.size());
  for (std::size_t block = 0; block < blocks.size(); ++block) {
    auto& tasks = finish_region[block];
    auto& next = blocks[block]->meshblock_data.Get(stage_name[stage]);
    const auto derived =
        tasks.AddTask(none, parthenon::Update::FillDerived<MeshBlockData<Real>>, next.get());
    if (!heats_passive && stage == integrator->nstages) {
      tasks.AddTask(derived,
                    parthenon::Update::EstimateTimestep<MeshBlockData<Real>>,
                    next.get());
      if (pmesh->adaptive) {
        tasks.AddTask(derived, parthenon::Refinement::Tag<MeshBlockData<Real>>, next.get());
      }
    }
  }
  if (heats_passive) {
    auto& passive_region = collection.AddRegion(partitions.size());
    for (std::size_t partition = 0; partition < partitions.size(); ++partition) {
      auto& tasks = passive_region[partition];
      auto& base = pmesh->mesh_data.Add("base", partitions[partition]);
      auto& current = pmesh->mesh_data.Add(stage_name[stage - 1], base);
      auto& next = pmesh->mesh_data.Add(stage_name[stage], base);
      auto& passive_boundary = pmesh->mesh_data.AddShallow(
          "passive_heated_" + stage_name[stage], next,
          std::vector<std::string>{std::string(passive->conserved_field)});
      const auto heating = tasks.AddTask(none, passive->apply_heating, current.get(), next.get());
      const auto conserved = tasks.AddTask(heating, passive->primitive_to_conserved, next.get());
      const auto passive_receive =
          tasks.AddTask(conserved, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>,
                        passive_boundary);
      const auto passive_boundaries = parthenon::AddBoundaryExchangeTasks(
          conserved | passive_receive, tasks, passive_boundary, pmesh->multilevel,
          [](TaskID dependency, TaskList*, std::shared_ptr<MeshData<Real>>, bool) {
            return dependency;
          });
      const auto physical =
          tasks.AddTask(passive_boundaries, passive->apply_physical_boundaries, next.get());
      const auto canonical = tasks.AddTask(physical, passive->conserved_to_primitive, next.get());
      if (stage == integrator->nstages) {
        const auto timestep =
            tasks.AddTask(canonical,
                          parthenon::Update::EstimateTimestep<MeshData<Real>>,
                          next.get());
        if (pmesh->adaptive)
          tasks.AddTask(timestep, parthenon::Refinement::Tag<MeshData<Real>>, next.get());
      }
    }
  }
  return collection;
}

} // namespace pangu::driver
