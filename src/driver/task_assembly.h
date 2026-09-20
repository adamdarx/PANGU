#ifndef PANGU_DRIVER_TASK_ASSEMBLY_H_
#define PANGU_DRIVER_TASK_ASSEMBLY_H_

#include <memory>
#include <string>
#include <vector>

#include <parthenon/parthenon.hpp>

#include "app/problem_registry.h"

namespace pangu::driver {

struct StageContext {
  int stage;
  int stages;
  parthenon::Real beta;
  parthenon::Real gamma0;
  parthenon::Real gamma1;
  parthenon::Real dt;
  parthenon::Real source_time;

  constexpr bool IsFinal() const { return stage == stages; }
  constexpr parthenon::Real StageDt() const { return beta * dt; }
};

parthenon::TaskID AddUniformBoundaryExchangeTasks(
    parthenon::TaskID dependency, parthenon::TaskList& tasks,
    std::shared_ptr<parthenon::MeshData<parthenon::Real>>& data);

class TaskAssembly {
public:
  TaskAssembly(parthenon::ParameterInput* input, parthenon::Mesh* mesh,
               parthenon::LowStorageIntegrator& integrator);

  parthenon::TaskCollection Build(parthenon::Mesh* mesh,
                                  parthenon::LowStorageIntegrator* integrator,
                                  const parthenon::SimTime& time,
                                  parthenon::BlockList_t& blocks, int stage);
  void PostStep(parthenon::Mesh* mesh, const parthenon::SimTime& time) const;

private:
  app::BlockFluxTask flux_task_;
  app::BlockFluxCorrectionTask flux_correction_task_;
  app::BlockSourceTask source_task_;
};

parthenon::TaskID AddStandardMeshFinalization(
    parthenon::TaskID dependency, parthenon::TaskID receive_start,
    parthenon::TaskList& tasks,
    std::shared_ptr<parthenon::MeshData<parthenon::Real>>& next,
    const StageContext& stage);

parthenon::TaskID AddFinalStageTasks(
    parthenon::TaskID dependency, parthenon::TaskList& tasks,
    parthenon::MeshData<parthenon::Real>* next, const StageContext& stage);

// Cell-centred fields advanced by the conserved RK update.
std::vector<std::string> CellConservedVariables(parthenon::Mesh* mesh);

parthenon::TaskStatus UpdateCellConservedMeshTask(
    parthenon::MeshData<parthenon::Real>* current,
    parthenon::MeshData<parthenon::Real>* base, parthenon::Real gamma0,
    parthenon::Real gamma1, parthenon::Real beta_dt,
    parthenon::MeshData<parthenon::Real>* next);

parthenon::TaskStatus CopyCellDerivedMeshTask(
    parthenon::MeshData<parthenon::Real>* current,
    parthenon::MeshData<parthenon::Real>* next);



} // namespace pangu::driver

#endif
