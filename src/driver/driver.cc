#include "driver/driver.h"

#include "driver/task_assembly.h"

namespace pangu::driver {

Driver::Driver(parthenon::ParameterInput* input,
               parthenon::ApplicationInput* application,
               parthenon::Mesh* mesh)
    : BaseDriver(input, application, mesh),
      task_assembly_(std::make_unique<TaskAssembly>(input, mesh, *integrator)) {}

Driver::~Driver() = default;

TaskListStatus Driver::Step() {
  const auto status = BaseDriver::Step();
  if (status == TaskListStatus::complete)
    task_assembly_->PostStep(pmesh, tm);
  return status;
}

TaskCollection Driver::MakeTaskCollection(BlockList& blocks, const int stage) {
  return task_assembly_->Build(pmesh, integrator.get(), tm, blocks, stage);
}

} // namespace pangu::driver
