#ifndef PANGU_DRIVER_DRIVER_H_
#define PANGU_DRIVER_DRIVER_H_

#include <memory>

#include <parthenon/parthenon.hpp>

namespace pangu::driver {

class TaskAssembly;

using BaseDriver = parthenon::driver::prelude::MultiStageDriver;
using TaskCollection = parthenon::driver::prelude::TaskCollection;
using TaskListStatus = parthenon::driver::prelude::TaskListStatus;
using BlockList = parthenon::driver::prelude::BlockList_t;

class Driver : public BaseDriver {
public:
  Driver(parthenon::ParameterInput* pin, parthenon::ApplicationInput* app_input,
         parthenon::Mesh* mesh);
  ~Driver();

  TaskListStatus Step() override;
  TaskCollection MakeTaskCollection(BlockList& blocks, int stage) override;

private:
  std::unique_ptr<TaskAssembly> task_assembly_;
};

} // namespace pangu::driver

#endif
