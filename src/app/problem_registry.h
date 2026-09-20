#ifndef PANGU_APP_PROBLEM_REGISTRY_H_
#define PANGU_APP_PROBLEM_REGISTRY_H_

#include <memory>
#include <string>
#include <vector>

#include <parthenon/parthenon.hpp>

namespace pangu::app {

using BlockFluxTask =
    parthenon::TaskStatus (*)(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>&);
using BlockSourceTask = parthenon::TaskStatus (*)(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>&, parthenon::Real);
using BlockFluxCorrectionTask =
    parthenon::TaskStatus (*)(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>&,
                              std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>&,
                              parthenon::Real, parthenon::Real, parthenon::Real);
using PackageInitializer =
  std::shared_ptr<parthenon::package::prelude::StateDescriptor> (*)(parthenon::ParameterInput*);

void ConfigureProblem(const std::string& problem_id, parthenon::ApplicationInput* app_input);
std::vector<std::string> RegisteredProblems();
BlockFluxTask GetBlockFluxTask(const std::string& problem_id);
BlockFluxCorrectionTask GetBlockFluxCorrectionTask(const std::string& problem_id);
BlockSourceTask GetBlockSourceTask(const std::string& problem_id);
PackageInitializer GetPackageInitializer(const std::string& problem_id);
PackageInitializer GetCoupledPackageInitializer(const std::string& problem_id);

} // namespace pangu::app

#endif
