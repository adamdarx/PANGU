#pragma once
#include <memory>
#include <vector>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon::driver::prelude;

class Simulator : public MultiStageDriver {
public:
    Simulator(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm);
    TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);
};

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin);
