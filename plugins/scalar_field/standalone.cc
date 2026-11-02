// Minimal local reproduction executable. Uses exactly the same package, CUDA
// kernels, RK4 contribution and Parthenon boundary exchange as the full plugin.
#include "driver/contributor.h"
#include "plugin.h"
#include <iostream>

namespace {
class ScalarDriver : public parthenon::driver::prelude::MultiStageDriver {
  using Base = parthenon::driver::prelude::MultiStageDriver;
  const pangu::driver::StageContribution *contribution_;

public:
  ScalarDriver(parthenon::ParameterInput *pin, parthenon::ApplicationInput *app,
               parthenon::Mesh *mesh)
      : Base(pin, app, mesh) {
    contribution_ = &mesh->packages.Get("plugin.scalar_field")
                         ->Param<pangu::driver::StageContribution>(std::string(
                             pangu::driver::stage_contribution_key));
    contribution_->configure(mesh, *integrator);
  }
  parthenon::TaskListStatus Step() override {
    const auto status = Base::Step();
    if (status == parthenon::TaskListStatus::complete && contribution_->post_step)
      contribution_->post_step(pmesh, tm);
    return status;
  }
  parthenon::TaskCollection MakeTaskCollection(parthenon::BlockList_t &blocks,
                                               int stage) override {
    pangu::driver::StageBuildContext context{
        pmesh, integrator.get(), &tm, &blocks, stage, nullptr, nullptr};
    return contribution_->build(context);
  }
};
} // namespace
int main(int argc, char **argv) {
  parthenon::ParthenonManager manager;
  try {
    manager.app_input->ProcessPackages =
        [](std::unique_ptr<parthenon::ParameterInput> &pin) {
          parthenon::Packages_t packages;
          pangu::plugin::InitContext context{*pin, packages};
          packages.Add(
              pangu::plugins::scalar_field::Plugin::Initialize(context));
          return packages;
        };
    manager.app_input->ProblemGenerator = [](parthenon::MeshBlock *,
                                             parthenon::ParameterInput *) {};
    auto status = manager.ParthenonInitEnv(argc, argv);
    if (status != parthenon::ParthenonStatus::ok) {
      manager.ParthenonFinalize();
      return status == parthenon::ParthenonStatus::complete ? 0 : 1;
    }
    if (parthenon::Globals::my_rank == 0)
      std::cout << "Scalar execution space: "
                << parthenon::DevExecSpace::name() << '\n';
    manager.ParthenonInitPackagesAndMesh();
    int rc;
    {
      ScalarDriver driver(manager.pinput.get(), manager.app_input.get(),
                          manager.pmesh.get());
      rc = driver.Execute() == parthenon::DriverStatus::complete ? 0 : 1;
    }
    manager.ParthenonFinalize();
    return rc;
  } catch (const std::exception &e) {
    std::cerr << "Scalar reproduction failed: " << e.what() << '\n';
    if (Kokkos::is_initialized())
      manager.ParthenonFinalize();
    return 1;
  }
}
