// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "parthenon_manager.hpp"
#include "task_list/task_list.h"

int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  pman.app_input->ProcessPackages = ProcessPackages;
  pman.app_input->ProblemGenerator = ProblemGenerator;
  pman.app_input->MeshPostInitialization = MeshPostInitialization;
  pman.app_input->RegisterDefaultReflectingBoundaryConditions();

  // Register "user" boundary condition — standard function is a no-op;
  // the actual boundary values are filled by package-level UserBoundaryFunctions.
  using parthenon::BoundaryFace;
  const std::string USER_BC = "user";
  auto user_noop = [](std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &,
                      bool) {};
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::inner_x1, USER_BC, user_noop);
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::outer_x1, USER_BC, user_noop);
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::inner_x2, USER_BC, user_noop);
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::outer_x2, USER_BC, user_noop);
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::inner_x3, USER_BC, user_noop);
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::outer_x3, USER_BC, user_noop);

  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }

  pman.ParthenonInitPackagesAndMesh();
  {
    Simulator driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());

    auto driver_status = driver.Execute();
  }
  pman.ParthenonFinalize();

  return (0);
}
