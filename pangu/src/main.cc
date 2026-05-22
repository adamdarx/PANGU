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
