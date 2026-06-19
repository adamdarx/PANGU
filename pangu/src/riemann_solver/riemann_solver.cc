// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "riemann_solver/riemann_solver.h"

#include <parthenon/package.hpp>
#include <stdexcept>
#include <string>

#include "riemann_solver/hll.h"
#include "riemann_solver/laxf.h"

parthenon::TaskStatus CalculateFluxes(parthenon::MeshData<parthenon::Real> *md) {
  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  const auto package_core = pmb0->packages.Get("core");
  const auto &solver_name = package_core->Param<std::string>("riemann_solver");

  if (solver_name == "laxf") {
    return CalculateLAXF(md);
  }
  if (solver_name == "hll") {
    return CalculateHLL(md);
  }

  throw std::invalid_argument("Unknown Riemann solver: " + solver_name);
}
