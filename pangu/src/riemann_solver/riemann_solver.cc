// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "riemann_solver/riemann_solver.h"

#include <memory>
#include <parthenon/package.hpp>
#include <stdexcept>
#include <string>

#include "riemann_solver/hll.h"
#include "riemann_solver/hlld.h"
#include "riemann_solver/laxf.h"

parthenon::TaskStatus CalculateFluxes(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &init_resource) {
  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
  const auto &solver_name = package_core->Param<std::string>("riemann_solver");

  if (solver_name == "laxf") {
    return CalculateLAXF(resource, init_resource);
  }
  if (solver_name == "hll") {
    return CalculateHLL(resource, init_resource);
  }
  if (solver_name == "hlld") {
    return CalculateHLLD(resource, init_resource);
  }

  throw std::invalid_argument("Unknown Riemann solver: " + solver_name);
}
