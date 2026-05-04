// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_RIEMANNSOLVER_LAXFRIEDRICHSFLUX_H
#define PANGU_SRC_RIEMANNSOLVER_LAXFRIEDRICHSFLUX_H

#include <memory>
#include <parthenon/package.hpp>

// Computes GRMHD numerical fluxes with the Lax-Friedrichs solver.
parthenon::TaskStatus CalculateFluxes(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &init_resource);

#endif  // PANGU_SRC_RIEMANNSOLVER_LAXFRIEDRICHSFLUX_H
