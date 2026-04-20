// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/riemann_solver module defines geometric_source_term.h
// responsibilities for the Pangu runtime. It centers on memory to express core data flow,
// keep interfaces readable, and preserve predictable behavior across task coordination,
// recovery paths, and performance-sensitive execution.

#ifndef PANGU_SRC_RIEMANNSOLVER_GEOMETRICSOURCETERM_H
#define PANGU_SRC_RIEMANNSOLVER_GEOMETRICSOURCETERM_H

#include <memory>
#include <parthenon/package.hpp>

// Adds geometric source terms to the GRMHD conservative variables.
parthenon::TaskStatus AddSourceGRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    parthenon::Real dt,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource);

#endif  // PANGU_SRC_RIEMANNSOLVER_GEOMETRICSOURCETERM_H
