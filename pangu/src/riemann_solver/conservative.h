// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/riemann_solver module defines conservative.h responsibilities for
// the Pangu runtime. It centers on memory to express core data flow, keep interfaces
// readable, and preserve predictable behavior across task coordination, recovery paths, and
// performance-sensitive execution.

#ifndef PANGU_SRC_RIEMANNSOLVER_CONSERVATIVE_H
#define PANGU_SRC_RIEMANNSOLVER_CONSERVATIVE_H

#include <memory>
#include <parthenon/package.hpp>

// Builds SRMHD conservative variables from primitive variables.
parthenon::TaskStatus CalculateConservativeSRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource);

// Builds GRMHD conservative variables from primitive and metric variables.
parthenon::TaskStatus CalculateConservativeGRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource);

#endif  // PANGU_SRC_RIEMANNSOLVER_CONSERVATIVE_H
