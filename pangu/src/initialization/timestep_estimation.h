// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_INITIALIZATION_TIMESTEPESTIMATION_H
#define PANGU_SRC_INITIALIZATION_TIMESTEPESTIMATION_H

#include <parthenon/package.hpp>

// Estimates the next stable timestep for one mesh block.
parthenon::Real EstimateTimestepBlock(
    parthenon::MeshBlockData<parthenon::Real> *resource);

#endif  // PANGU_SRC_INITIALIZATION_TIMESTEPESTIMATION_H
