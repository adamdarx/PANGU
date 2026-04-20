// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/initialization module defines timestep_estimation.h responsibilities
// for the Pangu runtime. It centers on parthenon to express core data flow, keep interfaces
// readable, and preserve predictable behavior across task coordination, recovery paths, and
// performance-sensitive execution.

#ifndef PANGU_SRC_INITIALIZATION_TIMESTEPESTIMATION_H
#define PANGU_SRC_INITIALIZATION_TIMESTEPESTIMATION_H

#include <parthenon/package.hpp>

// Estimates the next stable timestep for one mesh block.
parthenon::Real EstimateTimestepBlock(
    parthenon::MeshBlockData<parthenon::Real> *resource);

#endif  // PANGU_SRC_INITIALIZATION_TIMESTEPESTIMATION_H
