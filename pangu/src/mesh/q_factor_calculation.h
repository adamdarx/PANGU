// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/mesh module defines q_factor_calculation.h responsibilities for the
// Pangu runtime. It centers on memory to express core data flow, keep interfaces readable,
// and preserve predictable behavior across task coordination, recovery paths, and
// performance-sensitive execution.

#ifndef PANGU_SRC_MESH_QFACTORCALCULATION_H
#define PANGU_SRC_MESH_QFACTORCALCULATION_H

#include <memory>
#include <parthenon/package.hpp>

// Computes and stores the MRI quality factor field for a block.
parthenon::TaskStatus CalculateQFactor(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource);

#endif  // PANGU_SRC_MESH_QFACTORCALCULATION_H
