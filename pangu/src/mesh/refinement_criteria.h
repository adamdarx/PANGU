// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/mesh module defines refinement_criteria.h responsibilities for the
// Pangu runtime. It centers on parthenon to express core data flow, keep interfaces
// readable, and preserve predictable behavior across task coordination, recovery paths, and
// performance-sensitive execution.

#ifndef PANGU_SRC_MESH_REFINEMENTCRITERIA_H
#define PANGU_SRC_MESH_REFINEMENTCRITERIA_H

#include <parthenon/package.hpp>

// Returns the AMR action for a block from q-factor bounds.
parthenon::AmrTag CheckRefinement(
    parthenon::MeshBlockData<parthenon::Real> *resource);

#endif  // PANGU_SRC_MESH_REFINEMENTCRITERIA_H
