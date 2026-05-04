// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_MESH_REFINEMENTCRITERIA_H
#define PANGU_SRC_MESH_REFINEMENTCRITERIA_H

#include <parthenon/package.hpp>

// Returns the AMR action for a block from q-factor bounds.
parthenon::AmrTag CheckRefinement(
    parthenon::MeshBlockData<parthenon::Real> *resource);

#endif  // PANGU_SRC_MESH_REFINEMENTCRITERIA_H
