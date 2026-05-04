// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_FIXER_PRIMITIVEFIXER_H
#define PANGU_SRC_FIXER_PRIMITIVEFIXER_H

#include <memory>
#include <parthenon/package.hpp>

// Applies GRMHD primitive floors and velocity ceilings using metric data.
parthenon::TaskStatus FixPrimitive(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource);

#endif  // PANGU_SRC_FIXER_PRIMITIVEFIXER_H
