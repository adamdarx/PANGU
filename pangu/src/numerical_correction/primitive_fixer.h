// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/numerical_correction module defines primitive_fixer.h
// responsibilities for the Pangu runtime. It centers on memory to express core data flow,
// keep interfaces readable, and preserve predictable behavior across task coordination,
// recovery paths, and performance-sensitive execution.

#ifndef PANGU_SRC_NUMERICALCORRECTION_PRIMITIVEFIXER_H
#define PANGU_SRC_NUMERICALCORRECTION_PRIMITIVEFIXER_H

#include <memory>
#include <parthenon/package.hpp>

// Applies SRMHD primitive floors and velocity ceilings.
parthenon::TaskStatus FixPrimitiveSRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource);

// Applies GRMHD primitive floors and velocity ceilings using metric data.
parthenon::TaskStatus FixPrimitiveGRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource);

#endif  // PANGU_SRC_NUMERICALCORRECTION_PRIMITIVEFIXER_H
