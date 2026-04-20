// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/recovery module defines tetrad.h responsibilities for the Pangu
// runtime. It centers on transformation between SRMHD and GRMHD to better re-use SRMHD 
// recovery schemes with the purpose of accelerating this essential process in GRMHD solver.

#ifndef PANGU_SRC_RECOVERY_TETRAD_H
#define PANGU_SRC_RECOVERY_TETRAD_H
#include <memory>
#include <parthenon/package.hpp>

// Recovers SRMHD primitive variables from conservative variables.
parthenon::TaskStatus TransformToSRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource);

// Recovers GRMHD primitive variables from conservative and metric variables.
parthenon::TaskStatus TransformToGRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource);

#endif  // PANGU_SRC_RECOVERY_TETRAD_H