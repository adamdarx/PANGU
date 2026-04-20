// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/recovery module defines tetrad.cc responsibilities for the Pangu
// runtime. It centers on transformation between SRMHD and GRMHD to better re-use SRMHD 
// recovery schemes with the purpose of accelerating this essential process in GRMHD solver.
#include "tetrad.h"

#include <memory>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"

parthenon::TaskStatus TransformToSRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource) {
  // TODO:  An SRMHD Invertor is needed before this function can be implemented.
}

parthenon::TaskStatus TransformToGRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource) {
  // TODO:  An SRMHD Invertor is needed before this function can be implemented.
}