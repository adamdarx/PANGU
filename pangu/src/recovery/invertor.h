// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_RECOVERY_INVERTOR_H
#define PANGU_SRC_RECOVERY_INVERTOR_H
#include <memory>
#include <parthenon/package.hpp>

// Recovers GRMHD primitive variables from conservative and metric variables.
parthenon::TaskStatus Recovery(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource);

#endif  // PANGU_SRC_RECOVERY_INVERTOR_H
