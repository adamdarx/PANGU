// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/recovery module defines invertor.h responsibilities for the Pangu
// runtime. It centers on memory to express core data flow, keep interfaces readable, and
// preserve predictable behavior across task coordination, recovery paths, and
// performance-sensitive execution.

#ifndef PANGU_SRC_RECOVERY_INVERTOR_H
#define PANGU_SRC_RECOVERY_INVERTOR_H
#include <memory>
#include <parthenon/package.hpp>

// Recovers SRMHD primitive variables from conservative variables.
parthenon::TaskStatus RecoverySRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource);

// Recovers GRMHD primitive variables from conservative and metric variables.
parthenon::TaskStatus RecoveryGRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource);

#endif  // PANGU_SRC_RECOVERY_INVERTOR_H
