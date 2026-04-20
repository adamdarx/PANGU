// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/numerical_correction module defines recovery_fixer.h
// responsibilities for the Pangu runtime. It centers on memory to express core data flow,
// keep interfaces readable, and preserve predictable behavior across task coordination,
// recovery paths, and performance-sensitive execution.

#ifndef PANGU_SRC_NUMERICALCORRECTION_RECOVERYFIXER_H
#define PANGU_SRC_NUMERICALCORRECTION_RECOVERYFIXER_H

#include <memory>
#include <parthenon/package.hpp>

// Repairs cells where primitive recovery failed in the previous stage.
parthenon::TaskStatus FixRecovery(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource);

#endif  // PANGU_SRC_NUMERICALCORRECTION_RECOVERYFIXER_H
