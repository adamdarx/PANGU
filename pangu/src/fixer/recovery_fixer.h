// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_FIXER_RECOVERYFIXER_H
#define PANGU_SRC_FIXER_RECOVERYFIXER_H

#include <memory>
#include <parthenon/package.hpp>

// Repairs cells where primitive recovery failed in the previous stage.
parthenon::TaskStatus FixRecovery(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource);

#endif  // PANGU_SRC_FIXER_RECOVERYFIXER_H
