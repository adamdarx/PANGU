// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_RIEMANNSOLVER_ELECTRONHEATING_H
#define PANGU_SRC_RIEMANNSOLVER_ELECTRONHEATING_H

#include <memory>
#include <parthenon/package.hpp>

// Applies electron heating using advected and recovered entropy values.
parthenon::TaskStatus ApplyElectronHeating(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource);

#endif  // PANGU_SRC_RIEMANNSOLVER_ELECTRONHEATING_H