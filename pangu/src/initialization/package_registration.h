// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_INITIALIZATION_PACKAGEREGISTRATION_H
#define PANGU_SRC_INITIALIZATION_PACKAGEREGISTRATION_H

#include <memory>
#include <parthenon/package.hpp>

namespace core {
// Builds and registers the core package descriptor and fields.
std::shared_ptr<parthenon::StateDescriptor> Initialize(
    parthenon::ParameterInput *pin);
}  // namespace core

namespace metric {
// Builds and registers the metric package descriptor and fields.
std::shared_ptr<parthenon::StateDescriptor> Initialize(
    parthenon::ParameterInput *pin);
}  // namespace metric

#endif  // PANGU_SRC_INITIALIZATION_PACKAGEREGISTRATION_H
