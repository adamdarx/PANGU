// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/initialization module defines package_registration.h
// responsibilities for the Pangu runtime. It centers on memory to express core data flow,
// keep interfaces readable, and preserve predictable behavior across task coordination,
// recovery paths, and performance-sensitive execution.

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
