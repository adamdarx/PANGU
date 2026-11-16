#pragma once
#include "pangu/plugin_api/plugin.h"

namespace pangu::plugins::scalar_field {
// Coupled EMS uses a common action and selectable coordinate formulations.
std::shared_ptr<parthenon::StateDescriptor>
InitializeCoupledEMS(plugin::InitContext &context);
std::shared_ptr<parthenon::StateDescriptor>
InitializePGEMS(plugin::InitContext &context);
std::shared_ptr<parthenon::StateDescriptor>
InitializeCartesianEMS(plugin::InitContext &context);
std::shared_ptr<parthenon::StateDescriptor>
InitializeSphericalEMS(plugin::InitContext &context);
} // namespace pangu::plugins::scalar_field
