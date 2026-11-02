#pragma once
#include "pangu/plugin_api/plugin.h"

namespace pangu::plugins::scalar_field {
// Fully nonlinear spherical Einstein-Maxwell-scalar evolution in PG slicing.
// This is a separate geometry formulation; it is not the 3D Z4c matter path.
std::shared_ptr<parthenon::StateDescriptor>
InitializeCoupledEMS(plugin::InitContext &context);
} // namespace pangu::plugins::scalar_field
