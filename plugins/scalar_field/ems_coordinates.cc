#include "ems_coupled.h"
#include "geometry_assembly.h"
#include "pangu/plugin_api/scalar_field.h"

namespace pangu::plugins::scalar_field {
std::shared_ptr<parthenon::StateDescriptor>
InitializeCoupledEMS(plugin::InitContext& context) {
  const auto coordinates = plugin_api::scalar_field::CoordinateName(context.input);
  PARTHENON_REQUIRE(coordinates == "pg" || coordinates == "cartesian" || coordinates == "spherical",
                    "coupled EMS coordinates must be pg, cartesian or spherical");
  if (coordinates == "pg") return InitializePGEMS(context);
  if (coordinates == "spherical") return InitializeSphericalEMS(context);
#ifndef PANGU_SCALAR_REPRODUCTION_DRIVER
  if constexpr (geometry::ConfiguredGeometryIsSynchronized())
    return InitializeCartesianEMS(context);
#endif
  PARTHENON_FAIL("Cartesian EMS requires the full pangu executable built with METRIC=z4c MODE=sync");
  return nullptr;
}
} // namespace pangu::plugins::scalar_field
