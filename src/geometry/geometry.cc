#include "geometry/geometry.h"

#include <cstdint>
#include <string>

#include "geometry_assembly.h"
#include "utils/error_checking.hpp"

namespace pangu::geometry {
using namespace parthenon::package::prelude;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput* pin) {
  auto package = std::make_shared<StateDescriptor>("geometry");
  const std::string name = pin->GetOrAddString("geometry", "background", "minkowski");
  Background background;
  if (name == "minkowski" || name == "sr")
    background = Background::minkowski;
  else if (name == "kerr_schild" || name == "cartesian_ks")
    background = Background::kerr_schild;
  else if (name == "mks" || name == "modified_ks" || name == "spherical_ks")
    background = Background::kerr_schild;
  else
    PARTHENON_FAIL("geometry/background must select Minkowski, Cartesian KS, or MKS");
  const Real spin = pin->GetOrAddReal("geometry", "bh_spin", 0.0);
  const auto excision = ConfigureMetricExcision(pin);
  PARTHENON_REQUIRE(fabs(spin) <= 1.0, "geometry/bh_spin must be in [-1,1]");
  if constexpr (ConfiguredMetricSupportsExcision()) {
    PARTHENON_REQUIRE(excision.radius > 0.0, "geometry/excision_radius must be positive");
    PARTHENON_REQUIRE(
        (excision.density > 0.0 && excision.pressure > 0.0) ||
            (excision.density < 0.0 && excision.pressure < 0.0),
        "geometry/dexcise and geometry/pexcise must either both be positive or both be omitted");
  }
  package->AddParam<int>("background", static_cast<int>(background));
  package->AddParam<std::string>("metric", ConfiguredMetricName());
  package->AddParam<std::string>("mode", ConfiguredModeName());
  package->AddParam<int>("symmetry", static_cast<int>(ConfiguredMetricSymmetry()));
  package->AddParam<bool>("supports_refinement", ConfiguredModeSupportsRefinement());
  package->AddParam<Real>("bh_spin", spin);
  package->AddParam<bool>("excision", excision.enabled);
  package->AddParam<Real>("excision_radius", excision.radius);
  package->AddParam<Real>("excision_density", excision.density);
  package->AddParam<Real>("excision_pressure", excision.pressure);
  package->AddParam<std::uint64_t>("storage_bytes", 0, parthenon::Params::Mutability::Mutable);
  package->AddParam<std::uint64_t>("fingerprint", 0, parthenon::Params::Mutability::Mutable);
  AddConfiguredGeometry(*package, background, spin, pin);
  return package;
}

} // namespace pangu::geometry
