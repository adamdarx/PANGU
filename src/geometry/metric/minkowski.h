#ifndef PANGU_GEOMETRY_METRIC_MINKOWSKI_H_
#define PANGU_GEOMETRY_METRIC_MINKOWSKI_H_

#include <cmath>

#include "geometry/geometry.h"
#include "utils/error_checking.hpp"

namespace pangu::geometry::metric {

// Internal, zero-storage flat-spacetime façade used by PHYSICS=sr.  It is not
// a user-selectable GR metric: CMake ignores METRIC and MODE for SR builds.
struct Minkowski {
  struct Parameters {};

  static constexpr Symmetry symmetry = Symmetry::constant;
  static constexpr bool supports_excision = false;
  static constexpr bool unit_determinant = true;
  static constexpr bool unit_coordinate_light_bound = true;

  static Parameters Configure(const Background background, const parthenon::Real,
                              parthenon::ParameterInput*) {
    PARTHENON_REQUIRE(background == Background::minkowski,
                      "PHYSICS=sr requires Cartesian Minkowski coordinates");
    return {};
  }

  KOKKOS_INLINE_FUNCTION static MetricPoint Calculate(const parthenon::Real,
                                                       const parthenon::Real,
                                                       const parthenon::Real, const Location,
                                                       const Parameters&) {
    MetricPoint metric{};
    metric.lower[0][0] = -1.0;
    metric.upper[0][0] = -1.0;
    for (int axis = 1; axis < 4; ++axis) {
      metric.lower[axis][axis] = 1.0;
      metric.upper[axis][axis] = 1.0;
    }
    metric.lapse = 1.0;
    metric.spatial_det = 1.0;
    metric.gdet = 1.0;
    return metric;
  }

  KOKKOS_INLINE_FUNCTION static MetricDerivatives CalculateDerivatives(
      const parthenon::Real, const parthenon::Real, const parthenon::Real, const Parameters&) {
    return {};
  }

  KOKKOS_INLINE_FUNCTION static SphericalKerrSchildPoint
  SphericalCoordinates(const parthenon::Real x, const parthenon::Real y,
                       const parthenon::Real z, const Parameters&) {
    const parthenon::Real radius = sqrt(x * x + y * y + z * z);
    const parthenon::Real theta =
        radius > 0.0 ? acos(fmax(-1.0, fmin(1.0, z / radius))) : 0.0;
    return {radius, theta, atan2(y, x)};
  }

  KOKKOS_INLINE_FUNCTION static void BoyerLindquistSpatialToNative(
      const parthenon::Real x, const parthenon::Real y, const parthenon::Real z,
      const parthenon::Real spherical[3], parthenon::Real cartesian[3], const Parameters&) {
    const parthenon::Real radius = sqrt(x * x + y * y + z * z);
    const parthenon::Real cylindrical2 = x * x + y * y;
    const parthenon::Real cylindrical = sqrt(cylindrical2);
    const parthenon::Real inverse_radius = radius > 0.0 ? 1.0 / radius : 0.0;
    const parthenon::Real inverse_cylindrical = cylindrical > 0.0 ? 1.0 / cylindrical : 0.0;
    cartesian[0] = spherical[0] * x * inverse_radius +
                   spherical[1] * x * z * inverse_radius * inverse_cylindrical -
                   spherical[2] * y;
    cartesian[1] = spherical[0] * y * inverse_radius +
                   spherical[1] * y * z * inverse_radius * inverse_cylindrical +
                   spherical[2] * x;
    cartesian[2] = spherical[0] * z * inverse_radius - spherical[1] * cylindrical;
  }

  KOKKOS_INLINE_FUNCTION static void AzimuthalCovectorToNative(
      const parthenon::Real x, const parthenon::Real y, const parthenon::Real,
      const parthenon::Real azimuthal, parthenon::Real cartesian[3], const Parameters&) {
    const parthenon::Real cylindrical2 = x * x + y * y;
    const parthenon::Real inverse = cylindrical2 > 0.0 ? 1.0 / cylindrical2 : 0.0;
    cartesian[0] = -azimuthal * y * inverse;
    cartesian[1] = azimuthal * x * inverse;
    cartesian[2] = 0.0;
  }
};

} // namespace pangu::geometry::metric

#endif
