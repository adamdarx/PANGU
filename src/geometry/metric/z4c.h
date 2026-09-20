#ifndef PANGU_GEOMETRY_METRIC_Z4C_H_
#define PANGU_GEOMETRY_METRIC_Z4C_H_

#include "geometry/geometry.h"

namespace pangu::geometry::metric {

// Z4c is a marker for an evolved Cartesian spacetime. It supplies coordinate
// utilities and compile-time capabilities; metric values live in Parthenon
// fields and are accessed only through Geometry<Z4c, SyncMode>.
struct Z4c {
  struct Parameters {
    parthenon::Real chi_psi_power = -4.0;
  };

  static constexpr Symmetry symmetry = Symmetry::general_3d;
  static constexpr bool supports_excision = false;
  static constexpr bool unit_determinant = false;
  static constexpr bool unit_coordinate_light_bound = false;

  static Parameters Configure(Background, parthenon::Real, parthenon::ParameterInput* pin) {
    Parameters parameters{};
    parameters.chi_psi_power = pin->GetOrAddReal(
        "numerical_relativity", "chi_psi_power", parameters.chi_psi_power);
    PARTHENON_REQUIRE(parameters.chi_psi_power != 0.0,
                      "numerical_relativity/chi_psi_power must be nonzero");
    return parameters;
  }

  KOKKOS_INLINE_FUNCTION static SphericalKerrSchildPoint
  SphericalCoordinates(const parthenon::Real x, const parthenon::Real y, const parthenon::Real z,
                       const Parameters&) {
    const parthenon::Real radius = sqrt(x * x + y * y + z * z);
    const parthenon::Real cosine = radius > 0.0 ? z / radius : 1.0;
    return {radius, acos(fmax(-1.0, fmin(1.0, cosine))), atan2(y, x)};
  }

  KOKKOS_INLINE_FUNCTION static void
  BoyerLindquistSpatialToNative(const parthenon::Real x, const parthenon::Real y,
                                const parthenon::Real z, const parthenon::Real spherical_vector[3],
                                parthenon::Real cartesian_vector[3], const Parameters&) {
    const parthenon::Real radius = sqrt(x * x + y * y + z * z);
    const parthenon::Real cylindrical = sqrt(x * x + y * y);
    if (radius == 0.0 || cylindrical == 0.0) {
      cartesian_vector[0] = spherical_vector[0];
      cartesian_vector[1] = spherical_vector[1];
      cartesian_vector[2] = spherical_vector[2];
      return;
    }
    const parthenon::Real sine = cylindrical / radius;
    const parthenon::Real cosine = z / radius;
    const parthenon::Real cosphi = x / cylindrical;
    const parthenon::Real sinphi = y / cylindrical;
    cartesian_vector[0] = sine * cosphi * spherical_vector[0] +
                          cosine * cosphi * spherical_vector[1] - sinphi * spherical_vector[2];
    cartesian_vector[1] = sine * sinphi * spherical_vector[0] +
                          cosine * sinphi * spherical_vector[1] + cosphi * spherical_vector[2];
    cartesian_vector[2] = cosine * spherical_vector[0] - sine * spherical_vector[1];
  }

  KOKKOS_INLINE_FUNCTION static void
  AzimuthalCovectorToNative(const parthenon::Real x, const parthenon::Real y, const parthenon::Real,
                            const parthenon::Real azimuthal, parthenon::Real native[3],
                            const Parameters&) {
    const parthenon::Real cylindrical2 = x * x + y * y;
    if (cylindrical2 == 0.0) {
      native[0] = native[1] = native[2] = 0.0;
      return;
    }
    native[0] = -azimuthal * y / cylindrical2;
    native[1] = azimuthal * x / cylindrical2;
    native[2] = 0.0;
  }
};

} // namespace pangu::geometry::metric

#endif
