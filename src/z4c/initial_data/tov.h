#ifndef PANGU_Z4C_TOV_H_
#define PANGU_Z4C_TOV_H_

#include <parthenon/parthenon.hpp>

namespace pangu::nr {

struct TOVParameters {
  parthenon::Real central_density = 1.28e-3;
  parthenon::Real polytropic_constant = 100.0;
  parthenon::Real gamma = 2.0;
  parthenon::Real density_floor = 1.28e-13;
  parthenon::Real radial_step = 1.0e-3;
  int maximum_points = 20000;
};

struct TOVPoint {
  parthenon::Real density = 0.0;
  parthenon::Real pressure = 0.0;
  parthenon::Real mass = 0.0;
  parthenon::Real lapse = 1.0;
  parthenon::Real schwarzschild_radius = 0.0;
  parthenon::Real conformal_factor_four = 1.0;
  bool inside = false;
};

// A read-only device profile for an unmagnetized, fixed-polytrope TOV star.
// The one-dimensional solution is integrated on the host once, copied to the
// GPU, and shared by all MeshBlocks through the numerical-relativity package.
struct TOVProfile {
  Kokkos::View<parthenon::Real*> schwarzschild_radius;
  Kokkos::View<parthenon::Real*> isotropic_radius;
  Kokkos::View<parthenon::Real*> mass;
  Kokkos::View<parthenon::Real*> pressure;
  Kokkos::View<parthenon::Real*> lapse;

  parthenon::Real central_density = 0.0;
  parthenon::Real polytropic_constant = 0.0;
  parthenon::Real gamma = 0.0;
  parthenon::Real density_floor = 0.0;
  parthenon::Real pressure_floor = 0.0;
  parthenon::Real surface_radius = 0.0;
  parthenon::Real surface_isotropic_radius = 0.0;
  parthenon::Real total_mass = 0.0;
  int points = 0;

  KOKKOS_INLINE_FUNCTION TOVPoint EvaluateIsotropic(const parthenon::Real radius) const {
    TOVPoint point{};
    if (radius >= surface_isotropic_radius) {
      const parthenon::Real safe_radius = fmax(radius, 1.0e-30);
      const parthenon::Real ratio = total_mass / (2.0 * safe_radius);
      // Match AthenaK's TOV problem generator: the one-dimensional stellar
      // profile returns vacuum outside the surface, and the hydrodynamics
      // initializer subsequently applies its independently configured
      // atmosphere floors.  Returning rho_cut here would conflate the radial
      // integration cutoff with the evolution atmosphere and produces a
      // visible surface-layer velocity mismatch after one RK step.
      point.density = 0.0;
      point.pressure = 0.0;
      point.mass = total_mass;
      point.lapse = (1.0 - ratio) / (1.0 + ratio);
      const parthenon::Real psi = 1.0 + ratio;
      point.schwarzschild_radius = safe_radius * psi * psi;
      point.conformal_factor_four = psi * psi * psi * psi;
      return point;
    }

    int lower = 0;
    int upper = points - 1;
    while (upper - lower > 1) {
      const int middle = (lower + upper) / 2;
      if (isotropic_radius(middle) <= radius)
        lower = middle;
      else
        upper = middle;
    }
    const parthenon::Real left = isotropic_radius(lower);
    const parthenon::Real right = isotropic_radius(upper);
    const parthenon::Real weight = right > left ? (radius - left) / (right - left) : 0.0;
    point.pressure =
        fmax(pressure(lower) + weight * (pressure(upper) - pressure(lower)), pressure_floor);
    point.density = fmax(pow(point.pressure / polytropic_constant, 1.0 / gamma), density_floor);
    point.mass = mass(lower) + weight * (mass(upper) - mass(lower));
    point.lapse = lapse(lower) + weight * (lapse(upper) - lapse(lower));
    point.schwarzschild_radius =
        schwarzschild_radius(lower) +
        weight * (schwarzschild_radius(upper) - schwarzschild_radius(lower));
    const parthenon::Real ratio = radius > 0.0
                                      ? point.schwarzschild_radius / radius
                                      : schwarzschild_radius(1) / isotropic_radius(1);
    point.conformal_factor_four = ratio * ratio;
    point.inside = true;
    return point;
  }
};

TOVProfile BuildPolytropicTOV(const TOVParameters& parameters);
TOVProfile BuildPolytropicTOV(parthenon::ParameterInput* pin);

} // namespace pangu::nr

#endif
