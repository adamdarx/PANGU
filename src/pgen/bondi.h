#ifndef PANGU_PGEN_BONDI_H_
#define PANGU_PGEN_BONDI_H_

#include <cmath>

#include <parthenon/parthenon.hpp>

#include "geometry/geometry.h"

namespace pangu::pgen::bondi {

using parthenon::Real;

struct Parameters {
  Real density_excision;
  Real pressure_excision;
  Real n_adi;
  Real k_adi;
  Real r_crit;
  Real c1;
  Real c2;
};

inline Parameters MakeParameters(const Real gamma, const Real k_adi, const Real r_crit,
                                 const Real density_excision, const Real pressure_excision) {
  const Real n_adi = 1.0 / (gamma - 1.0);
  const Real u_crit_squared = 1.0 / (2.0 * r_crit);
  const Real u_crit = -std::sqrt(u_crit_squared);
  const Real t_crit = n_adi / (n_adi + 1.0) * u_crit_squared /
                      (1.0 - (n_adi + 3.0) * u_crit_squared);
  return {density_excision,
          pressure_excision,
          n_adi,
          k_adi,
          r_crit,
          std::pow(t_crit, n_adi) * u_crit * r_crit * r_crit,
          (1.0 + (n_adi + 1.0) * t_crit) * (1.0 + (n_adi + 1.0) * t_crit) *
              (1.0 - 3.0 / (2.0 * r_crit))};
}

KOKKOS_INLINE_FUNCTION
Real TemperatureResidual(const Parameters& p, const Real temperature, const Real radius) {
  const Real enthalpy = 1.0 + (p.n_adi + 1.0) * temperature;
  return enthalpy * enthalpy *
             (1.0 - 2.0 / radius +
              p.c1 * p.c1 /
                  (radius * radius * radius * radius * pow(temperature, 2.0 * p.n_adi))) -
         p.c2;
}

KOKKOS_INLINE_FUNCTION
Real TemperatureMinimum(const Parameters& p, const Real radius) {
  constexpr Real ratio = 0.3819660112501051;
  Real lower = 1.0e-2;
  Real upper = 1.0e1;
  Real middle = lower + ratio * (upper - lower);
  Real residual = TemperatureResidual(p, middle, radius);
  bool right = true;
  for (int iteration = 0; iteration < 40; ++iteration) {
    if (residual < 0.0)
      return middle;
    const Real candidate =
        right ? middle + ratio * (upper - middle) : middle - ratio * (middle - lower);
    const Real candidate_residual = TemperatureResidual(p, candidate, radius);
    if (candidate_residual < residual) {
      if (right)
        lower = middle;
      else
        upper = middle;
      middle = candidate;
      residual = candidate_residual;
    } else {
      if (right)
        upper = candidate;
      else
        lower = candidate;
      right = !right;
    }
  }
  return middle;
}

KOKKOS_INLINE_FUNCTION
Real TemperatureRoot(const Parameters& p, const Real radius, Real lower, Real upper) {
  Real lower_residual = TemperatureResidual(p, lower, radius);
  const Real upper_residual = TemperatureResidual(p, upper, radius);
  if (fabs(lower_residual) < 1.0e-12)
    return lower;
  if (fabs(upper_residual) < 1.0e-12)
    return upper;
  Real middle = 0.5 * (lower + upper);
  for (int iteration = 0; iteration < 40; ++iteration) {
    middle = 0.5 * (lower + upper);
    if (upper - lower < 1.0e-12)
      break;
    const Real residual = TemperatureResidual(p, middle, radius);
    if (fabs(residual) < 1.0e-12)
      break;
    if ((residual < 0.0) == (lower_residual < 0.0)) {
      lower = middle;
      lower_residual = residual;
    } else {
      upper = middle;
    }
  }
  return middle;
}

template <class GeometryType, class PrimitiveType>
KOKKOS_INLINE_FUNCTION void Primitive(const Parameters& p, const Real x, const Real y,
                                      const Real z, const GeometryType& spacetime,
                                      const geometry::MetricPoint& metric, PrimitiveType& state) {
  const auto spherical = spacetime.SphericalCoordinates(x, y, z);
  const Real radius = spherical.radius;
  if constexpr (GeometryType::supports_excision) {
    if (radius <= 1.0) {
      state.density = p.density_excision;
      state.velocity[0] = 0.0;
      state.velocity[1] = 0.0;
      state.velocity[2] = 0.0;
      state.pressure = p.pressure_excision;
      return;
    }
  }
  const Real minimum = TemperatureMinimum(p, radius);
  const Real temperature = radius <= p.r_crit ? TemperatureRoot(p, radius, 1.0e-2, minimum)
                                               : TemperatureRoot(p, radius, minimum, 1.0e1);
  state.density = pow(temperature / p.k_adi, p.n_adi);
  state.pressure = temperature * state.density;
  const Real ur = p.c1 / (radius * radius * pow(temperature, p.n_adi));
  const Real boyer_lindquist[3]{ur, 0.0, 0.0};
  Real coordinate_u[3]{};
  spacetime.BoyerLindquistSpatialToNative(x, y, z, boyer_lindquist, coordinate_u);
  const Real spatial_norm = metric.lower[1][1] * coordinate_u[0] * coordinate_u[0] +
                            2.0 * metric.lower[1][2] * coordinate_u[0] * coordinate_u[1] +
                            2.0 * metric.lower[1][3] * coordinate_u[0] * coordinate_u[2] +
                            metric.lower[2][2] * coordinate_u[1] * coordinate_u[1] +
                            2.0 * metric.lower[2][3] * coordinate_u[1] * coordinate_u[2] +
                            metric.lower[3][3] * coordinate_u[2] * coordinate_u[2];
  const Real normalization = 1.0 + spatial_norm;
  const Real shift_dot = metric.lower[0][1] * coordinate_u[0] +
                         metric.lower[0][2] * coordinate_u[1] +
                         metric.lower[0][3] * coordinate_u[2];
  const Real u0 =
      (-shift_dot - sqrt(fmax(shift_dot * shift_dot - metric.lower[0][0] * normalization, 0.0))) /
      metric.lower[0][0];
  for (int axis = 0; axis < 3; ++axis)
    state.velocity[axis] =
        coordinate_u[axis] - metric.upper[0][axis + 1] / metric.upper[0][0] * u0;
}

} // namespace pangu::pgen::bondi

#endif
