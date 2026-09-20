#include "pgen/mhd_problems.h"

// Copyright (c) 2020 James M. Stone and the AthenaK team.
// Initial conditions follow AthenaK's CPAW, field-loop, Orszag--Tang and shock-tube
// problem generators under BSD-3-Clause; see LICENSE and NOTICE.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <Kokkos_Random.hpp>

#include "bvals/boundary_conditions.hpp"
#include "eos/newtonian_mhd_eos.h"
#include "geometry/geometry.h"
#include "geometry_assembly.h"
#include "globals.hpp"
#include "mhd/mhd_types.h"
#include "pangu.h"
#include "pgen/bondi.h"
#include "relativity/relativistic_hydro.h"
#include "relativity/relativistic_mhd.h"
#include "utils/error_checking.hpp"

namespace pangu::pgen {
using namespace parthenon;

namespace {

enum class ProblemType : int {
  shock,
  cpaw,
  field_loop,
  orszag_tang,
  relativistic_linear_wave,
  electron_hubble,
  electron_noh,
  magnetised_bondi_mks,
  gr_monopole,
  gr_torus_sane,
  gr_chakrabarti_torus
};

ProblemType GetProblemType(const std::string& name) {
  if (name == "mhd_shock" || name == "brio_wu" || name == "sr_mhd_shock" || name == "gr_mhd_shock")
    return ProblemType::shock;
  if (name == "cpaw")
    return ProblemType::cpaw;
  if (name == "field_loop")
    return ProblemType::field_loop;
  if (name == "orszag_tang")
    return ProblemType::orszag_tang;
  if (name == "sr_mhd_linear_wave" || name == "sr_mhd_modes" || name == "gr_mhd_linear_wave")
    return ProblemType::relativistic_linear_wave;
  if (name == "electron_hubble")
    return ProblemType::electron_hubble;
  if (name == "electron_noh")
    return ProblemType::electron_noh;
  if (name == "magnetised_bondi_mks")
    return ProblemType::magnetised_bondi_mks;
  if (name == "gr_monopole")
    return ProblemType::gr_monopole;
  if (name == "gr_torus_sane")
    return ProblemType::gr_torus_sane;
  if (name == "gr_chakrabarti_torus" || name == "gr_chakrabarti_torus_sane")
    return ProblemType::gr_chakrabarti_torus;
  PARTHENON_FAIL("Unknown MHD problem_id='" + name + "'");
}

eos::NewtonianMHDEOS GetEos(const std::shared_ptr<StateDescriptor>& package) {
  return {static_cast<mhd::EosMode>(package->Param<int>("eos_mode")), package->Param<Real>("gamma"),
          package->Param<Real>("iso_sound_speed"), package->Param<Real>("density_floor"),
          package->Param<Real>("pressure_floor")};
}

KOKKOS_INLINE_FUNCTION
Real LoopPotential(const Real x, const Real y, const Real radius, const Real amplitude) {
  const Real distance = sqrt(x * x + y * y);
  return distance < radius ? amplitude * (radius - distance) : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real OrszagPotential(const Real x, const Real y, const Real b0) {
  constexpr Real pi = 3.1415926535897932384626433832795;
  return (b0 / (4.0 * pi)) * (cos(4.0 * pi * x) - 2.0 * cos(2.0 * pi * y));
}

template <int Component, class GeometryType>
KOKKOS_INLINE_FUNCTION Real MonopolePotential(const Real normalization, const Real x1,
                                              const Real x2, Real x3,
                                              const GeometryType& spacetime) {
  if constexpr (GeometryType::supports_excision) {
    const Real spherical = sqrt(x1 * x1 + x2 * x2 + x3 * x3);
    if (spherical < 1.0 && fabs(x3) < 1.0e-5)
      x3 = 1.0e-5;
  }
  const auto spherical = spacetime.SphericalCoordinates(x1, x2, x3);
  Real azimuthal = normalization * (1.0 - cos(spherical.theta));
  if constexpr (GeometryType::supports_excision) {
    if (spherical.radius < 1.0)
      azimuthal *=
          sin(0.5 * 3.1415926535897932384626433832795 * spherical.radius * spherical.radius);
  }
  Real native[3]{};
  spacetime.AzimuthalCovectorToNative(x1, x2, x3, azimuthal, native);
  return native[Component];
}

// Fixed-metric Fishbone--Moncrief or Chakrabarti magnetized torus. The analytic
// expressions, tilt transformation, and staggered vector-potential
// construction follow AthenaK's gr_torus generator under BSD-3-Clause.
struct SaneTorusParameters {
  Real spin;
  Real gamma;
  Real density_excision;
  Real pressure_excision;
  Real density_atmosphere;
  Real density_power;
  Real pressure_atmosphere;
  Real pressure_power;
  Real density_peak;
  Real radius_edge;
  Real radius_peak;
  Real angular_momentum;
  Real log_enthalpy_edge;
  Real density_normalization;
  Real potential_cutoff;
  Real potential_falloff;
  Real potential_radius_power;
  Real potential_density_power;
  Real tilt;
  Real sin_tilt;
  Real cos_tilt;
  Real chakrabarti_c;
  Real chakrabarti_n;
  bool prograde;
  bool fishbone_moncrief;
};

KOKKOS_INLINE_FUNCTION
Real TorusAngularMomentum(const Real spin, const Real radius, const bool prograde) {
  const Real sign = prograde ? 1.0 : -1.0;
  const Real numerator = sign * (radius * radius * radius * radius + spin * spin * radius * radius -
                                 2.0 * spin * spin * radius) -
                         spin * (radius * radius - spin * spin) * sqrt(radius);
  const Real denominator = radius * radius - 3.0 * radius + sign * 2.0 * spin * sqrt(radius);
  return numerator / (radius * sqrt(radius) * denominator);
}

KOKKOS_INLINE_FUNCTION
Real ChakrabartiKeplerianAngularMomentum(const Real spin, const Real radius) {
  return (radius * radius + spin * spin - 2.0 * spin * sqrt(radius)) /
         (sqrt(radius) * (radius - 2.0) + spin);
}

KOKKOS_INLINE_FUNCTION
Real ChakrabartiLambda(const Real spin, const Real radius, const Real angular_momentum) {
  const Real numerator =
      angular_momentum *
      (-2.0 * spin * angular_momentum + radius * radius * radius + spin * spin * (2.0 + radius));
  const Real denominator = 2.0 * spin + angular_momentum * (radius - 2.0);
  return sqrt(numerator / denominator);
}

void CalculateChakrabartiConstants(const Real spin, const Real radius_edge, const Real radius_peak,
                                   const Real n_input, Real& c, Real& n) {
  const Real l_edge = ChakrabartiKeplerianAngularMomentum(spin, radius_edge);
  const Real l_peak = ChakrabartiKeplerianAngularMomentum(spin, radius_peak);
  const Real lambda_edge = ChakrabartiLambda(spin, radius_edge, l_edge);
  const Real lambda_peak = ChakrabartiLambda(spin, radius_peak, l_peak);
  n = n_input == 0.0 ? log(l_peak / l_edge) / log(lambda_peak / lambda_edge) : n_input;
  c = n_input == 0.0 ? l_edge * pow(lambda_edge, -n) : l_peak * pow(lambda_peak, -n);
}

KOKKOS_INLINE_FUNCTION
Real ChakrabartiAngularMomentum(const SaneTorusParameters& p, const Real radius,
                                const Real sin_theta) {
  const Real sin2 = sin_theta * sin_theta;
  const Real sigma = radius * radius + p.spin * p.spin * (1.0 - sin2);
  const Real g00 = -1.0 + 2.0 * radius / sigma;
  const Real g03 = -2.0 * p.spin * radius / sigma * sin2;
  const Real g33 =
      (radius * radius + p.spin * p.spin + 2.0 * p.spin * p.spin * radius / sigma * sin2) * sin2;
  Real lower = 1.0;
  Real upper = 100.0;
  Real value = 0.5 * (lower + upper);
  for (int iteration = 0; iteration < 25; ++iteration) {
    if (0.5 * (upper - lower) / value < 1.0e-8)
      break;
    const Real residual = pow(value / p.chakrabarti_c, 2.0 / p.chakrabarti_n) +
                          (value * g33 + value * value * g03) / (g03 + value * g00);
    if (residual < 0.0)
      lower = value;
    else if (residual > 0.0)
      upper = value;
    else
      break;
    value = 0.5 * (lower + upper);
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real TorusCovariantUT(const SaneTorusParameters& p, const Real radius, const Real sin_theta,
                      const Real angular_momentum) {
  const Real sin2 = sin_theta * sin_theta;
  const Real sigma = radius * radius + p.spin * p.spin * (1.0 - sin2);
  const Real g00 = -1.0 + 2.0 * radius / sigma;
  const Real g03 = -2.0 * p.spin * radius / sigma * sin2;
  const Real g33 =
      (radius * radius + p.spin * p.spin + 2.0 * p.spin * p.spin * radius / sigma * sin2) * sin2;
  return -sqrt(fmax((g03 * g03 - g00 * g33) / (g33 + 2.0 * angular_momentum * g03 +
                                               angular_momentum * angular_momentum * g00),
                    0.0));
}

KOKKOS_INLINE_FUNCTION
Real TorusLogEnthalpy(const SaneTorusParameters& p, const Real radius, const Real sin_theta) {
  if (!p.fishbone_moncrief) {
    const Real angular_momentum = ChakrabartiAngularMomentum(p, radius, sin_theta);
    const Real u_t = TorusCovariantUT(p, radius, sin_theta, angular_momentum);
    const Real l_edge = ChakrabartiAngularMomentum(p, p.radius_edge, 1.0);
    const Real u_t_edge = TorusCovariantUT(p, p.radius_edge, 1.0, l_edge);
    Real enthalpy = u_t_edge / u_t;
    if (p.chakrabarti_n == 1.0) {
      enthalpy *= pow(l_edge / angular_momentum, p.chakrabarti_c * p.chakrabarti_c /
                                                     (p.chakrabarti_c * p.chakrabarti_c - 1.0));
    } else {
      const Real power_c = 2.0 / p.chakrabarti_n;
      const Real power_l = 2.0 - 2.0 / p.chakrabarti_n;
      const Real power_abs = p.chakrabarti_n / (2.0 - 2.0 * p.chakrabarti_n);
      enthalpy *= pow(fabs(1.0 - pow(p.chakrabarti_c, power_c) * pow(angular_momentum, power_l)),
                      power_abs) *
                  pow(fabs(1.0 - pow(p.chakrabarti_c, power_c) * pow(l_edge, power_l)), -power_abs);
    }
    if (isfinite(enthalpy) && enthalpy >= 1.0)
      return log(enthalpy);
    if (fabs(enthalpy - 1.0) <= 1.0e-15)
      return 0.0;
    return -1.0;
  }

  const Real spin = p.spin;
  const Real angular_momentum = p.angular_momentum;
  const Real sin2 = sin_theta * sin_theta;
  const Real cos2 = 1.0 - sin2;
  const Real delta = radius * radius - 2.0 * radius + spin * spin;
  const Real sigma = radius * radius + spin * spin * cos2;
  const Real aa = (radius * radius + spin * spin) * (radius * radius + spin * spin) -
                  delta * spin * spin * sin2;
  const Real exp2nu = sigma * delta / aa;
  const Real exp2psi = aa / sigma * sin2;
  const Real exp_neg2chi = exp2nu / exp2psi;
  const Real omega = 2.0 * spin * radius / aa;
  const Real a = sqrt(1.0 + 4.0 * angular_momentum * angular_momentum * exp_neg2chi);
  return 0.5 * log((1.0 + a) / (sigma * delta / aa)) - 0.5 * a - angular_momentum * omega;
}

KOKKOS_INLINE_FUNCTION
void TorusBoyerLindquistVelocity(const SaneTorusParameters& p, const Real radius,
                                 const Real sin_theta, Real& u0, Real& u3) {
  const Real sin2 = sin_theta * sin_theta;
  const Real cos2 = 1.0 - sin2;
  const Real spin2 = p.spin * p.spin;
  const Real delta = radius * radius - 2.0 * radius + spin2;
  const Real sigma = radius * radius + spin2 * cos2;
  const Real aa = (radius * radius + spin2) * (radius * radius + spin2) - delta * spin2 * sin2;
  const Real g00 = -(1.0 - 2.0 * radius / sigma);
  const Real g03 = -2.0 * p.spin * radius / sigma * sin2;
  const Real g33 = (sigma + (1.0 + 2.0 * radius / sigma) * spin2 * sin2) * sin2;
  if (p.fishbone_moncrief) {
    const Real exp2nu = sigma * delta / aa;
    const Real exp2psi = aa / sigma * sin2;
    const Real projected2 =
        0.5 * (-1.0 + sqrt(1.0 + 4.0 * p.angular_momentum * p.angular_momentum * exp2nu / exp2psi));
    Real projected = sqrt(fmax(projected2, 0.0));
    projected *= p.prograde ? 1.0 : -1.0;
    const Real u3a = (1.0 + projected * projected) / (aa * sigma * delta);
    u3 = 2.0 * p.spin * radius * sqrt(u3a) + sqrt(sigma / aa) / sin_theta * projected;
    const Real u0a = (g03 * g03 - g00 * g33) * u3 * u3;
    u0 = -(g03 * u3 + sqrt(fmax(u0a - g00, 0.0))) / g00;
  } else {
    const Real angular_momentum = ChakrabartiAngularMomentum(p, radius, sin_theta);
    const Real u_t = TorusCovariantUT(p, radius, sin_theta, angular_momentum);
    const Real g_upper_00 = -aa / (delta * sigma);
    const Real g_upper_03 = -2.0 * p.spin * radius / (delta * sigma);
    const Real omega = -(g03 + angular_momentum * g00) / (g33 + angular_momentum * g03);
    u0 = (g_upper_00 - angular_momentum * g_upper_03) * u_t;
    u3 = omega * u0;
  }
}

template <class GeometryType>
KOKKOS_INLINE_FUNCTION void
TorusCoordinates(const SaneTorusParameters& p, const Real x, const Real y, const Real z,
                 const GeometryType& spacetime, Real& radius, Real& theta, Real& phi) {
  const auto spherical = spacetime.SphericalCoordinates(x, y, z);
  radius = p.tilt == 0.0 ? spherical.radius : fmax(spherical.radius, 1.0);
  theta = spherical.theta;
  phi = spherical.phi;
  if (p.tilt != 0.0) {
    const Real delta = radius * radius - 2.0 * radius + p.spin * p.spin;
    phi -= p.spin * radius / delta;
  }
}

KOKKOS_INLINE_FUNCTION void TorusRotatedAngles(const SaneTorusParameters& p, const Real theta,
                                               const Real phi, Real& sin_vartheta,
                                               Real& cos_vartheta, Real& varphi) {
  const Real sin_theta = sin(theta);
  const Real cos_theta = cos(theta);
  if (p.tilt == 0.0) {
    sin_vartheta = fabs(sin_theta);
    cos_vartheta = cos_theta;
    varphi = phi;
    return;
  }
  const Real x = sin_theta * cos(phi);
  const Real y = sin_theta * sin(phi);
  const Real z = cos_theta;
  const Real rotated_x = p.cos_tilt * x - p.sin_tilt * z;
  const Real rotated_y = y;
  const Real rotated_z = p.sin_tilt * x + p.cos_tilt * z;
  sin_vartheta = sqrt(rotated_x * rotated_x + rotated_y * rotated_y);
  cos_vartheta = rotated_z;
  varphi = atan2(rotated_y, rotated_x);
}

template <class GeometryType>
KOKKOS_INLINE_FUNCTION bool
SaneTorusPrimitive(const SaneTorusParameters& p, const Real x, const Real y, const Real z,
                   const Real dx, const Real dy, const Real dz, const GeometryType& spacetime,
                   const geometry::MetricPoint& metric, mhd::Primitive& state) {
  Real radius = 0.0, theta = 0.0, phi = 0.0;
  TorusCoordinates(p, x, y, z, spacetime, radius, theta, phi);
  Real sin_vartheta = 0.0, cos_vartheta = 0.0, varphi = 0.0;
  TorusRotatedAngles(p, theta, phi, sin_vartheta, cos_vartheta, varphi);
  const Real log_h = radius >= p.radius_edge
                         ? TorusLogEnthalpy(p, radius, sin_vartheta) - p.log_enthalpy_edge
                         : -1.0;

  bool inside_excision = false;
  if constexpr (GeometryType::supports_excision) {
    const auto outer_corner = spacetime.SphericalCoordinates(
        x + copysign(0.5 * dx, x), y + copysign(0.5 * dy, y), z + copysign(0.5 * dz, z));
    inside_excision = outer_corner.radius <= 1.0;
  }
  const Real background_density =
      inside_excision ? p.density_excision : p.density_atmosphere * pow(radius, p.density_power);
  const Real background_pressure =
      inside_excision ? p.pressure_excision : p.pressure_atmosphere * pow(radius, p.pressure_power);
  state = {background_density, {0.0, 0.0, 0.0}, background_pressure, {0.0, 0.0, 0.0}};
  if (!(log_h >= 0.0))
    return false;

  const Real gm1 = p.gamma - 1.0;
  const Real pressure_over_density = gm1 / p.gamma * (exp(log_h) - 1.0);
  const Real density = pow(pressure_over_density, 1.0 / gm1) / p.density_normalization;
  const Real pressure = pressure_over_density * density;
  state.density = fmax(density, background_density);
  state.pressure = fmax(pressure, background_pressure);

  Real u0_bl = 0.0, u3_tilt = 0.0;
  TorusBoyerLindquistVelocity(p, radius, sin_vartheta, u0_bl, u3_tilt);
  Real u2_bl = 0.0;
  Real u3_bl = u3_tilt;
  if (p.tilt != 0.0) {
    const Real sin_theta = fmax(fabs(sin(theta)), 1.0e-20);
    const Real sin_vartheta_safe = fmax(sin_vartheta, 1.0e-20);
    const Real dtheta_dvarphi = -p.sin_tilt * sin_vartheta_safe * sin(varphi) / sin_theta;
    const Real dphi_dvarphi =
        sin_vartheta_safe / (sin_theta * sin_theta) *
        (p.cos_tilt * sin_vartheta_safe + p.sin_tilt * cos_vartheta * cos(varphi));
    u2_bl = dtheta_dvarphi * u3_tilt;
    u3_bl = dphi_dvarphi * u3_tilt;
  }
  const Real boyer_lindquist[3]{0.0, u2_bl, u3_bl};
  Real native[3]{};
  spacetime.BoyerLindquistSpatialToNative(x, y, z, boyer_lindquist, native);
  for (int axis = 0; axis < 3; ++axis)
    state.velocity[axis] = native[axis] - metric.upper[0][axis + 1] / metric.upper[0][0] * u0_bl;
  return true;
}

template <class GeometryType>
KOKKOS_INLINE_FUNCTION Real SaneTorusDensity(const SaneTorusParameters& p, const Real x,
                                             const Real y, const Real z,
                                             const GeometryType& spacetime) {
  Real radius = 0.0, theta = 0.0, phi = 0.0;
  TorusCoordinates(p, x, y, z, spacetime, radius, theta, phi);
  if (radius < p.radius_edge)
    return 0.0;
  Real sin_vartheta = 0.0, cos_vartheta = 0.0, varphi = 0.0;
  TorusRotatedAngles(p, theta, phi, sin_vartheta, cos_vartheta, varphi);
  const Real log_h = TorusLogEnthalpy(p, radius, sin_vartheta) - p.log_enthalpy_edge;
  if (!(log_h >= 0.0))
    return 0.0;
  const Real gm1 = p.gamma - 1.0;
  const Real pressure_over_density = gm1 / p.gamma * (exp(log_h) - 1.0);
  return pow(pressure_over_density, 1.0 / gm1) / p.density_normalization;
}

template <class GeometryType>
KOKKOS_INLINE_FUNCTION Real SaneAzimuthalPotential(const SaneTorusParameters& p, const Real x,
                                                   const Real y, const Real z,
                                                   const GeometryType& spacetime) {
  Real radius = 0.0, theta = 0.0, phi = 0.0;
  TorusCoordinates(p, x, y, z, spacetime, radius, theta, phi);
  const Real density = SaneTorusDensity(p, x, y, z, spacetime);
  if (!(density > 0.0))
    return 0.0;
  Real sin_vartheta = 0.0, cos_vartheta = 0.0, varphi = 0.0;
  TorusRotatedAngles(p, theta, phi, sin_vartheta, cos_vartheta, varphi);
  Real scaling = pow((radius / p.radius_edge) * sin_vartheta, p.potential_radius_power);
  if (p.potential_falloff != 0.0)
    scaling *= exp(-radius / p.potential_falloff);
  return fmax(
      pow(density / p.density_peak, p.potential_density_power) * scaling - p.potential_cutoff, 0.0);
}

template <int Component, class GeometryType>
KOKKOS_INLINE_FUNCTION Real SaneTorusPotential(const SaneTorusParameters& p, const Real x,
                                               const Real y, const Real z,
                                               const GeometryType& spacetime) {
  const Real azimuthal = SaneAzimuthalPotential(p, x, y, z, spacetime);
  if (!(azimuthal > 0.0))
    return 0.0;
  if (p.tilt != 0.0) {
    Real radius = 0.0, theta = 0.0, phi = 0.0;
    TorusCoordinates(p, x, y, z, spacetime, radius, theta, phi);
    Real sin_vartheta = 0.0, cos_vartheta = 0.0, varphi = 0.0;
    TorusRotatedAngles(p, theta, phi, sin_vartheta, cos_vartheta, varphi);
    const Real sin_vartheta2 = fmax(sin_vartheta * sin_vartheta, 1.0e-40);
    const Real sin_theta = sin(theta);
    const Real cos_theta = cos(theta);
    const Real polar = -p.sin_tilt * sin(phi) / sin_vartheta2 * azimuthal;
    const Real azimuthal_spherical = sin_theta / sin_vartheta2 *
                                     (p.cos_tilt * sin_theta - p.sin_tilt * cos_theta * cos(phi)) *
                                     azimuthal;
    const Real cartesian_radius = sqrt(x * x + y * y + z * z);
    const Real root = 2.0 * radius * radius - cartesian_radius * cartesian_radius + p.spin * p.spin;
    const Real cylindrical2 = fmax(x * x + y * y, 1.0e-12);
    const Real inverse_sine = sqrt((p.spin * p.spin + radius * radius) / cylindrical2);
    if constexpr (Component == 0)
      return polar * (x * z * inverse_sine / (radius * root)) +
             azimuthal_spherical *
                 (-y / cylindrical2 +
                  p.spin * x * radius / ((p.spin * p.spin + radius * radius) * root));
    if constexpr (Component == 1)
      return polar * (y * z * inverse_sine / (radius * root)) +
             azimuthal_spherical *
                 (x / cylindrical2 +
                  p.spin * y * radius / ((p.spin * p.spin + radius * radius) * root));
    return polar * (((1.0 + p.spin * p.spin / (radius * radius)) * z * z - root) * inverse_sine /
                    (radius * root)) +
           azimuthal_spherical * (p.spin * z / (radius * root));
  }
  Real native[3]{};
  spacetime.AzimuthalCovectorToNative(x, y, z, azimuthal, native);
  return native[Component];
}

KOKKOS_INLINE_FUNCTION
Real GRMagneticSquared(const relativity::MHDPrimitiveState& state,
                       const geometry::MetricPoint& metric) {
  const Real vx = state.fluid.u[0], vy = state.fluid.u[1], vz = state.fluid.u[2];
  const Real q = metric.lower[1][1] * vx * vx + 2.0 * metric.lower[1][2] * vx * vy +
                 2.0 * metric.lower[1][3] * vx * vz + metric.lower[2][2] * vy * vy +
                 2.0 * metric.lower[2][3] * vy * vz + metric.lower[3][3] * vz * vz;
  const Real alpha = sqrt(-1.0 / metric.upper[0][0]);
  const Real lorentz = sqrt(1.0 + q);
  const Real u0 = lorentz / alpha;
  const Real u1 = vx - alpha * lorentz * metric.upper[0][1];
  const Real u2 = vy - alpha * lorentz * metric.upper[0][2];
  const Real u3 = vz - alpha * lorentz * metric.upper[0][3];
  const Real ul1 = metric.lower[1][0] * u0 + metric.lower[1][1] * u1 + metric.lower[1][2] * u2 +
                   metric.lower[1][3] * u3;
  const Real ul2 = metric.lower[2][0] * u0 + metric.lower[2][1] * u1 + metric.lower[2][2] * u2 +
                   metric.lower[2][3] * u3;
  const Real ul3 = metric.lower[3][0] * u0 + metric.lower[3][1] * u1 + metric.lower[3][2] * u2 +
                   metric.lower[3][3] * u3;
  const Real b0 = ul1 * state.magnetic[0] + ul2 * state.magnetic[1] + ul3 * state.magnetic[2];
  const Real b1 = (state.magnetic[0] + b0 * u1) / u0;
  const Real b2 = (state.magnetic[1] + b0 * u2) / u0;
  const Real b3 = (state.magnetic[2] + b0 * u3) / u0;
  const Real bl0 = metric.lower[0][0] * b0 + metric.lower[0][1] * b1 + metric.lower[0][2] * b2 +
                   metric.lower[0][3] * b3;
  const Real bl1 = metric.lower[1][0] * b0 + metric.lower[1][1] * b1 + metric.lower[1][2] * b2 +
                   metric.lower[1][3] * b3;
  const Real bl2 = metric.lower[2][0] * b0 + metric.lower[2][1] * b1 + metric.lower[2][2] * b2 +
                   metric.lower[2][3] * b3;
  const Real bl3 = metric.lower[3][0] * b0 + metric.lower[3][1] * b1 + metric.lower[3][2] * b2 +
                   metric.lower[3][3] * b3;
  return b0 * bl0 + b1 * bl1 + b2 * bl2 + b3 * bl3;
}

template <class GeometryType, class CoordinatesType, class FaceType>
KOKKOS_INLINE_FUNCTION void
PhysicalCellMagnetic(const GeometryType& spacetime, const int geometry_block,
                     const CoordinatesType& coordinates, const FaceType& face, const int k,
                     const int j, const int i, const int ndim, Real magnetic[3]) {
  const int j2 = j + (ndim >= 2);
  const int k2 = k + (ndim >= 3);
  if constexpr (GeometryType::unit_determinant) {
    magnetic[0] = 0.5 * (face(0, 0, 0, 0, k, j, i) + face(0, 0, 0, 0, k, j, i + 1));
    magnetic[1] = 0.5 * (face(1, 0, 0, 0, k, j, i) + face(1, 0, 0, 0, k, j2, i));
    magnetic[2] = 0.5 * (face(2, 0, 0, 0, k, j, i) + face(2, 0, 0, 0, k2, j, i));
  } else {
    const auto x1_left =
        spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i, coordinates);
    const auto x1_right =
        spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i + 1, coordinates);
    const auto x2_left =
        spacetime.MetricAt(geometry::Location::face2, geometry_block, k, j, i, coordinates);
    const auto x2_right =
        spacetime.MetricAt(geometry::Location::face2, geometry_block, k, j2, i, coordinates);
    const auto x3_left =
        spacetime.MetricAt(geometry::Location::face3, geometry_block, k, j, i, coordinates);
    const auto x3_right =
        spacetime.MetricAt(geometry::Location::face3, geometry_block, k2, j, i, coordinates);
    magnetic[0] = 0.5 * (face(0, 0, 0, 0, k, j, i) / x1_left.gdet +
                         face(0, 0, 0, 0, k, j, i + 1) / x1_right.gdet);
    magnetic[1] = 0.5 * (face(1, 0, 0, 0, k, j, i) / x2_left.gdet +
                         face(1, 0, 0, 0, k, j2, i) / x2_right.gdet);
    magnetic[2] = 0.5 * (face(2, 0, 0, 0, k, j, i) / x3_left.gdet +
                         face(2, 0, 0, 0, k2, j, i) / x3_right.gdet);
  }
}

struct ProfileRow {
  Real x;
  Real y;
  Real z;
  Real density;
  Real velocity1;
  Real velocity2;
  Real velocity3;
  Real pressure;
  Real magnetic1;
  Real magnetic2;
  Real magnetic3;
  Real divb;
};

template <int Direction, bool Inner>
void MonopoleBoundary(std::shared_ptr<MeshBlockData<Real>>& data, const bool coarse) {
  // AthenaK registers these faces as `user`; its generic HydroBCs/BFieldBCs
  // therefore leave them untouched before ReflectingMonopole runs.  Calling
  // Parthenon's stock outflow/reflect helpers here also rewrites the active
  // boundary face (most visibly B3 at z=0), which is not part of AthenaK's
  // custom stencil.
  if (coarse)
    return;

  auto block = data->GetBlockPointer();
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto face = data->Get("mhd.b_face").data;
  const auto geometry_package = block->packages.Get("geometry");
  const bool excision = geometry_package->Param<bool>("excision");
  const Real excision_radius = geometry_package->Param<Real>("excision_radius");
  const auto mhd_package = block->packages.Get("mhd");
  const auto eos = pangu::eos::ReadMagnetised(*mhd_package);
  const Real configured_excision_density = geometry_package->Param<Real>("excision_density");
  const Real configured_excision_pressure = geometry_package->Param<Real>("excision_pressure");
  const Real excision_density =
      configured_excision_density > 0.0 ? configured_excision_density : eos.density_floor;
  const Real excision_pressure =
      configured_excision_pressure > 0.0 ? configured_excision_pressure : eos.pressure_floor;
  const Real gamma_max = mhd_package->Param<Real>("gamma_max");
  const Real sigma_max = mhd_package->Param<Real>("sigma_max");
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto ei = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto ej = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto ek = block->cellbounds.GetBoundsK(IndexDomain::entire);

  if constexpr (Direction == 0) {
    const int gs = Inner ? ei.s : ib.e + 1;
    const int ge = Inner ? ib.s - 1 : ei.e;
    const int source = Inner ? ib.s : ib.e;
    const int fs = Inner ? ei.s : ib.e + 2;
    const int fe = Inner ? ib.s - 1 : ei.e + 1;
    const int face_source = Inner ? ib.s : ib.e + 1;
    block->par_for(
        "PANGU monopole x1 normal field", ek.s, ek.e, ej.s, ej.e, fs, fe,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          face(0, 0, 0, 0, k, j, i) = face(0, 0, 0, 0, k, j, face_source);
        });
    block->par_for(
        "PANGU monopole x1 tangential field", 0, 1, ek.s, ek.e + 1, ej.s, ej.e + 1, gs, ge,
        KOKKOS_LAMBDA(const int component, const int k, const int j, const int i) {
          if (component == 0 && k <= ek.e)
            face(1, 0, 0, 0, k, j, i) = face(1, 0, 0, 0, k, j, source);
          if (component == 1 && j <= ej.e)
            face(2, 0, 0, 0, k, j, i) = face(2, 0, 0, 0, k, j, source);
        });
  } else if constexpr (Direction == 1) {
    const int gs = Inner ? ej.s : jb.e + 1;
    const int ge = Inner ? jb.s - 1 : ej.e;
    const int source = Inner ? jb.s : jb.e;
    const int fs = Inner ? ej.s : jb.e + 2;
    const int fe = Inner ? jb.s - 1 : ej.e + 1;
    const int face_source = Inner ? jb.s : jb.e + 1;
    block->par_for(
        "PANGU monopole x2 normal field", ek.s, ek.e, fs, fe, ei.s, ei.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          face(1, 0, 0, 0, k, j, i) = face(1, 0, 0, 0, k, face_source, i);
        });
    block->par_for(
        "PANGU monopole x2 tangential field", 0, 1, ek.s, ek.e + 1, gs, ge, ei.s, ei.e + 1,
        KOKKOS_LAMBDA(const int component, const int k, const int j, const int i) {
          if (component == 0 && k <= ek.e)
            face(0, 0, 0, 0, k, j, i) = face(0, 0, 0, 0, k, source, i);
          if (component == 1 && i <= ei.e)
            face(2, 0, 0, 0, k, j, i) = face(2, 0, 0, 0, k, source, i);
        });
  } else {
    const int gs = Inner ? ek.s : kb.e + 1;
    const int ge = Inner ? kb.s - 1 : ek.e;
    const int source = Inner ? kb.s : kb.e;
    const int fs = Inner ? ek.s : kb.e + 2;
    const int fe = Inner ? kb.s - 1 : ek.e + 1;
    const int face_source = Inner ? kb.s + 1 : kb.e + 1;
    block->par_for(
        "PANGU monopole x3 normal field", fs, fe, ej.s, ej.e, ei.s, ei.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real sign = Inner ? -1.0 : 1.0;
          face(2, 0, 0, 0, k, j, i) = sign * face(2, 0, 0, 0, face_source, j, i);
        });
    block->par_for(
        "PANGU monopole x3 tangential field", 0, 1, gs, ge, ej.s, ej.e + 1, ei.s, ei.e + 1,
        KOKKOS_LAMBDA(const int component, const int k, const int j, const int i) {
          if (component == 0 && j <= ej.e)
            face(0, 0, 0, 0, k, j, i) = face(0, 0, 0, 0, source, j, i);
          if (component == 1 && i <= ei.e)
            face(1, 0, 0, 0, k, j, i) = face(1, 0, 0, 0, source, j, i);
        });
  }

  // ReflectingMonopole runs ConsToPrim on the complete boundary slab, including
  // the boundary-most active plane, before copying that plane into the ghost
  // zones.  In particular, C2P writes a repaired conserved state back when a
  // floor, ceiling, failed inversion, or excision is encountered.  Performing
  // this as its own kernel also avoids concurrent writes to the same source
  // cell from the ghost-copy kernel below.
  int ail = ei.s, aiu = ei.e, ajl = ej.s, aju = ej.e, akl = ek.s, aku = ek.e;
  if constexpr (Direction == 0)
    ail = aiu = Inner ? ib.s : ib.e;
  if constexpr (Direction == 1)
    ajl = aju = Inner ? jb.s : jb.e;
  if constexpr (Direction == 2)
    akl = aku = Inner ? kb.s : kb.e;
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  block->par_for(
      "PANGU monopole active boundary C2P", akl, aku, ajl, aju, ail, aiu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real magnetic[3]{};
        if (decltype(spacetime)::unit_determinant) {
          magnetic[0] = 0.5 * (face(0, 0, 0, 0, k, j, i) + face(0, 0, 0, 0, k, j, i + 1));
          magnetic[1] = 0.5 * (face(1, 0, 0, 0, k, j, i) + face(1, 0, 0, 0, k, j + 1, i));
          magnetic[2] = 0.5 * (face(2, 0, 0, 0, k, j, i) + face(2, 0, 0, 0, k + 1, j, i));
        } else {
          const auto x1_left =
              spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i, coords);
          const auto x1_right =
              spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i + 1, coords);
          const auto x2_left =
              spacetime.MetricAt(geometry::Location::face2, geometry_block, k, j, i, coords);
          const auto x2_right =
              spacetime.MetricAt(geometry::Location::face2, geometry_block, k, j + 1, i, coords);
          const auto x3_left =
              spacetime.MetricAt(geometry::Location::face3, geometry_block, k, j, i, coords);
          const auto x3_right =
              spacetime.MetricAt(geometry::Location::face3, geometry_block, k + 1, j, i, coords);
          magnetic[0] = 0.5 * (face(0, 0, 0, 0, k, j, i) / x1_left.gdet +
                               face(0, 0, 0, 0, k, j, i + 1) / x1_right.gdet);
          magnetic[1] = 0.5 * (face(1, 0, 0, 0, k, j, i) / x2_left.gdet +
                               face(1, 0, 0, 0, k, j + 1, i) / x2_right.gdet);
          magnetic[2] = 0.5 * (face(2, 0, 0, 0, k, j, i) / x3_left.gdet +
                               face(2, 0, 0, 0, k + 1, j, i) / x3_right.gdet);
        }
        const relativity::HydroConservedState old_conserved{conserved(mhd::IDN, k, j, i),
                                                            {conserved(mhd::IM1, k, j, i),
                                                             conserved(mhd::IM2, k, j, i),
                                                             conserved(mhd::IM3, k, j, i)},
                                                            conserved(mhd::IEN, k, j, i)};
        const auto metric =
            spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
        auto result =
            relativity::SolveGRMHDC2P(old_conserved, magnetic, eos, gamma_max, sigma_max, metric);
        const bool excised_cell = excision && spacetime.IsExcised(coords.Xc<1>(i), coords.Xc<2>(j),
                                                                  coords.Xc<3>(k), excision_radius);
        if (excised_cell)
          result.primitive.fluid = {excision_density, {0.0, 0.0, 0.0}, excision_pressure};
        if (excised_cell)
          result.internal_energy = eos.InternalEnergyDensity(excision_pressure);
        primitive(mhd::IDN, k, j, i) = result.primitive.fluid.density;
        primitive(mhd::IV1, k, j, i) = result.primitive.fluid.u[0];
        primitive(mhd::IV2, k, j, i) = result.primitive.fluid.u[1];
        primitive(mhd::IV3, k, j, i) = result.primitive.fluid.u[2];
        primitive(mhd::IPR, k, j, i) = result.internal_energy;
        for (int axis = 0; axis < 3; ++axis)
          bcell(axis, k, j, i) = magnetic[axis];
        if (!result.success || result.density_floor || result.pressure_floor ||
            result.lorentz_ceiling || result.sigma_ceiling || excised_cell) {
          const auto repaired = relativity::ConvertGRMHDP2CWithInternalEnergy(
              result.primitive, result.internal_energy, eos, metric);
          conserved(mhd::IDN, k, j, i) = repaired.density;
          conserved(mhd::IM1, k, j, i) = repaired.momentum[0];
          conserved(mhd::IM2, k, j, i) = repaired.momentum[1];
          conserved(mhd::IM3, k, j, i) = repaired.momentum[2];
          conserved(mhd::IEN, k, j, i) = repaired.energy;
        }
      });

  int il = ei.s, iu = ei.e, jl = ej.s, ju = ej.e, kl = ek.s, ku = ek.e;
  int source = 0;
  if constexpr (Direction == 0) {
    iu = Inner ? ib.s - 1 : ei.e;
    il = Inner ? ei.s : ib.e + 1;
    source = Inner ? ib.s : ib.e;
  } else if constexpr (Direction == 1) {
    ju = Inner ? jb.s - 1 : ej.e;
    jl = Inner ? ej.s : jb.e + 1;
    source = Inner ? jb.s : jb.e;
  } else {
    ku = Inner ? kb.s - 1 : ek.e;
    kl = Inner ? ek.s : kb.e + 1;
    source = Inner ? kb.s : kb.e;
  }
  block->par_for(
      "PANGU monopole no-inflow conserved boundary", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int si = Direction == 0 ? source : i;
        const int sj = Direction == 1 ? source : j;
        const int sk = Direction == 2 ? source : k;
        const Real internal = primitive(mhd::IPR, sk, sj, si);
        relativity::MHDPrimitiveState state{};
        state.fluid = {primitive(mhd::IDN, sk, sj, si),
                       {primitive(mhd::IV1, sk, sj, si), primitive(mhd::IV2, sk, sj, si),
                        primitive(mhd::IV3, sk, sj, si)},
                       eos.PressureFromInternalEnergyDensity(internal)};
        if (Direction == 2 && Inner)
          state.fluid.u[Direction] = -state.fluid.u[Direction];
        else if (Inner)
          state.fluid.u[Direction] = fmin(0.0, state.fluid.u[Direction]);
        else
          state.fluid.u[Direction] = fmax(0.0, state.fluid.u[Direction]);
        if (decltype(spacetime)::unit_determinant) {
          state.magnetic[0] = 0.5 * (face(0, 0, 0, 0, k, j, i) + face(0, 0, 0, 0, k, j, i + 1));
          state.magnetic[1] = 0.5 * (face(1, 0, 0, 0, k, j, i) + face(1, 0, 0, 0, k, j + 1, i));
          state.magnetic[2] = 0.5 * (face(2, 0, 0, 0, k, j, i) + face(2, 0, 0, 0, k + 1, j, i));
        } else {
          const auto x1_left =
              spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i, coords);
          const auto x1_right =
              spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i + 1, coords);
          const auto x2_left =
              spacetime.MetricAt(geometry::Location::face2, geometry_block, k, j, i, coords);
          const auto x2_right =
              spacetime.MetricAt(geometry::Location::face2, geometry_block, k, j + 1, i, coords);
          const auto x3_left =
              spacetime.MetricAt(geometry::Location::face3, geometry_block, k, j, i, coords);
          const auto x3_right =
              spacetime.MetricAt(geometry::Location::face3, geometry_block, k + 1, j, i, coords);
          state.magnetic[0] = 0.5 * (face(0, 0, 0, 0, k, j, i) / x1_left.gdet +
                                     face(0, 0, 0, 0, k, j, i + 1) / x1_right.gdet);
          state.magnetic[1] = 0.5 * (face(1, 0, 0, 0, k, j, i) / x2_left.gdet +
                                     face(1, 0, 0, 0, k, j + 1, i) / x2_right.gdet);
          state.magnetic[2] = 0.5 * (face(2, 0, 0, 0, k, j, i) / x3_left.gdet +
                                     face(2, 0, 0, 0, k + 1, j, i) / x3_right.gdet);
        }
        const auto metric =
            spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
        const auto converted =
            relativity::ConvertGRMHDP2CWithInternalEnergy(state, internal, eos, metric);
        conserved(mhd::IDN, k, j, i) = converted.density;
        conserved(mhd::IM1, k, j, i) = converted.momentum[0];
        conserved(mhd::IM2, k, j, i) = converted.momentum[1];
        conserved(mhd::IM3, k, j, i) = converted.momentum[2];
        conserved(mhd::IEN, k, j, i) = converted.energy;
      });
}

template <int Direction, bool Inner>
void MHDTorusBoundary(std::shared_ptr<MeshBlockData<Real>>& data, const bool coarse) {
  if (coarse)
    return;

  auto block = data->GetBlockPointer();
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto face = data->Get("mhd.b_face").data;
  const auto mhd_package = block->packages.Get("mhd");
  const auto geometry_package = block->packages.Get("geometry");
  const auto eos = pangu::eos::ReadMagnetised(*mhd_package);
  const Real gamma_max = mhd_package->Param<Real>("gamma_max");
  const Real sigma_max = mhd_package->Param<Real>("sigma_max");
  const auto background =
      static_cast<geometry::Background>(geometry_package->Param<int>("background"));
  const Real spin = geometry_package->Param<Real>("bh_spin");
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto ei = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto ej = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto ek = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const int ndim = block->pmy_mesh->ndim;

  // Copy face-centered fields with AthenaK's zero-gradient torus boundary.
  if constexpr (Direction == 0) {
    const int gs = Inner ? ei.s : ib.e + 1;
    const int ge = Inner ? ib.s - 1 : ei.e;
    const int nfs = Inner ? ei.s : ib.e + 2;
    const int nfe = Inner ? ib.s - 1 : ei.e + 1;
    const int source = Inner ? ib.s : ib.e;
    const int normal_source = Inner ? ib.s : ib.e + 1;
    block->par_for(
        "PANGU SANE torus x1 normal field boundary", ek.s, ek.e, ej.s, ej.e, nfs, nfe,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          face(0, 0, 0, 0, k, j, i) = face(0, 0, 0, 0, k, j, normal_source);
        });
    block->par_for(
        "PANGU SANE torus x1 tangential field boundary", 0, 1, ek.s, ek.e + 1, ej.s, ej.e + 1, gs,
        ge, KOKKOS_LAMBDA(const int component, const int k, const int j, const int i) {
          if (component == 0 && k <= ek.e)
            face(1, 0, 0, 0, k, j, i) = face(1, 0, 0, 0, k, j, source);
          if (component == 1 && j <= ej.e)
            face(2, 0, 0, 0, k, j, i) = face(2, 0, 0, 0, k, j, source);
        });
  } else if constexpr (Direction == 1) {
    const int gs = Inner ? ej.s : jb.e + 1;
    const int ge = Inner ? jb.s - 1 : ej.e;
    const int nfs = Inner ? ej.s : jb.e + 2;
    const int nfe = Inner ? jb.s - 1 : ej.e + 1;
    const int source = Inner ? jb.s : jb.e;
    const int normal_source = Inner ? jb.s : jb.e + 1;
    block->par_for(
        "PANGU SANE torus x2 normal field boundary", ek.s, ek.e, nfs, nfe, ei.s, ei.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          face(1, 0, 0, 0, k, j, i) = face(1, 0, 0, 0, k, normal_source, i);
        });
    block->par_for(
        "PANGU SANE torus x2 tangential field boundary", 0, 1, ek.s, ek.e + 1, gs, ge, ei.s,
        ei.e + 1, KOKKOS_LAMBDA(const int component, const int k, const int j, const int i) {
          if (component == 0 && k <= ek.e)
            face(0, 0, 0, 0, k, j, i) = face(0, 0, 0, 0, k, source, i);
          if (component == 1 && i <= ei.e)
            face(2, 0, 0, 0, k, j, i) = face(2, 0, 0, 0, k, source, i);
        });
  } else {
    const int gs = Inner ? ek.s : kb.e + 1;
    const int ge = Inner ? kb.s - 1 : ek.e;
    const int nfs = Inner ? ek.s : kb.e + 2;
    const int nfe = Inner ? kb.s - 1 : ek.e + 1;
    const int source = Inner ? kb.s : kb.e;
    const int normal_source = Inner ? kb.s : kb.e + 1;
    block->par_for(
        "PANGU SANE torus x3 normal field boundary", nfs, nfe, ej.s, ej.e, ei.s, ei.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          face(2, 0, 0, 0, k, j, i) = face(2, 0, 0, 0, normal_source, j, i);
        });
    block->par_for(
        "PANGU SANE torus x3 tangential field boundary", 0, 1, gs, ge, ej.s, ej.e + 1, ei.s,
        ei.e + 1, KOKKOS_LAMBDA(const int component, const int k, const int j, const int i) {
          if (component == 0 && j <= ej.e)
            face(0, 0, 0, 0, k, j, i) = face(0, 0, 0, 0, source, j, i);
          if (component == 1 && i <= ei.e)
            face(1, 0, 0, 0, k, j, i) = face(1, 0, 0, 0, source, j, i);
        });
  }

  // Recover a consistent primitive state at the source active plane before
  // filling ghosts, matching NoInflowTorus rather than copying conserved
  // variables through a spatially varying metric.
  int ail = ei.s, aiu = ei.e, ajl = ej.s, aju = ej.e, akl = ek.s, aku = ek.e;
  if constexpr (Direction == 0)
    ail = aiu = Inner ? ib.s : ib.e;
  if constexpr (Direction == 1)
    ajl = aju = Inner ? jb.s : jb.e;
  if constexpr (Direction == 2)
    akl = aku = Inner ? kb.s : kb.e;
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  block->par_for(
      "PANGU SANE torus active boundary C2P", akl, aku, ajl, aju, ail, aiu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real magnetic[3]{};
        PhysicalCellMagnetic(spacetime, geometry_block, coords, face, k, j, i, ndim, magnetic);
        const relativity::HydroConservedState old_conserved{conserved(mhd::IDN, k, j, i),
                                                            {conserved(mhd::IM1, k, j, i),
                                                             conserved(mhd::IM2, k, j, i),
                                                             conserved(mhd::IM3, k, j, i)},
                                                            conserved(mhd::IEN, k, j, i)};
        const auto metric =
            spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
        const auto result =
            relativity::SolveGRMHDC2P(old_conserved, magnetic, eos, gamma_max, sigma_max, metric);
        primitive(mhd::IDN, k, j, i) = result.primitive.fluid.density;
        primitive(mhd::IV1, k, j, i) = result.primitive.fluid.u[0];
        primitive(mhd::IV2, k, j, i) = result.primitive.fluid.u[1];
        primitive(mhd::IV3, k, j, i) = result.primitive.fluid.u[2];
        primitive(mhd::IPR, k, j, i) = result.internal_energy;
        for (int axis = 0; axis < 3; ++axis)
          bcell(axis, k, j, i) = magnetic[axis];
      });

  int il = ei.s, iu = ei.e, jl = ej.s, ju = ej.e, kl = ek.s, ku = ek.e;
  int source = 0;
  if constexpr (Direction == 0) {
    il = Inner ? ei.s : ib.e + 1;
    iu = Inner ? ib.s - 1 : ei.e;
    source = Inner ? ib.s : ib.e;
  } else if constexpr (Direction == 1) {
    jl = Inner ? ej.s : jb.e + 1;
    ju = Inner ? jb.s - 1 : ej.e;
    source = Inner ? jb.s : jb.e;
  } else {
    kl = Inner ? ek.s : kb.e + 1;
    ku = Inner ? kb.s - 1 : ek.e;
    source = Inner ? kb.s : kb.e;
  }
  block->par_for(
      "PANGU SANE torus no-inflow primitive boundary", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int si = Direction == 0 ? source : i;
        const int sj = Direction == 1 ? source : j;
        const int sk = Direction == 2 ? source : k;
        const Real internal = primitive(mhd::IPR, sk, sj, si);
        relativity::MHDPrimitiveState state{};
        state.fluid = {primitive(mhd::IDN, sk, sj, si),
                       {primitive(mhd::IV1, sk, sj, si), primitive(mhd::IV2, sk, sj, si),
                        primitive(mhd::IV3, sk, sj, si)},
                       eos.PressureFromInternalEnergyDensity(internal)};
        state.fluid.u[Direction] =
            Inner ? fmin(0.0, state.fluid.u[Direction]) : fmax(0.0, state.fluid.u[Direction]);
        PhysicalCellMagnetic(spacetime, geometry_block, coords, face, k, j, i, ndim,
                             state.magnetic);
        primitive(mhd::IDN, k, j, i) = state.fluid.density;
        primitive(mhd::IV1, k, j, i) = state.fluid.u[0];
        primitive(mhd::IV2, k, j, i) = state.fluid.u[1];
        primitive(mhd::IV3, k, j, i) = state.fluid.u[2];
        primitive(mhd::IPR, k, j, i) = internal;
        for (int axis = 0; axis < 3; ++axis)
          bcell(axis, k, j, i) = state.magnetic[axis];
        const auto metric =
            spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
        const auto converted =
            relativity::ConvertGRMHDP2CWithInternalEnergy(state, internal, eos, metric);
        conserved(mhd::IDN, k, j, i) = converted.density;
        conserved(mhd::IM1, k, j, i) = converted.momentum[0];
        conserved(mhd::IM2, k, j, i) = converted.momentum[1];
        conserved(mhd::IM3, k, j, i) = converted.momentum[2];
        conserved(mhd::IEN, k, j, i) = converted.energy;
      });
}

template <bool Inner>
void MagnetisedBondiBoundary(std::shared_ptr<MeshBlockData<Real>>& data, const bool coarse) {
  if (coarse)
    return;

  auto block = data->GetBlockPointer();
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto face = data->Get("mhd.b_face").data;
  const auto package = block->packages.Get("mhd");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  const auto bondi_parameters = bondi::MakeParameters(
      eos.gamma, package->Param<Real>("bondi_k_adi"), package->Param<Real>("bondi_r_crit"),
      package->Param<Real>("density_floor"), package->Param<Real>("pressure_floor"));
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto ei = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const int ndim = block->pmy_mesh->ndim;

  const int ghost_start = Inner ? ei.s : ib.e + 1;
  const int ghost_end = Inner ? ib.s - 1 : ei.e;
  const int normal_face_start = Inner ? ei.s : ib.e + 2;
  const int normal_face_end = Inner ? ib.s - 1 : ei.e + 1;
  const int cell_source = Inner ? ib.s : ib.e;
  const int normal_face_source = Inner ? ib.s : ib.e + 1;

  // The monopolar field is represented by the densitized CT face field.  A
  // radial analytic field is constant along x1 in this representation, so
  // copying the already normalized physical boundary face preserves both its
  // normalization and the discrete divergence constraint.
  block->par_for(
      "PANGU magnetised Bondi radial face boundary", kb.s, kb.e, jb.s, jb.e, normal_face_start,
      normal_face_end, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        face(0, 0, 0, 0, k, j, i) = face(0, 0, 0, 0, k, j, normal_face_source);
      });
  block->par_for(
      "PANGU magnetised Bondi theta face boundary", kb.s, kb.e, jb.s, jb.e + 1, ghost_start,
      ghost_end, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        face(1, 0, 0, 0, k, j, i) = face(1, 0, 0, 0, k, j, cell_source);
      });
  block->par_for(
      "PANGU magnetised Bondi phi face boundary", kb.s, kb.e + (ndim >= 3), jb.s, jb.e, ghost_start,
      ghost_end, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        face(2, 0, 0, 0, k, j, i) = face(2, 0, 0, 0, k, j, cell_source);
      });

  const auto coordinates = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  block->par_for(
      "PANGU analytic magnetised Bondi radial boundary", kb.s, kb.e, jb.s, jb.e, ghost_start,
      ghost_end, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        mhd::Primitive analytic{};
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k,
                                               j, i, coordinates);
        bondi::Primitive(bondi_parameters, coordinates.Xc<1>(i), coordinates.Xc<2>(j),
                         coordinates.Xc<3>(k), spacetime, metric, analytic);
        const Real internal = eos.InternalEnergyDensity(analytic.pressure);
        relativity::MHDPrimitiveState state{
            {analytic.density,
             {analytic.velocity[0], analytic.velocity[1], analytic.velocity[2]},
             analytic.pressure},
            {}};
        if (decltype(spacetime)::unit_determinant) {
          state.magnetic[0] = 0.5 * (face(0, 0, 0, 0, k, j, i) + face(0, 0, 0, 0, k, j, i + 1));
          state.magnetic[1] = 0.5 * (face(1, 0, 0, 0, k, j, i) + face(1, 0, 0, 0, k, j + 1, i));
          state.magnetic[2] =
              0.5 * (face(2, 0, 0, 0, k, j, i) + face(2, 0, 0, 0, k + (ndim >= 3), j, i));
        } else {
          const auto x1_left =
              spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i, coordinates);
          const auto x1_right = spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j,
                                                   i + 1, coordinates);
          const auto x2_left =
              spacetime.MetricAt(geometry::Location::face2, geometry_block, k, j, i, coordinates);
          const auto x2_right = spacetime.MetricAt(geometry::Location::face2, geometry_block, k,
                                                   j + 1, i, coordinates);
          const auto x3_left =
              spacetime.MetricAt(geometry::Location::face3, geometry_block, k, j, i, coordinates);
          const auto x3_right = spacetime.MetricAt(geometry::Location::face3, geometry_block,
                                                   k + (ndim >= 3), j, i, coordinates);
          state.magnetic[0] = 0.5 * (face(0, 0, 0, 0, k, j, i) / x1_left.gdet +
                                     face(0, 0, 0, 0, k, j, i + 1) / x1_right.gdet);
          state.magnetic[1] = 0.5 * (face(1, 0, 0, 0, k, j, i) / x2_left.gdet +
                                     face(1, 0, 0, 0, k, j + 1, i) / x2_right.gdet);
          state.magnetic[2] = 0.5 * (face(2, 0, 0, 0, k, j, i) / x3_left.gdet +
                                     face(2, 0, 0, 0, k + (ndim >= 3), j, i) / x3_right.gdet);
        }
        primitive(mhd::IDN, k, j, i) = state.fluid.density;
        primitive(mhd::IV1, k, j, i) = state.fluid.u[0];
        primitive(mhd::IV2, k, j, i) = state.fluid.u[1];
        primitive(mhd::IV3, k, j, i) = state.fluid.u[2];
        primitive(mhd::IPR, k, j, i) = internal;
        for (int axis = 0; axis < 3; ++axis)
          bcell(axis, k, j, i) = state.magnetic[axis];
        const auto converted =
            relativity::ConvertGRMHDP2CWithInternalEnergy(state, internal, eos, metric);
        conserved(mhd::IDN, k, j, i) = converted.density;
        conserved(mhd::IM1, k, j, i) = converted.momentum[0];
        conserved(mhd::IM2, k, j, i) = converted.momentum[1];
        conserved(mhd::IM3, k, j, i) = converted.momentum[2];
        conserved(mhd::IEN, k, j, i) = converted.energy;
      });
}

} // namespace

void RegisterMonopoleBoundaries(ApplicationInput* app_input) {
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x1, "monopole",
                                       MonopoleBoundary<0, true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x1, "monopole",
                                       MonopoleBoundary<0, false>);
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x2, "monopole",
                                       MonopoleBoundary<1, true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x2, "monopole",
                                       MonopoleBoundary<1, false>);
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x3, "monopole",
                                       MonopoleBoundary<2, true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x3, "monopole",
                                       MonopoleBoundary<2, false>);
}

void RegisterMagnetisedBondiBoundaries(ApplicationInput* app_input) {
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x1, "bondi_mhd_mks",
                                       MagnetisedBondiBoundary<true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x1, "bondi_mhd_mks",
                                       MagnetisedBondiBoundary<false>);
}

void RegisterChakrabartiTorusBoundaries(ApplicationInput* app_input) {
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x1, "torus_mhd",
                                       MHDTorusBoundary<0, true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x1, "torus_mhd",
                                       MHDTorusBoundary<0, false>);
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x2, "torus_mhd",
                                       MHDTorusBoundary<1, true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x2, "torus_mhd",
                                       MHDTorusBoundary<1, false>);
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x3, "torus_mhd",
                                       MHDTorusBoundary<2, true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x3, "torus_mhd",
                                       MHDTorusBoundary<2, false>);
}

void MHDProblem(MeshBlock* block, ParameterInput* pin) {
  const auto problem_name = pin->GetString("parthenon/job", "problem_id");
  const auto problem = GetProblemType(problem_name);
  // Register the optional compatibility output switch before Parthenon's
  // unused-parameter audit; the value is consumed by MHDAfterLoop.
  pin->GetOrAddBoolean("problem", "write_final_csv", true);
  const auto package = block->packages.Get("mhd");
  const auto eos = GetEos(package);
  const auto relativistic_eos = pangu::eos::ReadRelativistic(*package);
  const auto physics = static_cast<relativity::HydroMode>(package->Param<int>("physics_mode"));
  const auto geometry_package = block->packages.Get("geometry");
  const auto background =
      static_cast<geometry::Background>(geometry_package->Param<int>("background"));
  const Real spin = geometry_package->Param<Real>("bh_spin");
  const Real configured_excision_density = geometry_package->Param<Real>("excision_density");
  const Real configured_excision_pressure = geometry_package->Param<Real>("excision_pressure");
  const Real excision_density =
      configured_excision_density > 0.0 ? configured_excision_density : eos.density_floor;
  const Real excision_pressure =
      configured_excision_pressure > 0.0 ? configured_excision_pressure : eos.pressure_floor;
  auto data = block->meshblock_data.Get();
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto divb = data->PackVariables(std::vector<std::string>{"mhd.divb"});
  auto fofc = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto bface = data->Get("mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  const int ndim = block->pmy_mesh->ndim;
  const int fluid_components = eos.HasEnergy() ? mhd::kIdealComponents : mhd::kIsothermalComponents;
  const int nscalars = package->Param<int>("nscalars");
  const int kind = static_cast<int>(problem);

  const Real x0 = pin->GetOrAddReal("problem", "x0", 0.0);
  const Real y0 = pin->GetOrAddReal("problem", "y0", 0.0);
  const Real z0 = pin->GetOrAddReal("problem", "z0", 0.0);
  const Real radius = pin->GetOrAddReal("problem", "radius", 0.3);
  const Real amplitude = pin->GetOrAddReal("problem", "amplitude", 1.0e-3);
  const Real vflow = pin->GetOrAddReal("problem", "vflow", 1.0);
  const Real vx0 = pin->GetOrAddReal("problem", "vx0", 0.0);
  const Real vy0 = pin->GetOrAddReal("problem", "vy0", 0.0);
  const Real vz0 = pin->GetOrAddReal("problem", "vz0", 0.0);
  const Real density0 = pin->GetOrAddReal("problem", "ambient_density", 1.0);
  const Real gas_pressure0 = pin->GetOrAddReal("problem", "ambient_pressure", 1.0);
  const Real electron_mach = pin->GetOrAddReal("problem", "mach", 49.0);
  const Real electron_density = pin->GetOrAddReal("problem", "rho", 1.0);
  const Real electron_v0 = pin->GetOrAddReal("problem", "v0", 1.0e-3);
  const Real electron_sound2 = electron_v0 * electron_v0 / (electron_mach * electron_mach);
  const bool electron_zero_internal = pin->GetOrAddBoolean("problem", "zero_ug", false);
  const bool electron_noh_centered = pin->GetOrAddBoolean("problem", "centered", true);
  PARTHENON_REQUIRE(electron_mach > 0.0, "problem/mach must be positive");
  PARTHENON_REQUIRE(electron_density > 0.0, "problem/rho must be positive");
  PARTHENON_REQUIRE(fabs(electron_v0) < 1.0, "problem/v0 must have magnitude below one");
  if (kind == static_cast<int>(ProblemType::electron_hubble))
    PARTHENON_REQUIRE(electron_v0 > 0.0, "electron_hubble requires problem/v0 > 0");
  if (kind == static_cast<int>(ProblemType::electron_noh)) {
    PARTHENON_REQUIRE(electron_v0 > 0.0, "electron_noh requires problem/v0 > 0");
    PARTHENON_REQUIRE(electron_sound2 < eos.gamma - 1.0,
                      "electron_noh requires sound speed squared below gamma - 1");
  }
  const int wave_flag = pin->GetOrAddInteger("problem", "wave_flag", 3);
  const int kharma_mode = pin->GetOrAddInteger("problem", "kharma_mode", -1);
  const int mode_direction = pin->GetOrAddInteger("problem", "mode_direction", 3);
  const Real mode_phase = pin->GetOrAddReal("problem", "phase", 0.0);
  const Real mode_density0 = pin->GetOrAddReal("problem", "mode_density0", 1.0);
  const Real mode_internal0 = pin->GetOrAddReal("problem", "mode_internal0", 1.0);
  const Real mode_u10 = pin->GetOrAddReal("problem", "mode_u10", 0.0);
  const Real mode_u20 = pin->GetOrAddReal("problem", "mode_u20", 0.0);
  const Real mode_u30 = pin->GetOrAddReal("problem", "mode_u30", 0.0);
  const Real mode_b10 = pin->GetOrAddReal("problem", "mode_b10",
                                          (mode_direction == 0 || mode_direction == 3) ? 1.0 : 0.0);
  const Real mode_b20 = pin->GetOrAddReal("problem", "mode_b20", mode_direction == 1 ? 1.0 : 0.0);
  const Real mode_b30 = pin->GetOrAddReal("problem", "mode_b30", mode_direction == 2 ? 1.0 : 0.0);
  const Real b_parallel = pin->GetOrAddReal("problem", "b_par", 1.0);
  const Real b_perpendicular = pin->GetOrAddReal("problem", "b_perp", 0.1);
  const Real v_perpendicular = pin->GetOrAddReal("problem", "v_perp", b_perpendicular);
  const Real v_parallel = pin->GetOrAddReal("problem", "v_par", 0.0);
  const Real pressure0 = pin->GetOrAddReal("problem", "pres", 0.1);
  const Real field_loop_pressure = pin->GetOrAddReal("problem", "pressure", 1.0);
  const bool right_polar = pin->GetOrAddBoolean("problem", "right_polar", true);
  const Real density_left = pin->GetOrAddReal("problem", "density_left", 1.0);
  const Real density_right = pin->GetOrAddReal("problem", "density_right", 0.125);
  const Real pressure_left = pin->GetOrAddReal("problem", "pressure_left", 1.0);
  const Real pressure_right = pin->GetOrAddReal("problem", "pressure_right", 0.1);
  const Real by_left = pin->GetOrAddReal("problem", "b2_left", 1.0);
  const Real by_right = pin->GetOrAddReal("problem", "b2_right", -1.0);
  const Real bx0 = pin->GetOrAddReal("problem", "b1", 0.75);
  const Real wave_bx = pin->GetOrAddReal("problem", "bx0", bx0);
  const Real wave_by = pin->GetOrAddReal("problem", "by0", by_left);
  const Real wave_bz = pin->GetOrAddReal("problem", "bz0", 0.0);
  const Real velocity1_left = pin->GetOrAddReal("problem", "velocity_1_left", 0.0);
  const Real velocity2_left = pin->GetOrAddReal("problem", "velocity_2_left", 0.0);
  const Real velocity3_left = pin->GetOrAddReal("problem", "velocity_3_left", 0.0);
  const Real velocity1_right = pin->GetOrAddReal("problem", "velocity_1_right", 0.0);
  const Real velocity2_right = pin->GetOrAddReal("problem", "velocity_2_right", 0.0);
  const Real velocity3_right = pin->GetOrAddReal("problem", "velocity_3_right", 0.0);
  const Real bz_left = pin->GetOrAddReal("problem", "b3_left", 0.0);
  const Real bz_right = pin->GetOrAddReal("problem", "b3_right", 0.0);
  const Real monopole_sigma_norm = pin->GetOrAddReal("problem", "sigma_norm", 1.0e2);
  const Real monopole_sigma_power = pin->GetOrAddReal("problem", "sigma_pow", -1.0);
  const Real monopole_rho_min = pin->GetOrAddReal("problem", "rhomin", 1.0e-6);
  const Real monopole_u_min = pin->GetOrAddReal("problem", "umin", 1.0e-8);
  const Real monopole_a_norm = pin->GetOrAddReal("problem", "a_norm", 1.0);
  const Real bondi_k = pin->GetOrAddReal("problem", "k_adi", 1.0);
  const Real bondi_critical_radius = pin->GetOrAddReal("problem", "r_crit", 8.0);
  const Real bondi_sigma_target = pin->GetOrAddReal("problem", "sigma_target", 1.0e3);
  const Real bondi_sigma_rmin = pin->GetOrAddReal("problem", "sigma_rmin", 1.9);
  const bool torus_fm = pin->GetOrAddBoolean("problem", "fm_torus", true);
  const bool torus_chakrabarti = pin->GetOrAddBoolean("problem", "chakrabarti_torus", false);
  const Real torus_n_input = pin->GetOrAddReal("problem", "n_param", 0.0);
  const bool torus_prograde = pin->GetOrAddBoolean("problem", "prograde", true);
  const Real torus_edge = pin->GetOrAddReal("problem", "r_edge", 6.0);
  const Real torus_peak = pin->GetOrAddReal("problem", "r_peak", 12.0);
  const Real torus_density_peak = pin->GetOrAddReal("problem", "rho_max", 1.0);
  const Real torus_density_atmosphere = pin->GetOrAddReal("problem", "rho_min", eos.density_floor);
  const Real torus_density_power = pin->GetOrAddReal("problem", "rho_pow", -1.5);
  const Real torus_pressure_atmosphere =
      pin->GetOrAddReal("problem", "pgas_min", eos.pressure_floor);
  const Real torus_pressure_power = pin->GetOrAddReal("problem", "pgas_pow", -2.5);
  const Real torus_tilt = pin->GetOrAddReal("problem", "tilt_angle", 0.0);
  const Real torus_perturbation = pin->GetOrAddReal("problem", "pert_amp", 0.0);
  const Real potential_beta_min = pin->GetOrAddReal("problem", "potential_beta_min", 100.0);
  const Real potential_cutoff = pin->GetOrAddReal("problem", "potential_cutoff", 0.2);
  const Real potential_falloff = pin->GetOrAddReal("problem", "potential_falloff", 0.0);
  const Real potential_radius_power = pin->GetOrAddReal("problem", "potential_r_pow", 0.0);
  const Real potential_density_power = pin->GetOrAddReal("problem", "potential_rho_pow", 1.0);
  const bool vertical_field = pin->GetOrAddBoolean("problem", "vertical_field", false);
  const bool magnetized_torus_problem =
      problem == ProblemType::gr_torus_sane || problem == ProblemType::gr_chakrabarti_torus;
  const bool chakrabarti_problem = problem == ProblemType::gr_chakrabarti_torus;
  PARTHENON_REQUIRE(!magnetized_torus_problem ||
                        (physics == relativity::HydroMode::gr && eos.HasEnergy()),
                    "PANGU magnetized torus problems require ideal-GR MHD");
  PARTHENON_REQUIRE(problem != ProblemType::gr_torus_sane || torus_fm,
                    "PANGU gr_torus_sane requires problem/fm_torus=true");
  PARTHENON_REQUIRE(problem != ProblemType::gr_torus_sane || !torus_chakrabarti,
                    "PANGU gr_torus_sane is the Fishbone--Moncrief branch");
  PARTHENON_REQUIRE(problem != ProblemType::gr_torus_sane || torus_tilt == 0.0,
                    "PANGU gr_torus_sane currently requires tilt_angle=0");
  PARTHENON_REQUIRE(!chakrabarti_problem || (!torus_fm && torus_chakrabarti),
                    "gr_chakrabarti_torus requires fm_torus=false and "
                    "chakrabarti_torus=true");
  PARTHENON_REQUIRE(!chakrabarti_problem ||
                        (std::string(geometry::ConfiguredMetricName()) == "cks" &&
                         std::string(geometry::ConfiguredModeName()) == "dynamic"),
                    "gr_chakrabarti_torus requires -DMETRIC=cks -DMODE=dynamic");
  PARTHENON_REQUIRE(!chakrabarti_problem || (torus_tilt > -90.0 && torus_tilt < 90.0),
                    "gr_chakrabarti_torus requires tilt_angle in (-90,90) degrees");
  PARTHENON_REQUIRE(!magnetized_torus_problem || !vertical_field,
                    "PANGU magnetized torus problems use poloidal loops, not vertical_field");
  PARTHENON_REQUIRE(
      !magnetized_torus_problem ||
          (torus_perturbation >= 0.0 && torus_perturbation < 1.0 && potential_beta_min > 0.0),
      "PANGU magnetized torus problems require pert_amp in [0,1) and "
      "potential_beta_min>0");
  PARTHENON_REQUIRE(!magnetized_torus_problem ||
                        (torus_edge > 0.0 && torus_peak > torus_edge && torus_density_peak > 0.0),
                    "PANGU magnetized torus problems require 0<r_edge<r_peak and rho_max>0");
  PARTHENON_REQUIRE(problem != ProblemType::magnetised_bondi_mks ||
                        (physics == relativity::HydroMode::gr && eos.HasEnergy()),
                    "magnetised_bondi_mks requires ideal-GR MHD");
  PARTHENON_REQUIRE(problem != ProblemType::magnetised_bondi_mks ||
                        std::string(geometry::ConfiguredMetricName()) == "mks",
                    "magnetised_bondi_mks requires -DMETRIC=mks -DMODE=static");
  PARTHENON_REQUIRE(problem != ProblemType::magnetised_bondi_mks || spin == 0.0,
                    "magnetised_bondi_mks currently implements Schwarzschild Bondi flow (a=0)");
  PARTHENON_REQUIRE(problem != ProblemType::magnetised_bondi_mks ||
                        (bondi_k > 0.0 && bondi_critical_radius > 0.0 && bondi_sigma_target > 0.0 &&
                         bondi_sigma_rmin > 0.0),
                    "magnetised_bondi_mks requires positive k_adi, r_crit, sigma_target, and "
                    "sigma_rmin");
  const auto bondi_parameters = bondi::MakeParameters(eos.gamma, bondi_k, bondi_critical_radius,
                                                      excision_density, excision_pressure);
  Real chakrabarti_c = 0.0;
  Real chakrabarti_n = 0.0;
  if (chakrabarti_problem)
    CalculateChakrabartiConstants(spin, torus_edge, torus_peak, torus_n_input, chakrabarti_c,
                                  chakrabarti_n);
  PARTHENON_REQUIRE(!chakrabarti_problem || (std::isfinite(chakrabarti_c) && chakrabarti_c > 0.0 &&
                                             std::isfinite(chakrabarti_n) && chakrabarti_n > 0.0 &&
                                             chakrabarti_n < 1.0),
                    "invalid Chakrabarti angular-momentum constants");
  constexpr Real degrees_to_radians = 3.1415926535897932384626433832795 / 180.0;
  SaneTorusParameters torus{};
  torus.spin = spin;
  torus.gamma = eos.gamma;
  torus.density_excision = excision_density;
  torus.pressure_excision = excision_pressure;
  torus.density_atmosphere = torus_density_atmosphere;
  torus.density_power = torus_density_power;
  torus.pressure_atmosphere = torus_pressure_atmosphere;
  torus.pressure_power = torus_pressure_power;
  torus.density_peak = torus_density_peak;
  torus.radius_edge = torus_edge;
  torus.radius_peak = torus_peak;
  torus.angular_momentum = TorusAngularMomentum(spin, torus_peak, torus_prograde);
  torus.potential_cutoff = potential_cutoff;
  torus.potential_falloff = potential_falloff;
  torus.potential_radius_power = potential_radius_power;
  torus.potential_density_power = potential_density_power;
  torus.tilt = torus_tilt * degrees_to_radians;
  torus.sin_tilt = sin(torus.tilt);
  torus.cos_tilt = cos(torus.tilt);
  torus.chakrabarti_c = chakrabarti_c;
  torus.chakrabarti_n = chakrabarti_n;
  torus.prograde = torus_prograde;
  torus.fishbone_moncrief = !chakrabarti_problem;
  torus.log_enthalpy_edge = TorusLogEnthalpy(torus, torus_edge, 1.0);
  const Real torus_log_peak = TorusLogEnthalpy(torus, torus_peak, 1.0) - torus.log_enthalpy_edge;
  const Real torus_pressure_over_density_peak =
      (eos.gamma - 1.0) / eos.gamma * (exp(torus_log_peak) - 1.0);
  torus.density_normalization =
      pow(torus_pressure_over_density_peak, 1.0 / (eos.gamma - 1.0)) / torus_density_peak;
  PARTHENON_REQUIRE(!magnetized_torus_problem ||
                        (std::isfinite(torus_log_peak) && torus_log_peak > 0.0 &&
                         std::isfinite(torus.density_normalization) &&
                         torus.density_normalization > 0.0),
                    "PANGU magnetized torus has invalid peak enthalpy or density normalization");
  const Real xmin = block->pmy_mesh->mesh_size.xmin(X1DIR);
  const Real xmax = block->pmy_mesh->mesh_size.xmax(X1DIR);
  const Real ymin = block->pmy_mesh->mesh_size.xmin(X2DIR);
  const Real ymax = block->pmy_mesh->mesh_size.xmax(X2DIR);
  const Real xlength = xmax - xmin;
  const Real ylength = ymax - ymin;
  const Real diagonal = sqrt(xlength * xlength + ylength * ylength);
  constexpr Real pi = 3.1415926535897932384626433832795;
  const Real wave_number = 2.0 * pi / xlength;
  const Real polarization = right_polar ? 1.0 : -1.0;
  Kokkos::Random_XorShift64_Pool<> torus_random_pool(block->gid);
  const bool kharma_modes_problem = problem_name == "sr_mhd_modes";
  PARTHENON_REQUIRE(!kharma_modes_problem || physics == relativity::HydroMode::sr,
                    "sr_mhd_modes requires mhd/physics=sr");
  PARTHENON_REQUIRE(!kharma_modes_problem || (kharma_mode >= 0 && kharma_mode <= 3),
                    "sr_mhd_modes requires problem/kharma_mode in [0,3]");
  PARTHENON_REQUIRE(!kharma_modes_problem || mode_direction == 0 || mode_direction == 3,
                    "sr_mhd_modes currently supports KHARMA direction 0 or 3");
  PARTHENON_REQUIRE(problem != ProblemType::relativistic_linear_wave || kharma_modes_problem ||
                        wave_flag == 3,
                    "legacy SR/GR MHD linear wave requires problem/wave_flag=3");
  const Real velocity_squared = vx0 * vx0 + vy0 * vy0 + vz0 * vz0;
  PARTHENON_REQUIRE(problem != ProblemType::relativistic_linear_wave || velocity_squared < 1.0,
                    "relativistic MHD linear-wave background velocity must be subluminal");
  const Real wave_lorentz =
      problem == ProblemType::relativistic_linear_wave && !kharma_modes_problem
          ? 1.0 / sqrt(1.0 - velocity_squared)
          : 1.0;

  Real mode_drho = 0.0, mode_du = 0.0;
  Real mode_du1 = 0.0, mode_du2 = 0.0, mode_du3 = 0.0;
  Real mode_db1 = 0.0, mode_db2 = 0.0, mode_db3 = 0.0;
  if (kharma_modes_problem && mode_direction == 3) {
    if (kharma_mode == 0) {
      mode_drho = 1.0;
    } else if (kharma_mode == 1) {
      mode_drho = 0.558104461559;
      mode_du = 0.744139282078;
      mode_du1 = -0.277124827421;
      mode_du2 = 0.0630348927707;
      mode_db1 = -0.164323721928;
      mode_db2 = 0.164323721928;
    } else if (kharma_mode == 2) {
      mode_du3 = 0.480384461415;
      mode_db3 = 0.877058019307;
    } else {
      mode_drho = 0.476395427447;
      mode_du = 0.635193903263;
      mode_du1 = -0.102965815319;
      mode_du2 = -0.316873207561;
      mode_db1 = 0.359559114174;
      mode_db2 = -0.359559114174;
    }
  } else if (kharma_modes_problem) {
    if (kharma_mode == 0) {
      mode_drho = 1.0;
    } else if (kharma_mode == 1) {
      mode_drho = 0.556500332363;
      mode_du = 0.742000443151;
      mode_du1 = -0.282334999306;
      mode_du2 = 0.0367010491491;
      mode_du3 = 0.0367010491491;
      mode_db1 = -0.195509141461;
      mode_db2 = 0.0977545707307;
      mode_db3 = 0.0977545707307;
    } else if (kharma_mode == 2) {
      mode_du2 = -0.339683110243;
      mode_du3 = 0.339683110243;
      mode_db2 = 0.620173672946;
      mode_db3 = -0.620173672946;
    } else {
      mode_drho = 0.481846076323;
      mode_du = 0.642461435098;
      mode_du1 = -0.0832240462505;
      mode_du2 = -0.224080007379;
      mode_du3 = -0.224080007379;
      mode_db1 = 0.406380545676;
      mode_db2 = -0.203190272838;
      mode_db3 = -0.203190272838;
    }
  }
  const Real mode_k1 = mode_direction == 1 ? 0.0 : 2.0 * pi;
  const Real mode_k2 = mode_direction == 2 ? 0.0 : 2.0 * pi;
  const Real mode_k3 = mode_direction == 3 ? 0.0 : 2.0 * pi;
  // AthenaK deliberately materializes all three components of the monopole
  // vector potential before taking its discrete curl.  Keeping that kernel
  // boundary is numerically significant near the excision surface: inlining
  // repeated potential evaluations lets the compiler form a different
  // expression tree, and a few ulps can select a different GRMHD floor branch.
  // Index these scratch arrays relative to the MeshBlock interior bounds so
  // that the construction is also independent of the Parthenon ghost width.
  const int potential_ni = ib.e - ib.s + 2;
  const int potential_nj = jb.e - jb.s + 2;
  const int potential_nk = kb.e - kb.s + 2;
  ParArray4DRaw<Real> monopole_potential;
  if (problem == ProblemType::gr_monopole || problem == ProblemType::magnetised_bondi_mks) {
    monopole_potential = ParArray4DRaw<Real>("PANGU GR monopole vector potential", 3, potential_nk,
                                             potential_nj, potential_ni);
    block->par_for(
        "PANGU GR monopole vector potential", kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const int ii = i - ib.s;
          const int jj = j - jb.s;
          const int kk = k - kb.s;
          const Real xc = coordinates.Xc<1>(i);
          const Real yc = coordinates.Xc<2>(j);
          const Real zc = coordinates.Xc<3>(k);
          const Real xf = coordinates.Xf<1>(i);
          const Real yf = coordinates.Xf<2>(j);
          const Real zf = coordinates.Xf<3>(k);
          const Real potential_normalization =
              kind == static_cast<int>(ProblemType::magnetised_bondi_mks) ? 1.0 : monopole_a_norm;
          monopole_potential(0, kk, jj, ii) =
              MonopolePotential<0>(potential_normalization, xc, yf, zf, spacetime);
          monopole_potential(1, kk, jj, ii) =
              MonopolePotential<1>(potential_normalization, xf, yc, zf, spacetime);
          monopole_potential(2, kk, jj, ii) =
              MonopolePotential<2>(potential_normalization, xf, yf, zc, spacetime);
        });
  }

  ParArray4DRaw<Real> sane_potential;
  if (magnetized_torus_problem) {
    int a1_finer_mask = 0;
    int a2_finer_mask = 0;
    int a3_finer_mask = 0;
    for (const auto& neighbor : block->GetNeighbors()) {
      if (neighbor.loc.level() <= block->loc.level())
        continue;
      const int ox1 = neighbor.offsets(X1DIR);
      const int ox2 = neighbor.offsets(X2DIR);
      const int ox3 = neighbor.offsets(X3DIR);
      if (ox1 == 0)
        a1_finer_mask |= 1 << ((ox2 + 1) * 3 + (ox3 + 1));
      if (ox2 == 0)
        a2_finer_mask |= 1 << ((ox1 + 1) * 3 + (ox3 + 1));
      if (ox3 == 0)
        a3_finer_mask |= 1 << ((ox1 + 1) * 3 + (ox2 + 1));
    }
    sane_potential = ParArray4DRaw<Real>("PANGU SANE torus vector potential", 3, potential_nk,
                                         potential_nj, potential_ni);
    block->par_for(
        "PANGU SANE torus vector potential", kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const int ii = i - ib.s;
          const int jj = j - jb.s;
          const int kk = k - kb.s;
          const Real xc = coordinates.Xc<1>(i);
          const Real yc = coordinates.Xc<2>(j);
          const Real zc = coordinates.Xc<3>(k);
          const Real xf = coordinates.Xf<1>(i);
          const Real yf = coordinates.Xf<2>(j);
          const Real zf = coordinates.Xf<3>(k);
          const Real dx = coordinates.Dxc<X1DIR>(k, j, i);
          const Real dy = coordinates.Dxc<X2DIR>(k, j, i);
          const Real dz = coordinates.Dxc<X3DIR>(k, j, i);
          Real a1 = SaneTorusPotential<0>(torus, xc, yf, zf, spacetime);
          Real a2 = SaneTorusPotential<1>(torus, xf, yc, zf, spacetime);
          Real a3 = SaneTorusPotential<2>(torus, xf, yf, zc, spacetime);
          // At a coarse/fine interface use the mean of the two fine edge
          // samples. This makes the integrated edge potential, and hence the
          // shared magnetic flux, identical on both levels.
          const bool a1_correct = ((a1_finer_mask & (1 << 1)) && j == jb.s) ||
                                  ((a1_finer_mask & (1 << 7)) && j == jb.e + 1) ||
                                  ((a1_finer_mask & (1 << 3)) && k == kb.s) ||
                                  ((a1_finer_mask & (1 << 5)) && k == kb.e + 1) ||
                                  ((a1_finer_mask & (1 << 0)) && j == jb.s && k == kb.s) ||
                                  ((a1_finer_mask & (1 << 2)) && j == jb.s && k == kb.e + 1) ||
                                  ((a1_finer_mask & (1 << 6)) && j == jb.e + 1 && k == kb.s) ||
                                  ((a1_finer_mask & (1 << 8)) && j == jb.e + 1 && k == kb.e + 1);
          if (a1_correct)
            a1 = 0.5 * (SaneTorusPotential<0>(torus, xc - 0.25 * dx, yf, zf, spacetime) +
                        SaneTorusPotential<0>(torus, xc + 0.25 * dx, yf, zf, spacetime));
          const bool a2_correct = ((a2_finer_mask & (1 << 1)) && i == ib.s) ||
                                  ((a2_finer_mask & (1 << 7)) && i == ib.e + 1) ||
                                  ((a2_finer_mask & (1 << 3)) && k == kb.s) ||
                                  ((a2_finer_mask & (1 << 5)) && k == kb.e + 1) ||
                                  ((a2_finer_mask & (1 << 0)) && i == ib.s && k == kb.s) ||
                                  ((a2_finer_mask & (1 << 2)) && i == ib.s && k == kb.e + 1) ||
                                  ((a2_finer_mask & (1 << 6)) && i == ib.e + 1 && k == kb.s) ||
                                  ((a2_finer_mask & (1 << 8)) && i == ib.e + 1 && k == kb.e + 1);
          if (a2_correct)
            a2 = 0.5 * (SaneTorusPotential<1>(torus, xf, yc - 0.25 * dy, zf, spacetime) +
                        SaneTorusPotential<1>(torus, xf, yc + 0.25 * dy, zf, spacetime));
          const bool a3_correct = ((a3_finer_mask & (1 << 1)) && i == ib.s) ||
                                  ((a3_finer_mask & (1 << 7)) && i == ib.e + 1) ||
                                  ((a3_finer_mask & (1 << 3)) && j == jb.s) ||
                                  ((a3_finer_mask & (1 << 5)) && j == jb.e + 1) ||
                                  ((a3_finer_mask & (1 << 0)) && i == ib.s && j == jb.s) ||
                                  ((a3_finer_mask & (1 << 2)) && i == ib.s && j == jb.e + 1) ||
                                  ((a3_finer_mask & (1 << 6)) && i == ib.e + 1 && j == jb.s) ||
                                  ((a3_finer_mask & (1 << 8)) && i == ib.e + 1 && j == jb.e + 1);
          if (a3_correct)
            a3 = 0.5 * (SaneTorusPotential<2>(torus, xf, yf, zc - 0.25 * dz, spacetime) +
                        SaneTorusPotential<2>(torus, xf, yf, zc + 0.25 * dz, spacetime));
          sane_potential(0, kk, jj, ii) = a1;
          sane_potential(1, kk, jj, ii) = a2;
          sane_potential(2, kk, jj, ii) = a3;
        });
  }

  // Set every independently stored face component from an analytic or discrete curl.
  block->par_for(
      "PANGU MHD initial B1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real xf = coordinates.Xf<1>(i);
        const Real xc = coordinates.Xc<1>(i);
        const Real yc = coordinates.Xc<2>(j);
        const Real zc = coordinates.Xc<3>(k);
        const Real yf = coordinates.Xf<2>(j);
        const Real yfp = coordinates.Xf<2>(j + (ndim >= 2));
        Real value =
            kind == static_cast<int>(ProblemType::relativistic_linear_wave)
                ? (kharma_modes_problem
                       ? mode_b10 + amplitude * mode_db1 *
                                        cos(mode_k1 * xf + mode_k2 * yc + mode_k3 * zc + mode_phase)
                       : wave_bx)
                : bx0;
        if (kind == static_cast<int>(ProblemType::gr_monopole) ||
            kind == static_cast<int>(ProblemType::magnetised_bondi_mks)) {
          const int ii = i - ib.s;
          const int jj = j - jb.s;
          const int kk = k - kb.s;
          value = (monopole_potential(2, kk, jj + 1, ii) - monopole_potential(2, kk, jj, ii)) /
                      coordinates.Dxc<X2DIR>(k, j, i) -
                  (monopole_potential(1, kk + 1, jj, ii) - monopole_potential(1, kk, jj, ii)) /
                      coordinates.Dxc<X3DIR>(k, j, i);
        } else if (kind == static_cast<int>(ProblemType::gr_torus_sane) ||
                   kind == static_cast<int>(ProblemType::gr_chakrabarti_torus)) {
          const int ii = i - ib.s;
          const int jj = j - jb.s;
          const int kk = k - kb.s;
          value = (sane_potential(2, kk, jj + 1, ii) - sane_potential(2, kk, jj, ii)) /
                      coordinates.Dxc<X2DIR>(k, j, i) -
                  (sane_potential(1, kk + 1, jj, ii) - sane_potential(1, kk, jj, ii)) /
                      coordinates.Dxc<X3DIR>(k, j, i);
        }
        if (kind == static_cast<int>(ProblemType::cpaw)) {
          value = b_parallel;
        } else if (kind == static_cast<int>(ProblemType::field_loop)) {
          value = (LoopPotential(xf - x0, yfp - y0, radius, amplitude) -
                   LoopPotential(xf - x0, yf - y0, radius, amplitude)) /
                  coordinates.Dxc<X2DIR>(k, j, i);
        } else if (kind == static_cast<int>(ProblemType::orszag_tang)) {
          const Real b0 = 1.0 / sqrt(4.0 * pi);
          value = (OrszagPotential(xf, yfp, b0) - OrszagPotential(xf, yf, b0)) /
                  coordinates.Dxc<X2DIR>(k, j, i);
        }
        bface(0, 0, 0, 0, k, j, i) = value;
      });
  block->par_for(
      "PANGU MHD initial B2", kb.s, kb.e, jb.s, jb.e + (ndim >= 2), ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real xc = coordinates.Xc<1>(i);
        const Real xf = coordinates.Xf<1>(i);
        const Real xfp = coordinates.Xf<1>(i + 1);
        const Real yc = coordinates.Xc<2>(j);
        const Real yf = coordinates.Xf<2>(j);
        const Real zc = coordinates.Xc<3>(k);
        Real value =
            kind == static_cast<int>(ProblemType::relativistic_linear_wave)
                ? (kharma_modes_problem
                       ? mode_b20 + amplitude * mode_db2 *
                                        cos(mode_k1 * xc + mode_k2 * yf + mode_k3 * zc + mode_phase)
                       : wave_by)
                : (xc < x0 ? by_left : by_right);
        if (kind == static_cast<int>(ProblemType::gr_monopole) ||
            kind == static_cast<int>(ProblemType::magnetised_bondi_mks)) {
          const int ii = i - ib.s;
          const int jj = j - jb.s;
          const int kk = k - kb.s;
          value = (monopole_potential(0, kk + 1, jj, ii) - monopole_potential(0, kk, jj, ii)) /
                      coordinates.Dxc<X3DIR>(k, j, i) -
                  (monopole_potential(2, kk, jj, ii + 1) - monopole_potential(2, kk, jj, ii)) /
                      coordinates.Dxc<X1DIR>(k, j, i);
        } else if (kind == static_cast<int>(ProblemType::gr_torus_sane) ||
                   kind == static_cast<int>(ProblemType::gr_chakrabarti_torus)) {
          const int ii = i - ib.s;
          const int jj = j - jb.s;
          const int kk = k - kb.s;
          value = (sane_potential(0, kk + 1, jj, ii) - sane_potential(0, kk, jj, ii)) /
                      coordinates.Dxc<X3DIR>(k, j, i) -
                  (sane_potential(2, kk, jj, ii + 1) - sane_potential(2, kk, jj, ii)) /
                      coordinates.Dxc<X1DIR>(k, j, i);
        }
        if (kind == static_cast<int>(ProblemType::cpaw)) {
          const Real a_left = (b_perpendicular / wave_number) * cos(wave_number * xf);
          const Real a_right = (b_perpendicular / wave_number) * cos(wave_number * xfp);
          value = -(a_right - a_left) / coordinates.Dxc<X1DIR>(k, j, i);
        } else if (kind == static_cast<int>(ProblemType::field_loop)) {
          value = -(LoopPotential(xfp - x0, yf - y0, radius, amplitude) -
                    LoopPotential(xf - x0, yf - y0, radius, amplitude)) /
                  coordinates.Dxc<X1DIR>(k, j, i);
        } else if (kind == static_cast<int>(ProblemType::orszag_tang)) {
          const Real b0 = 1.0 / sqrt(4.0 * pi);
          value = -(OrszagPotential(xfp, yf, b0) - OrszagPotential(xf, yf, b0)) /
                  coordinates.Dxc<X1DIR>(k, j, i);
        }
        bface(1, 0, 0, 0, k, j, i) = value;
      });
  block->par_for(
      "PANGU MHD initial B3", kb.s, kb.e + (ndim >= 3), jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real xc = coordinates.Xc<1>(i);
        const Real yc = coordinates.Xc<2>(j);
        const Real xf = coordinates.Xf<1>(i);
        const Real xfp = coordinates.Xf<1>(i + 1);
        const Real yf = coordinates.Xf<2>(j);
        const Real yfp = coordinates.Xf<2>(j + 1);
        const Real zf = coordinates.Xf<3>(k);
        const Real zc = coordinates.Xc<3>(k);
        Real value = 0.0;
        if (kind == static_cast<int>(ProblemType::relativistic_linear_wave))
          value = kharma_modes_problem
                      ? mode_b30 + amplitude * mode_db3 *
                                       cos(mode_k1 * xc + mode_k2 * yc + mode_k3 * zf + mode_phase)
                      : wave_bz;
        else if (kind == static_cast<int>(ProblemType::shock))
          value = xc < x0 ? bz_left : bz_right;
        if (kind == static_cast<int>(ProblemType::gr_monopole) ||
            kind == static_cast<int>(ProblemType::magnetised_bondi_mks)) {
          const int ii = i - ib.s;
          const int jj = j - jb.s;
          const int kk = k - kb.s;
          value = (monopole_potential(1, kk, jj, ii + 1) - monopole_potential(1, kk, jj, ii)) /
                      coordinates.Dxc<X1DIR>(k, j, i) -
                  (monopole_potential(0, kk, jj + 1, ii) - monopole_potential(0, kk, jj, ii)) /
                      coordinates.Dxc<X2DIR>(k, j, i);
        } else if (kind == static_cast<int>(ProblemType::gr_torus_sane) ||
                   kind == static_cast<int>(ProblemType::gr_chakrabarti_torus)) {
          const int ii = i - ib.s;
          const int jj = j - jb.s;
          const int kk = k - kb.s;
          value = (sane_potential(1, kk, jj, ii + 1) - sane_potential(1, kk, jj, ii)) /
                      coordinates.Dxc<X1DIR>(k, j, i) -
                  (sane_potential(0, kk, jj + 1, ii) - sane_potential(0, kk, jj, ii)) /
                      coordinates.Dxc<X2DIR>(k, j, i);
        }
        if (kind == static_cast<int>(ProblemType::cpaw)) {
          const Real xf = coordinates.Xf<1>(i);
          const Real xfp = coordinates.Xf<1>(i + 1);
          const Real a_left =
              polarization * (b_perpendicular / wave_number) * sin(wave_number * xf);
          const Real a_right =
              polarization * (b_perpendicular / wave_number) * sin(wave_number * xfp);
          value = (a_right - a_left) / coordinates.Dxc<X1DIR>(k, j, i);
        }
        bface(2, 0, 0, 0, k, j, i) = value;
      });

  block->par_for(
      "PANGU Newtonian MHD cell initial condition", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coordinates.Xc<1>(i);
        const Real y = coordinates.Xc<2>(j);
        const Real z = coordinates.Xc<3>(k);
        mhd::Primitive state{};
        state.density = 1.0;
        state.pressure = pressure0;
        state.velocity[0] = 0.0;
        state.velocity[1] = 0.0;
        state.velocity[2] = 0.0;
        Real relativistic_internal = 0.0;
        geometry::MetricPoint metric{};
        if (physics == relativity::HydroMode::gr) {
          metric = spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i,
                                      coordinates);
        }
        if (kind == static_cast<int>(ProblemType::shock)) {
          const bool left = x < x0;
          state.density = left ? density_left : density_right;
          state.pressure = left ? pressure_left : pressure_right;
          state.velocity[0] = left ? velocity1_left : velocity1_right;
          state.velocity[1] = left ? velocity2_left : velocity2_right;
          state.velocity[2] = left ? velocity3_left : velocity3_right;
        } else if (kind == static_cast<int>(ProblemType::cpaw)) {
          const Real phase = wave_number * x;
          state.velocity[0] = v_parallel;
          state.velocity[1] = -polarization * v_perpendicular * sin(phase);
          state.velocity[2] = -polarization * v_perpendicular * cos(phase);
        } else if (kind == static_cast<int>(ProblemType::field_loop)) {
          state.pressure = field_loop_pressure;
          state.velocity[0] = vflow * xlength / diagonal;
          state.velocity[1] = vflow * ylength / diagonal;
        } else if (kind == static_cast<int>(ProblemType::orszag_tang)) {
          state.density = 25.0 / (36.0 * pi);
          state.pressure = 5.0 / (12.0 * pi);
          state.velocity[0] = sin(2.0 * pi * y);
          state.velocity[1] = -sin(2.0 * pi * x);
        } else if (kind == static_cast<int>(ProblemType::relativistic_linear_wave)) {
          if (kharma_modes_problem) {
            const Real mode = amplitude * cos(mode_k1 * x + mode_k2 * y + mode_k3 * z + mode_phase);
            state.density = mode_density0 + mode_drho * mode;
            state.pressure = (eos.gamma - 1.0) * (mode_internal0 + mode_du * mode);
            state.velocity[0] = mode_u10 + mode_du1 * mode;
            state.velocity[1] = mode_u20 + mode_du2 * mode;
            state.velocity[2] = mode_u30 + mode_du3 * mode;
          } else {
            state.density = density0 + amplitude * sin(wave_number * x);
            state.pressure = gas_pressure0;
            state.velocity[0] = wave_lorentz * vx0;
            state.velocity[1] = wave_lorentz * vy0;
            state.velocity[2] = wave_lorentz * vz0;
          }
        } else if (kind == static_cast<int>(ProblemType::electron_noh)) {
          const Real pressure =
              electron_zero_internal
                  ? 0.0
                  : electron_density * electron_sound2 /
                        (eos.gamma * (eos.gamma - 1.0) - electron_sound2 * eos.gamma);
          const Real lorentz = 1.0 / sqrt(1.0 - electron_v0 * electron_v0);
          state.density = electron_density;
          state.pressure = pressure;
          state.velocity[0] = electron_noh_centered
                                  ? (x < x0 ? electron_v0 : -electron_v0) * lorentz
                                  : -electron_v0 * lorentz;
        } else if (kind == static_cast<int>(ProblemType::electron_hubble)) {
          const Real normalization = sqrt(eos.gamma * (eos.gamma - 1.0));
          const Real density = electron_mach / electron_v0 * normalization;
          const Real internal = electron_v0 / electron_mach / normalization;
          const Real velocity = electron_v0 * x;
          const Real lorentz = 1.0 / sqrt(1.0 - velocity * velocity);
          state.density = density;
          state.pressure = (eos.gamma - 1.0) * internal;
          state.velocity[0] = velocity * lorentz;
        } else if (kind == static_cast<int>(ProblemType::magnetised_bondi_mks)) {
          bondi::Primitive(bondi_parameters, x, y, z, spacetime, metric, state);
        } else if (kind == static_cast<int>(ProblemType::gr_monopole)) {
          const auto spherical = spacetime.SphericalCoordinates(x, y, z);
          const Real radius_ks = spherical.radius;
          const Real horizon = 1.0 + sqrt(1.0 - spin * spin);
          const Real cutoff = 10.0 * horizon;
          bool excised_cell = false;
          if (decltype(spacetime)::supports_excision)
            excised_cell = radius_ks <= 1.0;
          if (!excised_cell) {
            const Real magnetic_pressure = pow(radius_ks / cutoff, -monopole_sigma_power) /
                                           pow(radius_ks, 4.0) / monopole_sigma_norm;
            state.density = monopole_rho_min + magnetic_pressure;
            relativistic_internal = monopole_u_min + magnetic_pressure;
            state.pressure = (eos.gamma - 1.0) * relativistic_internal;
          } else {
            state.density = excision_density;
            state.pressure = excision_pressure;
            relativistic_internal = excision_pressure / (eos.gamma - 1.0);
          }
        } else if (kind == static_cast<int>(ProblemType::gr_torus_sane) ||
                   kind == static_cast<int>(ProblemType::gr_chakrabarti_torus)) {
          const bool in_torus = SaneTorusPrimitive(
              torus, x, y, z, coordinates.Dxc<X1DIR>(k, j, i), coordinates.Dxc<X2DIR>(k, j, i),
              coordinates.Dxc<X3DIR>(k, j, i), spacetime, metric, state);
          if (in_torus && torus_perturbation > 0.0) {
            auto generator = torus_random_pool.get_state();
            const Real perturbation = 2.0 * torus_perturbation * (generator.frand() - 0.5);
            torus_random_pool.free_state(generator);
            state.pressure *= 1.0 + perturbation;
          }
        }
        if (kind != static_cast<int>(ProblemType::gr_monopole))
          relativistic_internal = state.pressure / (eos.gamma - 1.0);
        if (physics != relativity::HydroMode::gr || decltype(spacetime)::unit_determinant) {
          state.magnetic[0] = 0.5 * (bface(0, 0, 0, 0, k, j, i) + bface(0, 0, 0, 0, k, j, i + 1));
          state.magnetic[1] =
              0.5 * (bface(1, 0, 0, 0, k, j, i) + bface(1, 0, 0, 0, k, j + (ndim >= 2), i));
          state.magnetic[2] =
              0.5 * (bface(2, 0, 0, 0, k, j, i) + bface(2, 0, 0, 0, k + (ndim >= 3), j, i));
        } else {
          const auto x1_left =
              spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i, coordinates);
          const auto x1_right = spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j,
                                                   i + 1, coordinates);
          const auto x2_left =
              spacetime.MetricAt(geometry::Location::face2, geometry_block, k, j, i, coordinates);
          const auto x2_right = spacetime.MetricAt(geometry::Location::face2, geometry_block, k,
                                                   j + (ndim >= 2), i, coordinates);
          const auto x3_left =
              spacetime.MetricAt(geometry::Location::face3, geometry_block, k, j, i, coordinates);
          const auto x3_right = spacetime.MetricAt(geometry::Location::face3, geometry_block,
                                                   k + (ndim >= 3), j, i, coordinates);
          state.magnetic[0] = 0.5 * (bface(0, 0, 0, 0, k, j, i) / x1_left.gdet +
                                     bface(0, 0, 0, 0, k, j, i + 1) / x1_right.gdet);
          state.magnetic[1] = 0.5 * (bface(1, 0, 0, 0, k, j, i) / x2_left.gdet +
                                     bface(1, 0, 0, 0, k, j + (ndim >= 2), i) / x2_right.gdet);
          state.magnetic[2] = 0.5 * (bface(2, 0, 0, 0, k, j, i) / x3_left.gdet +
                                     bface(2, 0, 0, 0, k + (ndim >= 3), j, i) / x3_right.gdet);
        }
        if (!eos.HasEnergy())
          state.pressure = eos.iso_sound_speed * eos.iso_sound_speed * state.density;
        auto converted = eos.PrimitiveToConservedAndFlux(state, 0);
        if (physics != relativity::HydroMode::newtonian) {
          const relativity::MHDPrimitiveState rel_state{
              {state.density,
               {state.velocity[0], state.velocity[1], state.velocity[2]},
               state.pressure},
              {state.magnetic[0], state.magnetic[1], state.magnetic[2]}};
          const auto rel_conserved =
              physics == relativity::HydroMode::sr
                  ? relativity::ConvertSRMHDP2CWithInternalEnergy(rel_state, relativistic_internal,
                                                                  relativistic_eos)
                  : relativity::ConvertGRMHDP2CWithInternalEnergy(rel_state, relativistic_internal,
                                                                  relativistic_eos, metric);
          converted.conserved[mhd::IDN] = rel_conserved.density;
          converted.conserved[mhd::IM1] = rel_conserved.momentum[0];
          converted.conserved[mhd::IM2] = rel_conserved.momentum[1];
          converted.conserved[mhd::IM3] = rel_conserved.momentum[2];
          converted.conserved[mhd::IEN] = rel_conserved.energy;
        }
        for (int n = 0; n < fluid_components; ++n) {
          conserved(n, k, j, i) = converted.conserved[n];
          primitive(n, k, j, i) = n == mhd::IDN   ? state.density
                                  : n == mhd::IV1 ? state.velocity[0]
                                  : n == mhd::IV2 ? state.velocity[1]
                                  : n == mhd::IV3 ? state.velocity[2]
                                  : physics == relativity::HydroMode::newtonian
                                      ? state.pressure
                                      : relativistic_internal;
        }
        bcell(0, k, j, i) = state.magnetic[0];
        bcell(1, k, j, i) = state.magnetic[1];
        bcell(2, k, j, i) = state.magnetic[2];
        for (int scalar = 0; scalar < nscalars; ++scalar) {
          primitive(fluid_components + scalar, k, j, i) = 0.0;
          conserved(fluid_components + scalar, k, j, i) = 0.0;
        }
        divb(0, k, j, i) = 0.0;
        fofc(0, k, j, i) = 0.0;
      });
}

void NormalizeMagnetisedBondi(Mesh* mesh, ParameterInput* pin, MeshData<Real>*) {
  if (pin->GetString("parthenon/job", "problem_id") != "magnetised_bondi_mks")
    return;

  const Real sigma_target = pin->GetReal("problem", "sigma_target");
  const Real sigma_rmin = pin->GetReal("problem", "sigma_rmin");
  Real raw_sigma_max = 0.0;
  for (const auto& block : mesh->block_list) {
    const auto block_data = block->meshblock_data.Get();
    const auto primitive = block_data->PackVariables(std::vector<std::string>{"mhd.prim"});
    const auto bcell = block_data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
    const auto package = block->packages.Get("mhd");
    const auto eos = pangu::eos::ReadRelativistic(*package);
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    const auto coords = block->coords;
    const auto spacetime = geometry::GetGeometry(block->packages);
    const int geometry_block = block->lid;
    Real block_sigma_max = 0.0;
    ParReduce(
        "PANGU magnetised Bondi raw maximum sigma", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
          const auto spherical =
              spacetime.SphericalCoordinates(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k));
          if (spherical.radius < sigma_rmin)
            return;
          const Real internal = primitive(mhd::IPR, k, j, i);
          relativity::MHDPrimitiveState state{
              {primitive(mhd::IDN, k, j, i),
               {primitive(mhd::IV1, k, j, i), primitive(mhd::IV2, k, j, i),
                primitive(mhd::IV3, k, j, i)},
               eos.PressureFromInternalEnergyDensity(internal)},
              {bcell(0, k, j, i), bcell(1, k, j, i), bcell(2, k, j, i)}};
          const auto metric =
              spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
          local = fmax(local, GRMagneticSquared(state, metric) / state.fluid.density);
        },
        Kokkos::Max<Real>(block_sigma_max));
    raw_sigma_max = std::max(raw_sigma_max, block_sigma_max);
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, &raw_sigma_max, 1, MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
#endif
  PARTHENON_REQUIRE(std::isfinite(raw_sigma_max) && raw_sigma_max > 0.0,
                    "magnetised_bondi_mks found no positive finite sigma in its normalization "
                    "domain");
  const Real normalization = sqrt(sigma_target / raw_sigma_max);

  Real achieved_sigma_max = 0.0;
  Real divergence_max = 0.0;
  for (const auto& block : mesh->block_list) {
    const auto block_data = block->meshblock_data.Get();
    auto conserved = block_data->PackVariables(std::vector<std::string>{"mhd.cons"});
    auto primitive = block_data->PackVariables(std::vector<std::string>{"mhd.prim"});
    auto bcell = block_data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
    auto divb = block_data->PackVariables(std::vector<std::string>{"mhd.divb"});
    auto face = block_data->Get("mhd.b_face").data;
    const auto package = block->packages.Get("mhd");
    const auto eos = pangu::eos::ReadRelativistic(*package);
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    const auto coords = block->coords;
    const auto spacetime = geometry::GetGeometry(block->packages);
    const int geometry_block = block->lid;
    const int ndim = block->pmy_mesh->ndim;
    block->par_for(
        "PANGU normalize magnetised Bondi B1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          face(0, 0, 0, 0, k, j, i) *= normalization;
        });
    block->par_for(
        "PANGU normalize magnetised Bondi B2", kb.s, kb.e, jb.s, jb.e + (ndim >= 2), ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          face(1, 0, 0, 0, k, j, i) *= normalization;
        });
    block->par_for(
        "PANGU normalize magnetised Bondi B3", kb.s, kb.e + (ndim >= 3), jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          face(2, 0, 0, 0, k, j, i) *= normalization;
        });
    block->par_for(
        "PANGU finalize magnetised Bondi state", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real internal = primitive(mhd::IPR, k, j, i);
          relativity::MHDPrimitiveState state{
              {primitive(mhd::IDN, k, j, i),
               {primitive(mhd::IV1, k, j, i), primitive(mhd::IV2, k, j, i),
                primitive(mhd::IV3, k, j, i)},
               eos.PressureFromInternalEnergyDensity(internal)},
              {}};
          PhysicalCellMagnetic(spacetime, geometry_block, coords, face, k, j, i, ndim,
                               state.magnetic);
          for (int axis = 0; axis < 3; ++axis)
            bcell(axis, k, j, i) = state.magnetic[axis];
          Real divergence = (face(0, 0, 0, 0, k, j, i + 1) - face(0, 0, 0, 0, k, j, i)) /
                            coords.Dxc<X1DIR>(k, j, i);
          if (ndim >= 2)
            divergence += (face(1, 0, 0, 0, k, j + 1, i) - face(1, 0, 0, 0, k, j, i)) /
                          coords.Dxc<X2DIR>(k, j, i);
          if (ndim >= 3)
            divergence += (face(2, 0, 0, 0, k + 1, j, i) - face(2, 0, 0, 0, k, j, i)) /
                          coords.Dxc<X3DIR>(k, j, i);
          divb(0, k, j, i) = divergence;
          const auto metric =
              spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
          const auto converted =
              relativity::ConvertGRMHDP2CWithInternalEnergy(state, internal, eos, metric);
          conserved(mhd::IDN, k, j, i) = converted.density;
          conserved(mhd::IM1, k, j, i) = converted.momentum[0];
          conserved(mhd::IM2, k, j, i) = converted.momentum[1];
          conserved(mhd::IM3, k, j, i) = converted.momentum[2];
          conserved(mhd::IEN, k, j, i) = converted.energy;
        });
    Real block_sigma_max = 0.0;
    Real block_divergence_max = 0.0;
    ParReduce(
        "PANGU magnetised Bondi achieved maximum sigma", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
          const auto spherical =
              spacetime.SphericalCoordinates(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k));
          if (spherical.radius < sigma_rmin)
            return;
          const Real internal = primitive(mhd::IPR, k, j, i);
          relativity::MHDPrimitiveState state{
              {primitive(mhd::IDN, k, j, i),
               {primitive(mhd::IV1, k, j, i), primitive(mhd::IV2, k, j, i),
                primitive(mhd::IV3, k, j, i)},
               eos.PressureFromInternalEnergyDensity(internal)},
              {bcell(0, k, j, i), bcell(1, k, j, i), bcell(2, k, j, i)}};
          const auto metric =
              spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
          local = fmax(local, GRMagneticSquared(state, metric) / state.fluid.density);
        },
        Kokkos::Max<Real>(block_sigma_max));
    ParReduce(
        "PANGU magnetised Bondi initial divergence audit", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
          local = fmax(local, fabs(divb(0, k, j, i)));
        },
        Kokkos::Max<Real>(block_divergence_max));
    achieved_sigma_max = std::max(achieved_sigma_max, block_sigma_max);
    divergence_max = std::max(divergence_max, block_divergence_max);
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &achieved_sigma_max, 1, MPI_PARTHENON_REAL,
                                    MPI_MAX, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, &divergence_max, 1, MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
#endif
  if (Globals::my_rank == 0) {
    std::printf("PANGU magnetised Bondi initialization: sigma_raw=%.17e bnorm=%.17e "
                "sigma(r>=%.17e)=%.17e max|divB|=%.17e\n",
                static_cast<double>(raw_sigma_max), static_cast<double>(normalization),
                static_cast<double>(sigma_rmin), static_cast<double>(achieved_sigma_max),
                static_cast<double>(divergence_max));
  }
}

void NormalizeSaneTorus(Mesh* mesh, ParameterInput* pin, MeshData<Real>*) {
  const auto problem_id = pin->GetString("parthenon/job", "problem_id");
  if (problem_id != "gr_torus_sane" && problem_id != "gr_chakrabarti_torus" &&
      problem_id != "gr_chakrabarti_torus_sane")
    return;
  PARTHENON_REQUIRE(mesh->DefaultNumPartitions() == 1,
                    "magnetized torus initialization currently requires one MeshData partition "
                    "per MPI "
                    "rank (leave pack_size/packs_per_rank at their defaults)");

  const Real beta_min = pin->GetReal("problem", "potential_beta_min");
  Real pressure_max = 0.0;
  Real magnetic_squared_max = 0.0;
  for (const auto& block : mesh->block_list) {
    const auto block_data = block->meshblock_data.Get();
    const auto primitive = block_data->PackVariables(std::vector<std::string>{"mhd.prim"});
    const auto bcell = block_data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
    const auto package = block->packages.Get("mhd");
    const auto geometry_package = block->packages.Get("geometry");
    const auto eos = pangu::eos::ReadRelativistic(*package);
    const Real spin = geometry_package->Param<Real>("bh_spin");
    const auto background =
        static_cast<geometry::Background>(geometry_package->Param<int>("background"));
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    const auto coords = block->coords;
    const auto spacetime = geometry::GetGeometry(block->packages);
    const int geometry_block = block->lid;
    Real block_pressure_max = 0.0;
    ParReduce(
        "PANGU SANE torus maximum gas pressure", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
          local = fmax(local, eos.PressureFromInternalEnergyDensity(
                                  primitive(mhd::IPR, k, j, i)));
        },
        Kokkos::Max<Real>(block_pressure_max));
    pressure_max = std::max(pressure_max, block_pressure_max);

    Real block_magnetic_squared_max = 0.0;
    ParReduce(
        "PANGU SANE torus maximum magnetic pressure", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
          relativity::MHDPrimitiveState state{};
          const Real internal = primitive(mhd::IPR, k, j, i);
          state.fluid = {primitive(mhd::IDN, k, j, i),
                         {primitive(mhd::IV1, k, j, i), primitive(mhd::IV2, k, j, i),
                          primitive(mhd::IV3, k, j, i)},
                         eos.PressureFromInternalEnergyDensity(internal)};
          state.magnetic[0] = bcell(0, k, j, i);
          state.magnetic[1] = bcell(1, k, j, i);
          state.magnetic[2] = bcell(2, k, j, i);
          const auto metric =
              spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
          local = fmax(local, GRMagneticSquared(state, metric));
        },
        Kokkos::Max<Real>(block_magnetic_squared_max));
    magnetic_squared_max = std::max(magnetic_squared_max, block_magnetic_squared_max);
  }

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, &pressure_max, 1, MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &magnetic_squared_max, 1, MPI_PARTHENON_REAL,
                                    MPI_MAX, MPI_COMM_WORLD));
#endif
  PARTHENON_REQUIRE(std::isfinite(pressure_max) && pressure_max > 0.0,
                    "magnetized torus initialization found no positive gas-pressure maximum");
  PARTHENON_REQUIRE(std::isfinite(magnetic_squared_max) && magnetic_squared_max > 0.0,
                    "magnetized torus initialization found no positive magnetic-pressure maximum");
  const Real normalization = sqrt((pressure_max / (0.5 * magnetic_squared_max)) / beta_min);

  Real divergence_max = 0.0;
  for (const auto& block : mesh->block_list) {
    const auto block_data = block->meshblock_data.Get();
    auto conserved = block_data->PackVariables(std::vector<std::string>{"mhd.cons"});
    auto primitive = block_data->PackVariables(std::vector<std::string>{"mhd.prim"});
    auto bcell = block_data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
    auto divb = block_data->PackVariables(std::vector<std::string>{"mhd.divb"});
    auto face = block_data->Get("mhd.b_face").data;
    const auto package = block->packages.Get("mhd");
    const auto geometry_package = block->packages.Get("geometry");
    const auto eos = pangu::eos::ReadRelativistic(*package);
    const Real spin = geometry_package->Param<Real>("bh_spin");
    const auto background =
        static_cast<geometry::Background>(geometry_package->Param<int>("background"));
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    const auto coords = block->coords;
    const auto spacetime = geometry::GetGeometry(block->packages);
    const int geometry_block = block->lid;
    const int ndim = block->pmy_mesh->ndim;
    block->par_for(
        "PANGU normalize SANE torus B1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          face(0, 0, 0, 0, k, j, i) *= normalization;
        });
    block->par_for(
        "PANGU normalize SANE torus B2", kb.s, kb.e, jb.s, jb.e + (ndim >= 2), ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          face(1, 0, 0, 0, k, j, i) *= normalization;
        });
    block->par_for(
        "PANGU normalize SANE torus B3", kb.s, kb.e + (ndim >= 3), jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          face(2, 0, 0, 0, k, j, i) *= normalization;
        });
    block->par_for(
        "PANGU finalize SANE torus magnetic and conserved state", kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          relativity::MHDPrimitiveState state{};
          const Real internal = primitive(mhd::IPR, k, j, i);
          state.fluid = {primitive(mhd::IDN, k, j, i),
                         {primitive(mhd::IV1, k, j, i), primitive(mhd::IV2, k, j, i),
                          primitive(mhd::IV3, k, j, i)},
                         eos.PressureFromInternalEnergyDensity(internal)};
          PhysicalCellMagnetic(spacetime, geometry_block, coords, face, k, j, i, ndim,
                               state.magnetic);
          for (int axis = 0; axis < 3; ++axis)
            bcell(axis, k, j, i) = state.magnetic[axis];
          Real divergence = (face(0, 0, 0, 0, k, j, i + 1) - face(0, 0, 0, 0, k, j, i)) /
                            coords.Dxc<X1DIR>(k, j, i);
          if (ndim >= 2)
            divergence += (face(1, 0, 0, 0, k, j + 1, i) - face(1, 0, 0, 0, k, j, i)) /
                          coords.Dxc<X2DIR>(k, j, i);
          if (ndim >= 3)
            divergence += (face(2, 0, 0, 0, k + 1, j, i) - face(2, 0, 0, 0, k, j, i)) /
                          coords.Dxc<X3DIR>(k, j, i);
          divb(0, k, j, i) = divergence;
          const auto metric =
              spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
          const auto converted =
              relativity::ConvertGRMHDP2CWithInternalEnergy(state, internal, eos, metric);
          conserved(mhd::IDN, k, j, i) = converted.density;
          conserved(mhd::IM1, k, j, i) = converted.momentum[0];
          conserved(mhd::IM2, k, j, i) = converted.momentum[1];
          conserved(mhd::IM3, k, j, i) = converted.momentum[2];
          conserved(mhd::IEN, k, j, i) = converted.energy;
        });
    Real block_divergence_max = 0.0;
    ParReduce(
        "PANGU SANE torus initial divergence audit", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
          local = fmax(local, fabs(divb(0, k, j, i)));
        },
        Kokkos::Max<Real>(block_divergence_max));
    divergence_max = std::max(divergence_max, block_divergence_max);
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, &divergence_max, 1, MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
#endif
  if (Globals::my_rank == 0) {
    const Real achieved_beta =
        pressure_max / (0.5 * magnetic_squared_max * normalization * normalization);
    std::printf("PANGU magnetized torus initialization: pmax=%.17e bsqmax(raw)=%.17e "
                "bnorm=%.17e "
                "beta=%.17e max|divB|=%.17e\n",
                static_cast<double>(pressure_max), static_cast<double>(magnetic_squared_max),
                static_cast<double>(normalization), static_cast<double>(achieved_beta),
                static_cast<double>(divergence_max));
  }
}

void MHDAfterLoop(Mesh* mesh, ParameterInput* pin, SimTime&) {
  // Dense CSV is convenient for compact regression problems but duplicates
  // PHDF output and becomes multi-gigabyte at production resolution.
  if (!pin->GetBoolean("problem", "write_final_csv"))
    return;
  std::vector<ProfileRow> rows;
  for (const auto& block : mesh->block_list) {
    const auto data = block->meshblock_data.Get();
    const auto primitive = data->Get("mhd.prim").data.GetHostMirrorAndCopy();
    const auto bcell = data->Get("mhd.b_cell").data.GetHostMirrorAndCopy();
    const auto divb = data->Get("mhd.divb").data.GetHostMirrorAndCopy();
    const auto package = block->packages.Get("mhd");
    const auto eos = GetEos(package);
  const auto relativistic_eos = pangu::eos::ReadRelativistic(*package);
    const auto physics = static_cast<relativity::HydroMode>(package->Param<int>("physics_mode"));
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          const Real density = primitive(mhd::IDN, k, j, i);
          rows.push_back({block->coords.Xc<1>(i), block->coords.Xc<2>(j), block->coords.Xc<3>(k),
                          density, primitive(mhd::IV1, k, j, i), primitive(mhd::IV2, k, j, i),
                          primitive(mhd::IV3, k, j, i),
                          eos.HasEnergy() ? (physics == relativity::HydroMode::newtonian
                                                 ? primitive(mhd::IPR, k, j, i)
                                                 : (eos.gamma - 1.0) * primitive(mhd::IPR, k, j, i))
                                          : eos.iso_sound_speed * eos.iso_sound_speed * density,
                          bcell(0, k, j, i), bcell(1, k, j, i), bcell(2, k, j, i),
                          divb(0, k, j, i)});
        }
      }
    }
  }
  std::sort(rows.begin(), rows.end(), [](const ProfileRow& left, const ProfileRow& right) {
    if (left.z != right.z)
      return left.z < right.z;
    if (left.y != right.y)
      return left.y < right.y;
    return left.x < right.x;
  });
  char output_name[96];
  if (Globals::nranks == 1)
    std::snprintf(output_name, sizeof(output_name), "pangu-mhd-final.csv");
  else
    std::snprintf(output_name, sizeof(output_name), "pangu-mhd-final.rank%05d.csv",
                  Globals::my_rank);
  auto* output = std::fopen(output_name, "w");
  PARTHENON_REQUIRE(output != nullptr, "Unable to write PANGU MHD final profile");
  std::fprintf(output, "x,y,z,density,velocity1,velocity2,velocity3,pressure,B1,B2,B3,divB\n");
  for (const auto& row : rows) {
    std::fprintf(output,
                 "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                 static_cast<double>(row.x), static_cast<double>(row.y), static_cast<double>(row.z),
                 static_cast<double>(row.density), static_cast<double>(row.velocity1),
                 static_cast<double>(row.velocity2), static_cast<double>(row.velocity3),
                 static_cast<double>(row.pressure), static_cast<double>(row.magnetic1),
                 static_cast<double>(row.magnetic2), static_cast<double>(row.magnetic3),
                 static_cast<double>(row.divb));
  }
  std::fclose(output);
}

} // namespace pangu::pgen
