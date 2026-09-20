#ifndef PANGU_RELATIVITY_RELATIVISTIC_MHD_H_
#define PANGU_RELATIVITY_RELATIVISTIC_MHD_H_

#include <limits>
#include <memory>
#include <string>

#include <parthenon/parthenon.hpp>

#include "relativity/relativistic_hydro.h"

namespace pangu::relativity {

struct MHDPrimitiveState {
  HydroPrimitiveState fluid{};
  parthenon::Real magnetic[3]{};
};

struct MHDC2PResult {
  MHDPrimitiveState primitive{};
  // AthenaK stores the ideal-gas primitive energy as internal-energy density
  // and only converts it to pressure when evaluating the EOS.  Keep the exact
  // C2P value alongside the pressure-form state used by PANGU's kernels.
  parthenon::Real internal_energy = 0.0;
  bool success = true;
  bool density_floor = false;
  bool pressure_floor = false;
  bool lorentz_ceiling = false;
  bool sigma_ceiling = false;
  int iterations = 0;
};

KOKKOS_INLINE_FUNCTION parthenon::Real
ComputeComovingMagneticFieldSquared(const MHDPrimitiveState& primitive,
                                    const geometry::MetricPoint& metric) {
  const auto& fluid = primitive.fluid;
  parthenon::Real spatial_u2 = 0.0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      spatial_u2 += metric.lower[a + 1][b + 1] * fluid.u[a] * fluid.u[b];
  const parthenon::Real lorentz = sqrt(1.0 + spatial_u2);
  parthenon::Real u_upper[4]{lorentz / metric.lapse, 0.0, 0.0, 0.0};
  for (int axis = 0; axis < 3; ++axis)
    u_upper[axis + 1] = fluid.u[axis] - metric.lapse * lorentz * metric.upper[0][axis + 1];
  parthenon::Real u_lower[4]{};
  for (int mu = 0; mu < 4; ++mu)
    for (int nu = 0; nu < 4; ++nu)
      u_lower[mu] += metric.lower[mu][nu] * u_upper[nu];

  const parthenon::Real b_upper0 = u_lower[1] * primitive.magnetic[0] +
                                   u_lower[2] * primitive.magnetic[1] +
                                   u_lower[3] * primitive.magnetic[2];
  parthenon::Real b_upper[4]{b_upper0, 0.0, 0.0, 0.0};
  for (int axis = 0; axis < 3; ++axis)
    b_upper[axis + 1] = (primitive.magnetic[axis] + b_upper0 * u_upper[axis + 1]) / u_upper[0];
  parthenon::Real magnetic_squared = 0.0;
  for (int mu = 0; mu < 4; ++mu)
    for (int nu = 0; nu < 4; ++nu)
      magnetic_squared += metric.lower[mu][nu] * b_upper[mu] * b_upper[nu];
  return magnetic_squared;
}

KOKKOS_INLINE_FUNCTION HydroConservedState
ConvertSRMHDP2CWithInternalEnergy(const MHDPrimitiveState& primitive,
                                  const parthenon::Real internal,
                                  const eos::RelativisticEOS& eos) {
  HydroConservedState conserved{};
  const auto& fluid = primitive.fluid;
  const parthenon::Real u0 =
      sqrt(1.0 + fluid.u[0] * fluid.u[0] + fluid.u[1] * fluid.u[1] + fluid.u[2] * fluid.u[2]);
  const parthenon::Real b0 = primitive.magnetic[0] * fluid.u[0] +
                             primitive.magnetic[1] * fluid.u[1] +
                             primitive.magnetic[2] * fluid.u[2];
  parthenon::Real b[3];
  parthenon::Real bsq = -b0 * b0;
  for (int axis = 0; axis < 3; ++axis) {
    b[axis] = (primitive.magnetic[axis] + b0 * fluid.u[axis]) / u0;
    bsq += b[axis] * b[axis];
  }
  const parthenon::Real total_u02 = (fluid.density + eos.gamma * internal + bsq) * u0 * u0;
  conserved.density = fluid.density * u0;
  conserved.energy = total_u02 - b0 * b0 - (fluid.pressure + 0.5 * bsq) - conserved.density;
  for (int axis = 0; axis < 3; ++axis)
    conserved.momentum[axis] = total_u02 * fluid.u[axis] / u0 - b0 * b[axis];
  return conserved;
}

KOKKOS_INLINE_FUNCTION HydroConservedState ConvertSRMHDP2C(const MHDPrimitiveState& primitive,
                                                           const eos::RelativisticEOS& eos) {
  return ConvertSRMHDP2CWithInternalEnergy(
      primitive, eos.InternalEnergyDensity(primitive.fluid.pressure), eos);
}

KOKKOS_INLINE_FUNCTION HydroConservedState ConvertGRMHDP2CWithInternalEnergy(
    const MHDPrimitiveState& primitive, const parthenon::Real internal,
    const eos::RelativisticEOS& eos, const geometry::MetricPoint& metric) {
  HydroConservedState conserved{};
  const auto& fluid = primitive.fluid;
  const parthenon::Real vx = fluid.u[0], vy = fluid.u[1], vz = fluid.u[2];
  // Preserve AthenaK's expression tree.  In the magnetically dominated GR
  // tests, algebraically equivalent loop reductions can move a C2P result by
  // many ulps after the metric contractions are inverted.
  const parthenon::Real q = metric.lower[1][1] * vx * vx + 2.0 * metric.lower[1][2] * vx * vy +
                            2.0 * metric.lower[1][3] * vx * vz + metric.lower[2][2] * vy * vy +
                            2.0 * metric.lower[2][3] * vy * vz + metric.lower[3][3] * vz * vz;
  const parthenon::Real alpha = sqrt(-1.0 / metric.upper[0][0]);
  const parthenon::Real lorentz = sqrt(1.0 + q);
  const parthenon::Real u0 = lorentz / alpha;
  const parthenon::Real u1 = vx - alpha * lorentz * metric.upper[0][1];
  const parthenon::Real u2 = vy - alpha * lorentz * metric.upper[0][2];
  const parthenon::Real u3 = vz - alpha * lorentz * metric.upper[0][3];
  const parthenon::Real u_0 = metric.lower[0][0] * u0 + metric.lower[0][1] * u1 +
                              metric.lower[0][2] * u2 + metric.lower[0][3] * u3;
  const parthenon::Real u_1 = metric.lower[1][0] * u0 + metric.lower[1][1] * u1 +
                              metric.lower[1][2] * u2 + metric.lower[1][3] * u3;
  const parthenon::Real u_2 = metric.lower[2][0] * u0 + metric.lower[2][1] * u1 +
                              metric.lower[2][2] * u2 + metric.lower[2][3] * u3;
  const parthenon::Real u_3 = metric.lower[3][0] * u0 + metric.lower[3][1] * u1 +
                              metric.lower[3][2] * u2 + metric.lower[3][3] * u3;
  const parthenon::Real bx = primitive.magnetic[0];
  const parthenon::Real by = primitive.magnetic[1];
  const parthenon::Real bz = primitive.magnetic[2];
  const parthenon::Real b0 = u_1 * bx + u_2 * by + u_3 * bz;
  const parthenon::Real b1 = (bx + b0 * u1) / u0;
  const parthenon::Real b2 = (by + b0 * u2) / u0;
  const parthenon::Real b3 = (bz + b0 * u3) / u0;
  const parthenon::Real b_0 = metric.lower[0][0] * b0 + metric.lower[0][1] * b1 +
                              metric.lower[0][2] * b2 + metric.lower[0][3] * b3;
  const parthenon::Real b_1 = metric.lower[1][0] * b0 + metric.lower[1][1] * b1 +
                              metric.lower[1][2] * b2 + metric.lower[1][3] * b3;
  const parthenon::Real b_2 = metric.lower[2][0] * b0 + metric.lower[2][1] * b1 +
                              metric.lower[2][2] * b2 + metric.lower[2][3] * b3;
  const parthenon::Real b_3 = metric.lower[3][0] * b0 + metric.lower[3][1] * b1 +
                              metric.lower[3][2] * b2 + metric.lower[3][3] * b3;
  const parthenon::Real magnetic_squared = b0 * b_0 + b1 * b_1 + b2 * b_2 + b3 * b_3;
  const parthenon::Real total_enthalpy = fluid.density + eos.gamma * internal + magnetic_squared;
  const parthenon::Real total_pressure = fluid.pressure + 0.5 * magnetic_squared;
  conserved.density = fluid.density * u0;
  conserved.energy = total_enthalpy * u0 * u_0 - b0 * b_0 + total_pressure + conserved.density;
  conserved.momentum[0] = total_enthalpy * u0 * u_1 - b0 * b_1;
  conserved.momentum[1] = total_enthalpy * u0 * u_2 - b0 * b_2;
  conserved.momentum[2] = total_enthalpy * u0 * u_3 - b0 * b_3;
  conserved.density *= metric.gdet;
  conserved.energy *= metric.gdet;
  for (int axis = 0; axis < 3; ++axis)
    conserved.momentum[axis] *= metric.gdet;
  return conserved;
}

KOKKOS_INLINE_FUNCTION HydroConservedState ConvertGRMHDP2C(const MHDPrimitiveState& primitive,
                                                           const eos::RelativisticEOS& eos,
                                                           const geometry::MetricPoint& metric) {
  return ConvertGRMHDP2CWithInternalEnergy(
      primitive, eos.InternalEnergyDensity(primitive.fluid.pressure), eos, metric);
}

KOKKOS_INLINE_FUNCTION parthenon::Real ComputeSRMHDC2PBracketResidual(const parthenon::Real mu,
                                                                      const parthenon::Real b2,
                                                                      const parthenon::Real rpar,
                                                                      const parthenon::Real r) {
  const parthenon::Real x = 1.0 / (1.0 + mu * b2);
  const parthenon::Real rbar = x * x * r * r + mu * x * (1.0 + x) * rpar * rpar;
  return mu * sqrt(1.0 + rbar) - 1.0;
}

KOKKOS_INLINE_FUNCTION parthenon::Real
ComputeSRMHDC2PResidual(const parthenon::Real mu, const parthenon::Real b2,
                        const parthenon::Real rpar, const parthenon::Real r,
                        const parthenon::Real q, const parthenon::Real density,
                        const eos::RelativisticEOS& eos, const parthenon::Real v0_squared) {
  const parthenon::Real x = 1.0 / (1.0 + mu * b2);
  const parthenon::Real r_squared = r * r;
  const parthenon::Real rpar_squared = rpar * rpar;
  const parthenon::Real rbar_squared =
      x * (x * r_squared + mu * (1.0 + x) * rpar_squared);
  const parthenon::Real mu_x = mu * x;
  const parthenon::Real qbar =
      q - 0.5 * (b2 + mu_x * mu_x * (b2 * r_squared - rpar_squared));
  const parthenon::Real vhat_squared = fmin(mu * mu * rbar_squared, v0_squared);
  const parthenon::Real inverse_lorentz = sqrt(1.0 - vhat_squared);
  const parthenon::Real lorentz = 1.0 / inverse_lorentz;
  const parthenon::Real rhohat = fmax(density * inverse_lorentz, 0.0);
  const parthenon::Real ehat =
      fmax(lorentz * (qbar - mu * rbar_squared) +
               vhat_squared * lorentz * lorentz / (1.0 + lorentz),
           0.0);
  const parthenon::Real pressurehat = eos.PressureFromInternalEnergyDensity(ehat * rhohat);
  const parthenon::Real ahat = pressurehat / (rhohat * (1.0 + ehat));
  const parthenon::Real nua = (1.0 + ahat) * (1.0 + ehat) * inverse_lorentz;
  const parthenon::Real nub = (1.0 + ahat) * (1.0 + qbar - mu * rbar_squared);
  const parthenon::Real nuhat = fmax(nua, nub);
  return mu - 1.0 / (nuhat + mu * rbar_squared);
}

KOKKOS_INLINE_FUNCTION MHDC2PResult SolveSRMHDC2PFromInvariants(
    HydroConservedState conserved, const parthenon::Real magnetic[3],
    const parthenon::Real momentum2, parthenon::Real b2_physical, parthenon::Real rpar,
    const eos::RelativisticEOS& eos, const parthenon::Real gamma_max,
    const parthenon::Real sigma_max) {
  MHDC2PResult result{};
  const parthenon::Real effective_floor = fmax(eos.density_floor, b2_physical / sigma_max);
  if (!(conserved.density >= effective_floor)) {
    conserved.density = effective_floor;
    result.density_floor = true;
    result.sigma_ceiling = effective_floor > eos.density_floor;
  }
  const parthenon::Real internal_energy_floor = eos.InternalEnergyDensityFloor();
  if (!(conserved.energy >= internal_energy_floor + 0.5 * b2_physical)) {
    conserved.energy = internal_energy_floor + 0.5 * b2_physical;
    result.pressure_floor = true;
  }
  const parthenon::Real q = conserved.energy / conserved.density;
  const parthenon::Real r = sqrt(momentum2) / conserved.density;
  const parthenon::Real inverse_sqrt_density = 1.0 / sqrt(conserved.density);
  const parthenon::Real normalized_b1 = magnetic[0] * inverse_sqrt_density;
  const parthenon::Real normalized_b2 = magnetic[1] * inverse_sqrt_density;
  const parthenon::Real normalized_b3 = magnetic[2] * inverse_sqrt_density;
  b2_physical /= conserved.density;
  rpar *= inverse_sqrt_density;
  const parthenon::Real r_squared = r * r;
  const parthenon::Real rpar_squared = rpar * rpar;
  constexpr parthenon::Real maximum_lorentz = 51.0;
  const parthenon::Real v0_squared =
      fmin(r_squared / (1.0 + r_squared),
           1.0 - 1.0 / (maximum_lorentz * maximum_lorentz));
  constexpr int max_iterations = 25;
  constexpr parthenon::Real tolerance = 1.0e-12;
  parthenon::Real lower = 0.0, upper = 1.0;
  parthenon::Real fl = ComputeSRMHDC2PBracketResidual(lower, b2_physical, rpar, r);
  parthenon::Real fu = ComputeSRMHDC2PBracketResidual(upper, b2_physical, rpar, r);
  int iterations = max_iterations;
  if (fabs(lower - upper) < tolerance || fabs(fl) + fabs(fu) < 2.0 * tolerance)
    iterations = -1;
  parthenon::Real root = 0.5 * (lower + upper);
  int iteration = 0;
  for (; iteration < iterations; ++iteration) {
    root = (lower * fu - upper * fl) / (fu - fl);
    const parthenon::Real value = ComputeSRMHDC2PBracketResidual(root, b2_physical, rpar, r);
    if (fabs(upper - lower) < tolerance || fabs(value) < tolerance)
      break;
    if (value * fu < 0.0) {
      lower = upper;
      fl = fu;
      upper = root;
      fu = value;
    } else {
      fl *= 0.5;
      upper = root;
      fu = value;
    }
  }
  const int bracket_iterations = iteration;
  lower = 0.0;
  upper = root;
  fl = ComputeSRMHDC2PResidual(lower, b2_physical, rpar, r, q, conserved.density, eos,
                               v0_squared);
  fu = ComputeSRMHDC2PResidual(upper, b2_physical, rpar, r, q, conserved.density, eos,
                               v0_squared);
  iterations = max_iterations;
  if (fabs(lower - upper) < tolerance || fabs(fl) + fabs(fu) < 2.0 * tolerance)
    iterations = -1;
  root = 0.5 * (lower + upper);
  for (iteration = 0; iteration < iterations; ++iteration) {
    root = (lower * fu - upper * fl) / (fu - fl);
    const parthenon::Real value = ComputeSRMHDC2PResidual(
        root, b2_physical, rpar, r, q, conserved.density, eos, v0_squared);
    if (fabs(upper - lower) < tolerance || fabs(value) < tolerance)
      break;
    if (value * fu < 0.0) {
      lower = upper;
      fl = fu;
      upper = root;
      fu = value;
    } else {
      fl *= 0.5;
      upper = root;
      fu = value;
    }
  }
  result.iterations = bracket_iterations > iteration ? bracket_iterations : iteration;
  if (iteration == max_iterations) {
    result.success = false;
    result.primitive.fluid = {effective_floor, {0.0, 0.0, 0.0}, eos.pressure_floor};
    result.internal_energy = internal_energy_floor;
    result.primitive.magnetic[0] = magnetic[0];
    result.primitive.magnetic[1] = magnetic[1];
    result.primitive.magnetic[2] = magnetic[2];
    return result;
  }
  const parthenon::Real mu = root;
  const parthenon::Real x = 1.0 / (1.0 + mu * b2_physical);
  const parthenon::Real rbar_squared =
      x * (x * r_squared + mu * (1.0 + x) * rpar_squared);
  const parthenon::Real vhat_squared = fmin(mu * mu * rbar_squared, v0_squared);
  const parthenon::Real inverse_lorentz = sqrt(1.0 - vhat_squared);
  parthenon::Real lorentz = 1.0 / inverse_lorentz;
  const parthenon::Real mu_x = mu * x;
  const parthenon::Real qbar =
      q - 0.5 *
              (b2_physical +
               mu_x * mu_x * (b2_physical * r_squared - rpar_squared));
  parthenon::Real density = fmax(conserved.density * inverse_lorentz, 0.0);
  if (density < effective_floor) {
    density = effective_floor;
    result.density_floor = true;
  }
  parthenon::Real epsilon =
      fmax(lorentz * (qbar - mu * rbar_squared) +
               vhat_squared * lorentz * lorentz / (1.0 + lorentz),
           0.0);
  const parthenon::Real epsilon_floor = eos.MagnetisedSpecificInternalEnergyFloor(density);
  if (epsilon <= epsilon_floor) {
    epsilon = epsilon_floor;
    result.pressure_floor = true;
  }
  const parthenon::Real velocity_factor = fmax(lorentz * mu * x, 0.0);
  result.primitive.fluid.density = density;
  result.primitive.fluid.u[0] =
      velocity_factor * (conserved.momentum[0] / conserved.density +
                         mu * rpar * normalized_b1);
  result.primitive.fluid.u[1] =
      velocity_factor * (conserved.momentum[1] / conserved.density +
                         mu * rpar * normalized_b2);
  result.primitive.fluid.u[2] =
      velocity_factor * (conserved.momentum[2] / conserved.density +
                         mu * rpar * normalized_b3);
  result.primitive.magnetic[0] = magnetic[0];
  result.primitive.magnetic[1] = magnetic[1];
  result.primitive.magnetic[2] = magnetic[2];
  result.internal_energy = density * epsilon;
  result.primitive.fluid.pressure =
      eos.PressureFromInternalEnergyDensity(result.internal_energy);
  lorentz = sqrt(1.0 + result.primitive.fluid.u[0] * result.primitive.fluid.u[0] +
                 result.primitive.fluid.u[1] * result.primitive.fluid.u[1] +
                 result.primitive.fluid.u[2] * result.primitive.fluid.u[2]);
  if (lorentz > gamma_max) {
    const parthenon::Real scale = sqrt((gamma_max * gamma_max - 1.0) / (lorentz * lorentz - 1.0));
    result.primitive.fluid.u[0] *= scale;
    result.primitive.fluid.u[1] *= scale;
    result.primitive.fluid.u[2] *= scale;
    result.lorentz_ceiling = true;
  }
  return result;
}

KOKKOS_INLINE_FUNCTION MHDC2PResult SolveSRMHDC2P(HydroConservedState conserved,
                                                  const parthenon::Real magnetic[3],
                                                  const eos::RelativisticEOS& eos,
                                                  const parthenon::Real gamma_max,
                                                  const parthenon::Real sigma_max) {
  const parthenon::Real momentum2 = conserved.momentum[0] * conserved.momentum[0] +
                                    conserved.momentum[1] * conserved.momentum[1] +
                                    conserved.momentum[2] * conserved.momentum[2];
  const parthenon::Real magnetic2 =
      magnetic[0] * magnetic[0] + magnetic[1] * magnetic[1] + magnetic[2] * magnetic[2];
  const parthenon::Real rpar =
      (magnetic[0] * conserved.momentum[0] + magnetic[1] * conserved.momentum[1] +
       magnetic[2] * conserved.momentum[2]) /
      conserved.density;
  return SolveSRMHDC2PFromInvariants(conserved, magnetic, momentum2, magnetic2, rpar, eos,
                                     gamma_max, sigma_max);
}

KOKKOS_INLINE_FUNCTION MHDC2PResult
SolveGRMHDC2P(const HydroConservedState& conserved, const parthenon::Real magnetic[3],
              const eos::RelativisticEOS& eos, const parthenon::Real gamma_max,
              const parthenon::Real sigma_max, const geometry::MetricPoint& metric) {
  const parthenon::Real inverse_gdet = 1.0 / metric.gdet;
  HydroConservedState undensitized{};
  undensitized.density = conserved.density * inverse_gdet;
  undensitized.energy = conserved.energy * inverse_gdet;
  for (int axis = 0; axis < 3; ++axis)
    undensitized.momentum[axis] = conserved.momentum[axis] * inverse_gdet;

  HydroConservedState sr{};
  const parthenon::Real alpha = sqrt(-1.0 / metric.upper[0][0]);
  sr.density = alpha * undensitized.density;
  sr.energy = metric.upper[0][0] * (undensitized.energy - undensitized.density) +
              metric.upper[0][1] * undensitized.momentum[0] +
              metric.upper[0][2] * undensitized.momentum[1] +
              metric.upper[0][3] * undensitized.momentum[2];
  sr.energy *= -1.0 / metric.upper[0][0];
  sr.energy -= sr.density;

  const parthenon::Real m1l = undensitized.momentum[0] * alpha;
  const parthenon::Real m2l = undensitized.momentum[1] * alpha;
  const parthenon::Real m3l = undensitized.momentum[2] * alpha;
  sr.momentum[0] =
      (metric.upper[1][1] - metric.upper[0][1] * metric.upper[0][1] / metric.upper[0][0]) * m1l +
      (metric.upper[1][2] - metric.upper[0][1] * metric.upper[0][2] / metric.upper[0][0]) * m2l +
      (metric.upper[1][3] - metric.upper[0][1] * metric.upper[0][3] / metric.upper[0][0]) * m3l;
  sr.momentum[1] =
      (metric.upper[2][1] - metric.upper[0][2] * metric.upper[0][1] / metric.upper[0][0]) * m1l +
      (metric.upper[2][2] - metric.upper[0][2] * metric.upper[0][2] / metric.upper[0][0]) * m2l +
      (metric.upper[2][3] - metric.upper[0][2] * metric.upper[0][3] / metric.upper[0][0]) * m3l;
  sr.momentum[2] =
      (metric.upper[3][1] - metric.upper[0][3] * metric.upper[0][1] / metric.upper[0][0]) * m1l +
      (metric.upper[3][2] - metric.upper[0][3] * metric.upper[0][2] / metric.upper[0][0]) * m2l +
      (metric.upper[3][3] - metric.upper[0][3] * metric.upper[0][3] / metric.upper[0][0]) * m3l;
  const parthenon::Real momentum2 =
      m1l * sr.momentum[0] + m2l * sr.momentum[1] + m3l * sr.momentum[2];

  parthenon::Real sr_magnetic[3]{alpha * magnetic[0], alpha * magnetic[1], alpha * magnetic[2]};
  const parthenon::Real magnetic2 = metric.lower[1][1] * (sr_magnetic[0] * sr_magnetic[0]) +
                                    metric.lower[2][2] * (sr_magnetic[1] * sr_magnetic[1]) +
                                    metric.lower[3][3] * (sr_magnetic[2] * sr_magnetic[2]) +
                                    2.0 * (sr_magnetic[0] * (metric.lower[1][2] * sr_magnetic[1] +
                                                             metric.lower[1][3] * sr_magnetic[2]) +
                                           metric.lower[2][3] * sr_magnetic[1] * sr_magnetic[2]);
  const parthenon::Real rpar =
      (sr_magnetic[0] * m1l + sr_magnetic[1] * m2l + sr_magnetic[2] * m3l) / sr.density;

  // The SR invariant solve returns the spatial contravariant four-velocity. Its
  // Euclidean norm is the Lorentz factor only in Cartesian Minkowski space. In
  // fixed GR the ceiling must instead use the spatial metric, as in AthenaK's
  // IdealGRMHD::ConsToPrim.
  auto result = SolveSRMHDC2PFromInvariants(
      sr, sr_magnetic, momentum2, magnetic2, rpar, eos,
      std::numeric_limits<parthenon::Real>::max(), sigma_max);
  const parthenon::Real vx = result.primitive.fluid.u[0];
  const parthenon::Real vy = result.primitive.fluid.u[1];
  const parthenon::Real vz = result.primitive.fluid.u[2];
  const parthenon::Real spatial_u2 =
      metric.lower[1][1] * (vx * vx) + metric.lower[2][2] * (vy * vy) +
      metric.lower[3][3] * (vz * vz) + 2.0 * metric.lower[1][2] * vx * vy +
      2.0 * metric.lower[1][3] * vx * vz + 2.0 * metric.lower[2][3] * vy * vz;
  const parthenon::Real lorentz = sqrt(1.0 + spatial_u2);
  if (lorentz > gamma_max) {
    const parthenon::Real scale = sqrt((gamma_max * gamma_max - 1.0) / (lorentz * lorentz - 1.0));
    for (int axis = 0; axis < 3; ++axis)
      result.primitive.fluid.u[axis] *= scale;
    result.lorentz_ceiling = true;
  }
  for (int axis = 0; axis < 3; ++axis)
    result.primitive.magnetic[axis] = magnetic[axis];
  return result;
}

parthenon::TaskStatus
CalculateMHDFluxesBlockTask(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data);
parthenon::TaskStatus CalculateMHDFluxesMeshTask(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus CalculateMHDFluxesWithPassiveBlockTask(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data,
    const std::string& passive_conserved_name);
parthenon::TaskStatus
CalculateMHDFluxesWithPassiveMeshTask(parthenon::MeshData<parthenon::Real>* data,
                                      const std::string& passive_conserved_name);
parthenon::TaskStatus
ApplyMHDFluxCorrectionBlockTask(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data,
                                std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& base,
                                parthenon::Real gam0, parthenon::Real gam1,
                                parthenon::Real beta_dt);
parthenon::TaskStatus ApplyMHDFluxCorrectionWithPassiveBlockTask(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& base, parthenon::Real gam0,
    parthenon::Real gam1, parthenon::Real beta_dt, const std::string& passive_conserved_name);
parthenon::TaskStatus ApplyMHDFluxCorrectionNoCornerBlockTask(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& base, parthenon::Real gam0,
    parthenon::Real gam1, parthenon::Real beta_dt);
parthenon::TaskStatus ApplyMHDFluxCorrectionMeshTask(parthenon::MeshData<parthenon::Real>* data,
                                                     parthenon::MeshData<parthenon::Real>* base,
                                                     parthenon::Real gam0, parthenon::Real gam1,
                                                     parthenon::Real beta_dt);
parthenon::TaskStatus ApplyMHDFluxCorrectionWithPassiveMeshTask(
    parthenon::MeshData<parthenon::Real>* data, parthenon::MeshData<parthenon::Real>* base,
    parthenon::Real gam0, parthenon::Real gam1, parthenon::Real beta_dt,
    const std::string& passive_conserved_name);
void MHDConservedToPrimitiveBlock(parthenon::MeshBlockData<parthenon::Real>* data);
void MHDConservedToPrimitiveMesh(parthenon::MeshData<parthenon::Real>* data);
parthenon::Real EstimateMHDTimestepBlock(parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::Real EstimateMHDTimestepMesh(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus
ApplyMHDSourcesBlockTask(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data,
                         parthenon::Real dt);
parthenon::TaskStatus ApplyMHDSourcesMeshTask(parthenon::MeshData<parthenon::Real>* data,
                                              parthenon::Real dt);

} // namespace pangu::relativity

#endif
