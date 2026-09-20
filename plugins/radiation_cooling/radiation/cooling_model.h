#ifndef PANGU_RADIATION_COOLING_MODEL_H_
#define PANGU_RADIATION_COOLING_MODEL_H_

#include <parthenon/parthenon.hpp>

#include "eos/ideal_gas.h"
#include "relativity/relativistic_mhd.h"

namespace pangu::radiation {

struct FourVelocity {
  parthenon::Real upper[4]{};
  parthenon::Real lower[4]{};
};

struct CoolingIncrement {
  parthenon::Real momentum[3]{};
  parthenon::Real energy = 0.0;
};

struct CoolingState {
  parthenon::Real target_internal_energy = 0.0;
  parthenon::Real cooling_time = 0.0;
  parthenon::Real removed_internal_energy = 0.0;
  parthenon::Real removed_fraction = 0.0;
};

enum CoolingMask : int {
  cooled = 1 << 0,
  disabled = 1 << 1,
  before_start = 1 << 2,
  excised = 1 << 3,
  low_density = 1 << 4,
  high_magnetization = 1 << 5,
  unbound = 1 << 6,
  below_target = 1 << 7,
  c2p_limited = 1 << 8,
  base_state_invalid = 1 << 9
};

KOKKOS_INLINE_FUNCTION FourVelocity BuildFourVelocity(
    const relativity::HydroPrimitiveState& primitive, const geometry::MetricPoint& metric) {
  FourVelocity velocity{};
  const parthenon::Real vx = primitive.u[0];
  const parthenon::Real vy = primitive.u[1];
  const parthenon::Real vz = primitive.u[2];
  const parthenon::Real spatial_u2 =
      metric.lower[1][1] * vx * vx + 2.0 * metric.lower[1][2] * vx * vy +
      2.0 * metric.lower[1][3] * vx * vz + metric.lower[2][2] * vy * vy +
      2.0 * metric.lower[2][3] * vy * vz + metric.lower[3][3] * vz * vz;
  const parthenon::Real lorentz = sqrt(1.0 + spatial_u2);
  velocity.upper[0] = lorentz / metric.lapse;
  velocity.upper[1] = vx - metric.lapse * lorentz * metric.upper[0][1];
  velocity.upper[2] = vy - metric.lapse * lorentz * metric.upper[0][2];
  velocity.upper[3] = vz - metric.lapse * lorentz * metric.upper[0][3];
  for (int mu = 0; mu < 4; ++mu)
    for (int nu = 0; nu < 4; ++nu)
      velocity.lower[mu] += metric.lower[mu][nu] * velocity.upper[nu];
  return velocity;
}

// Integrate the covariant source S_nu = -Lambda u_nu over one coordinate-time
// stage. removed_internal_energy is the positive comoving internal-energy loss
// accumulated over Delta tau = Delta t/u^t. PANGU evolves sqrt(-g) T^t_nu,
// hence Delta U_nu = -sqrt(-g) Delta u u^t u_nu.
KOKKOS_INLINE_FUNCTION CoolingIncrement BuildCoolingIncrement(
    const relativity::HydroPrimitiveState& primitive, const geometry::MetricPoint& metric,
    const parthenon::Real removed_internal_energy) {
  CoolingIncrement increment{};
  if (!(removed_internal_energy > 0.0))
    return increment;
  const auto velocity = BuildFourVelocity(primitive, metric);
  const parthenon::Real normalization = -metric.gdet * removed_internal_energy * velocity.upper[0];
  increment.energy = normalization * velocity.lower[0];
  for (int axis = 0; axis < 3; ++axis)
    increment.momentum[axis] = normalization * velocity.lower[axis + 1];
  return increment;
}

KOKKOS_INLINE_FUNCTION relativity::HydroConservedState
AddCoolingIncrement(const relativity::HydroConservedState& state, const CoolingIncrement& increment,
                    const parthenon::Real fraction = 1.0) {
  auto result = state;
  result.energy += fraction * increment.energy;
  for (int axis = 0; axis < 3; ++axis)
    result.momentum[axis] += fraction * increment.momentum[axis];
  return result;
}

KOKKOS_INLINE_FUNCTION parthenon::Real CoolingRamp(const parthenon::Real time,
                                                   const parthenon::Real start_time,
                                                   const parthenon::Real ramp_time) {
  if (time < start_time)
    return 0.0;
  if (!(ramp_time > 0.0))
    return 1.0;
  const parthenon::Real x = fmin(fmax((time - start_time) / ramp_time, 0.0), 1.0);
  return x * x * (3.0 - 2.0 * x);
}

KOKKOS_INLINE_FUNCTION CoolingState
EvaluateTargetThicknessCooling(const parthenon::Real density, const parthenon::Real internal_energy,
                               const eos::IdealGas& eos, const parthenon::Real radius,
                               const parthenon::Real spin, const parthenon::Real h_target,
                               const parthenon::Real beta_cool, const parthenon::Real coordinate_dt,
                               const parthenon::Real time_component, const parthenon::Real ramp) {
  CoolingState state{};
  if (!(density > 0.0) || !(internal_energy > 0.0) || !(eos.gamma > 1.0) || !(radius > 0.0) ||
      !(beta_cool > 0.0) || !(time_component > 0.0) || !(coordinate_dt > 0.0) || !(ramp > 0.0))
    return state;
  const parthenon::Real radius_to_three_halves = radius * sqrt(radius);
  const parthenon::Real denominator = radius_to_three_halves + spin;
  if (!(denominator > 0.0))
    return state;
  const parthenon::Real omega = 1.0 / denominator;
  const parthenon::Real orbital_speed = radius * omega;
  state.target_internal_energy = eos.InternalEnergyDensity(
      density * h_target * h_target * orbital_speed * orbital_speed);
  state.cooling_time = beta_cool / omega;
  const parthenon::Real excess = internal_energy - state.target_internal_energy;
  if (!(excess > 0.0))
    return state;
  const parthenon::Real proper_dt = coordinate_dt / time_component;
  const parthenon::Real exponent = -ramp * proper_dt / state.cooling_time;
  state.removed_fraction = fmin(fmax(1.0 - exp(exponent), 0.0), 1.0);
  state.removed_internal_energy = excess * state.removed_fraction;
  return state;
}

KOKKOS_INLINE_FUNCTION bool IsBoundFluid(const relativity::HydroPrimitiveState& primitive,
                                         const parthenon::Real internal_energy,
                                         const eos::IdealGas& eos,
                                         const geometry::MetricPoint& metric) {
  if (!(primitive.density > 0.0))
    return false;
  const auto velocity = BuildFourVelocity(primitive, metric);
  const parthenon::Real specific_enthalpy =
      eos.SpecificEnthalpyFromSpecificInternalEnergy(internal_energy / primitive.density);
  return -specific_enthalpy * velocity.lower[0] < 1.0;
}

} // namespace pangu::radiation

#endif
