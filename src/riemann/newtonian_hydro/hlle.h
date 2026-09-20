#ifndef PANGU_RIEMANN_NEWTONIAN_HYDRO_HLLE_H_
#define PANGU_RIEMANN_NEWTONIAN_HYDRO_HLLE_H_

#include <cmath>

#include "eos/newtonian_eos.h"

namespace pangu::riemann::newtonian_hydro {

using hydro::Primitive;
using parthenon::Real;

// HLLE flux bounded by Roe-averaged and one-sided acoustic speeds.
struct HLLE {
  KOKKOS_INLINE_FUNCTION static void Solve(const Primitive& left, const Primitive& right,
                                           const eos::NewtonianEOS& eos, const int direction,
                                           Real* interface_flux) {
    const auto left_state = eos.PrimitiveToConservedAndFlux(left, direction);
    const auto right_state = eos.PrimitiveToConservedAndFlux(right, direction);
    const int components = eos.HasEnergy() ? hydro::kIdealComponents : hydro::kIsothermalComponents;
    const Real left_sound = eos.SoundSpeed(left);
    const Real right_sound = eos.SoundSpeed(right);
    const Real left_normal = left.velocity[direction];
    const Real right_normal = right.velocity[direction];
    const Real sqrt_left = sqrt(left.density);
    const Real sqrt_right = sqrt(right.density);
    const Real inverse_sum = 1.0 / (sqrt_left + sqrt_right);
    Real roe_velocity[3];
    for (int axis = 0; axis < 3; ++axis) {
      roe_velocity[axis] =
          (sqrt_left * left.velocity[axis] + sqrt_right * right.velocity[axis]) * inverse_sum;
    }
    Real roe_sound = eos.iso_sound_speed;
    if (eos.HasEnergy()) {
      const Real left_enthalpy = (left_state.conserved[hydro::IEN] + left.pressure) / sqrt_left;
      const Real right_enthalpy = (right_state.conserved[hydro::IEN] + right.pressure) / sqrt_right;
      const Real roe_enthalpy = (left_enthalpy + right_enthalpy) * inverse_sum;
      const Real velocity_squared = roe_velocity[0] * roe_velocity[0] +
                                    roe_velocity[1] * roe_velocity[1] +
                                    roe_velocity[2] * roe_velocity[2];
      const Real thermal_enthalpy = roe_enthalpy - 0.5 * velocity_squared;
      roe_sound = thermal_enthalpy < 0.0 ? 0.0 : sqrt((eos.gamma - 1.0) * thermal_enthalpy);
    }
    const Real outer_left = fmin(roe_velocity[direction] - roe_sound, left_normal - left_sound);
    const Real outer_right = fmax(roe_velocity[direction] + roe_sound, right_normal + right_sound);
    const Real positive_speed = outer_right > 0.0 ? outer_right : 1.0e-20;
    const Real negative_speed = outer_left < 0.0 ? outer_left : -1.0e-20;
    const Real weight = positive_speed != negative_speed ? 0.5 * (positive_speed + negative_speed) /
                                                               (positive_speed - negative_speed)
                                                         : 0.0;
    for (int n = 0; n < components; ++n) {
      const Real left_shifted = left_state.flux[n] - negative_speed * left_state.conserved[n];
      const Real right_shifted = right_state.flux[n] - positive_speed * right_state.conserved[n];
      interface_flux[n] =
          0.5 * (left_shifted + right_shifted) + weight * (left_shifted - right_shifted);
    }
  }
};

} // namespace pangu::riemann::newtonian_hydro

#endif
