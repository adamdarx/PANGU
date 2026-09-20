#ifndef PANGU_RIEMANN_NEWTONIAN_MHD_HLLE_H_
#define PANGU_RIEMANN_NEWTONIAN_MHD_HLLE_H_

// Copyright (c) 2020 James M. Stone and the AthenaK team.
// HLLE algebra is derived from AthenaK under BSD-3-Clause; see LICENSE/NOTICE.

#include <cmath>

#include "eos/newtonian_mhd_eos.h"
#include "riemann/newtonian_mhd/electric_field.h"

namespace pangu::riemann::newtonian_mhd {

using mhd::InterfaceFlux;
using mhd::Primitive;
using parthenon::Real;

// HLLE flux bounded by Roe-averaged and one-sided fast magnetosonic speeds.
struct HLLE {
  KOKKOS_INLINE_FUNCTION static void Solve(const Primitive& left, const Primitive& right,
                                           const eos::NewtonianMHDEOS& eos, const int direction,
                                           InterfaceFlux& result) {
    const int tangent1 = (direction + 1) % 3;
    const int tangent2 = (direction + 2) % 3;
    const Real sqrt_left = sqrt(left.density);
    const Real sqrt_right = sqrt(right.density);
    const Real inverse_sum = 1.0 / (sqrt_left + sqrt_right);
    Primitive roe{};
    roe.density = sqrt_left * sqrt_right;
    for (int axis = 0; axis < 3; ++axis) {
      roe.velocity[axis] =
          (sqrt_left * left.velocity[axis] + sqrt_right * right.velocity[axis]) * inverse_sum;
    }
    roe.magnetic[direction] = left.magnetic[direction];
    roe.magnetic[tangent1] =
        (sqrt_right * left.magnetic[tangent1] + sqrt_left * right.magnetic[tangent1]) * inverse_sum;
    roe.magnetic[tangent2] =
        (sqrt_right * left.magnetic[tangent2] + sqrt_left * right.magnetic[tangent2]) * inverse_sum;

    const Real magnetic_jump = 0.5 *
                               ((left.magnetic[tangent1] - right.magnetic[tangent1]) *
                                    (left.magnetic[tangent1] - right.magnetic[tangent1]) +
                                (left.magnetic[tangent2] - right.magnetic[tangent2]) *
                                    (left.magnetic[tangent2] - right.magnetic[tangent2])) /
                               ((sqrt_left + sqrt_right) * (sqrt_left + sqrt_right));
    const Real density_factor = 0.5 * (left.density + right.density) / roe.density;
    Real roe_fast;
    if (eos.HasEnergy()) {
      const auto ul = eos.PrimitiveToConservedAndFlux(left, direction);
      const auto ur = eos.PrimitiveToConservedAndFlux(right, direction);
      const Real left_b2 = left.magnetic[0] * left.magnetic[0] +
                           left.magnetic[1] * left.magnetic[1] +
                           left.magnetic[2] * left.magnetic[2];
      const Real right_b2 = right.magnetic[0] * right.magnetic[0] +
                            right.magnetic[1] * right.magnetic[1] +
                            right.magnetic[2] * right.magnetic[2];
      const Real enthalpy =
          ((ul.conserved[mhd::IEN] + left.pressure + 0.5 * left_b2) / sqrt_left +
           (ur.conserved[mhd::IEN] + right.pressure + 0.5 * right_b2) / sqrt_right) *
          inverse_sum;
      const Real velocity2 = roe.velocity[0] * roe.velocity[0] + roe.velocity[1] * roe.velocity[1] +
                             roe.velocity[2] * roe.velocity[2];
      const Real transverse2 = roe.magnetic[tangent1] * roe.magnetic[tangent1] +
                               roe.magnetic[tangent2] * roe.magnetic[tangent2];
      const Real normal_alfven2 = roe.magnetic[direction] * roe.magnetic[direction] / roe.density;
      const Real transverse_star2 =
          ((eos.gamma - 1.0) - (eos.gamma - 2.0) * density_factor) * transverse2;
      const Real hp = enthalpy - (normal_alfven2 + transverse2 / roe.density);
      const Real sound_star2 =
          fmax((eos.gamma - 1.0) * (hp - 0.5 * velocity2) - (eos.gamma - 2.0) * magnetic_jump, 0.0);
      const Real transverse_alfven2 = transverse_star2 / roe.density;
      const Real sum = normal_alfven2 + transverse_alfven2 + sound_star2;
      const Real difference = normal_alfven2 + transverse_alfven2 - sound_star2;
      roe_fast = sqrt(
          0.5 * (sum + sqrt(difference * difference + 4.0 * sound_star2 * transverse_alfven2)));
    } else {
      const Real transverse2 = roe.magnetic[tangent1] * roe.magnetic[tangent1] +
                               roe.magnetic[tangent2] * roe.magnetic[tangent2];
      const Real normal_alfven2 = roe.magnetic[direction] * roe.magnetic[direction] / roe.density;
      const Real transverse_alfven2 = transverse2 * density_factor / roe.density;
      const Real sound_star2 = eos.iso_sound_speed * eos.iso_sound_speed + magnetic_jump;
      const Real sum = normal_alfven2 + transverse_alfven2 + sound_star2;
      const Real difference = normal_alfven2 + transverse_alfven2 - sound_star2;
      roe_fast = sqrt(
          0.5 * (sum + sqrt(difference * difference + 4.0 * sound_star2 * transverse_alfven2)));
    }

    const Real left_speed = fmin(roe.velocity[direction] - roe_fast,
                                 left.velocity[direction] - eos.FastSpeed(left, direction));
    const Real right_speed = fmax(roe.velocity[direction] + roe_fast,
                                  right.velocity[direction] + eos.FastSpeed(right, direction));
    const Real positive = right_speed > 0.0 ? right_speed : 1.0e-20;
    const Real negative = left_speed < 0.0 ? left_speed : -1.0e-20;
    const Real weight =
        positive != negative ? 0.5 * (positive + negative) / (positive - negative) : 0.0;
    const auto ul = eos.PrimitiveToConservedAndFlux(left, direction);
    const auto ur = eos.PrimitiveToConservedAndFlux(right, direction);
    const int components = eos.HasEnergy() ? mhd::kIdealComponents : mhd::kIsothermalComponents;
    for (int n = 0; n < components; ++n) {
      const Real shifted_left = ul.flux[n] - negative * ul.conserved[n];
      const Real shifted_right = ur.flux[n] - positive * ur.conserved[n];
      result.fluid[n] =
          0.5 * (shifted_left + shifted_right) + weight * (shifted_left - shifted_right);
    }

    const Real left_by_flux = left.magnetic[tangent1] * left.velocity[direction] -
                              left.magnetic[direction] * left.velocity[tangent1];
    const Real right_by_flux = right.magnetic[tangent1] * right.velocity[direction] -
                               right.magnetic[direction] * right.velocity[tangent1];
    const Real left_bz_flux = left.magnetic[tangent2] * left.velocity[direction] -
                              left.magnetic[direction] * left.velocity[tangent2];
    const Real right_bz_flux = right.magnetic[tangent2] * right.velocity[direction] -
                               right.magnetic[direction] * right.velocity[tangent2];
    const Real left_by_shifted = left_by_flux - negative * left.magnetic[tangent1];
    const Real right_by_shifted = right_by_flux - positive * right.magnetic[tangent1];
    const Real left_bz_shifted = left_bz_flux - negative * left.magnetic[tangent2];
    const Real right_bz_shifted = right_bz_flux - positive * right.magnetic[tangent2];
    const Real by_flux =
        0.5 * (left_by_shifted + right_by_shifted) + weight * (left_by_shifted - right_by_shifted);
    const Real bz_flux =
        0.5 * (left_bz_shifted + right_bz_shifted) + weight * (left_bz_shifted - right_bz_shifted);
    StoreElectricField(by_flux, bz_flux, result);
  }
};

} // namespace pangu::riemann::newtonian_mhd

#endif
