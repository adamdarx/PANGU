#ifndef PANGU_RIEMANN_NEWTONIAN_HYDRO_ROE_H_
#define PANGU_RIEMANN_NEWTONIAN_HYDRO_ROE_H_

#include <cfloat>
#include <cmath>

#include "eos/newtonian_eos.h"

namespace pangu::riemann::newtonian_hydro {

using hydro::Primitive;
using parthenon::Real;

// Roe linearized flux; negative intermediate densities fall back to LLF dissipation.
struct Roe {
  KOKKOS_INLINE_FUNCTION static void Solve(const Primitive& left, const Primitive& right,
                                           const eos::NewtonianEOS& eos, const int direction,
                                           Real* interface_flux) {
    const int components = eos.HasEnergy() ? hydro::kIdealComponents : hydro::kIsothermalComponents;
    const Real left_sound = eos.SoundSpeed(left);
    const Real right_sound = eos.SoundSpeed(right);
    const Real left_normal = left.velocity[direction];
    const Real right_normal = right.velocity[direction];
    constexpr int D = 0;
    constexpr int N = 1;
    constexpr int T1 = 2;
    constexpr int T2 = 3;
    constexpr int E = 4;
    const int tangent1 = (direction + 1) % 3;
    const int tangent2 = (direction + 2) % 3;
    const Real gm1 = eos.gamma - 1.0;
    const Real sqrt_left = sqrt(left.density);
    const Real sqrt_right = sqrt(right.density);
    const Real inverse_sum = 1.0 / (sqrt_left + sqrt_right);
    Real primitive_left[5] = {left.density, left_normal, left.velocity[tangent1],
                              left.velocity[tangent2], left.pressure};
    Real primitive_right[5] = {right.density, right_normal, right.velocity[tangent1],
                               right.velocity[tangent2], right.pressure};
    Real roe_state[5]{};
    roe_state[D] = sqrt_left * sqrt_right;
    roe_state[N] = (sqrt_left * primitive_left[N] + sqrt_right * primitive_right[N]) * inverse_sum;
    roe_state[T1] =
        (sqrt_left * primitive_left[T1] + sqrt_right * primitive_right[T1]) * inverse_sum;
    roe_state[T2] =
        (sqrt_left * primitive_left[T2] + sqrt_right * primitive_right[T2]) * inverse_sum;

    Real left_energy = 0.0;
    Real right_energy = 0.0;
    if (eos.HasEnergy()) {
      left_energy = primitive_left[E] / gm1 + 0.5 * primitive_left[D] *
                                                  (primitive_left[N] * primitive_left[N] +
                                                   primitive_left[T1] * primitive_left[T1] +
                                                   primitive_left[T2] * primitive_left[T2]);
      right_energy = primitive_right[E] / gm1 + 0.5 * primitive_right[D] *
                                                    (primitive_right[N] * primitive_right[N] +
                                                     primitive_right[T1] * primitive_right[T1] +
                                                     primitive_right[T2] * primitive_right[T2]);
      roe_state[E] = ((left_energy + primitive_left[E]) / sqrt_left +
                      (right_energy + primitive_right[E]) / sqrt_right) *
                     inverse_sum;
    }

    Real left_flux[5]{};
    Real right_flux[5]{};
    const Real left_normal_momentum = primitive_left[D] * primitive_left[N];
    const Real right_normal_momentum = primitive_right[D] * primitive_right[N];
    left_flux[D] = left_normal_momentum;
    right_flux[D] = right_normal_momentum;
    left_flux[N] = left_normal_momentum * primitive_left[N];
    right_flux[N] = right_normal_momentum * primitive_right[N];
    left_flux[T1] = left_normal_momentum * primitive_left[T1];
    right_flux[T1] = right_normal_momentum * primitive_right[T1];
    left_flux[T2] = left_normal_momentum * primitive_left[T2];
    right_flux[T2] = right_normal_momentum * primitive_right[T2];
    if (eos.HasEnergy()) {
      left_flux[N] += primitive_left[E];
      right_flux[N] += primitive_right[E];
      left_flux[E] = (left_energy + primitive_left[E]) * primitive_left[N];
      right_flux[E] = (right_energy + primitive_right[E]) * primitive_right[N];
    } else {
      left_flux[N] += eos.iso_sound_speed * eos.iso_sound_speed * primitive_left[D];
      right_flux[N] += eos.iso_sound_speed * eos.iso_sound_speed * primitive_right[D];
    }

    Real delta[5]{};
    delta[D] = primitive_right[D] - primitive_left[D];
    delta[N] = primitive_right[D] * primitive_right[N] - primitive_left[D] * primitive_left[N];
    delta[T1] = primitive_right[D] * primitive_right[T1] - primitive_left[D] * primitive_left[T1];
    delta[T2] = primitive_right[D] * primitive_right[T2] - primitive_left[D] * primitive_left[T2];
    if (eos.HasEnergy())
      delta[E] = right_energy - left_energy;

    Real local_flux[5]{};
    for (int n = 0; n < components; ++n)
      local_flux[n] = 0.5 * (left_flux[n] + right_flux[n]);

    Real eigenvalues[5]{};
    int llf_fallback = 0;
    if (eos.HasEnergy()) {
      const Real normal = roe_state[N];
      const Real tangent_velocity1 = roe_state[T1];
      const Real tangent_velocity2 = roe_state[T2];
      const Real enthalpy = roe_state[E];
      const Real velocity_squared = normal * normal + tangent_velocity1 * tangent_velocity1 +
                                    tangent_velocity2 * tangent_velocity2;
      const Real thermal_enthalpy = enthalpy - 0.5 * velocity_squared;
      const Real sound_squared =
          thermal_enthalpy < 0.0 ? static_cast<Real>(FLT_MIN) : gm1 * thermal_enthalpy;
      const Real sound = sqrt(sound_squared);
      eigenvalues[0] = normal - sound;
      eigenvalues[1] = normal;
      eigenvalues[2] = normal;
      eigenvalues[3] = normal;
      eigenvalues[4] = normal + sound;

      Real amplitude[5];
      const Real normalization = 0.5 / sound_squared;
      amplitude[0] = delta[D] * (0.5 * gm1 * velocity_squared + normal * sound);
      amplitude[0] -= delta[N] * (gm1 * normal + sound);
      amplitude[0] -= delta[T1] * gm1 * tangent_velocity1;
      amplitude[0] -= delta[T2] * gm1 * tangent_velocity2;
      amplitude[0] += delta[E] * gm1;
      amplitude[0] *= normalization;

      amplitude[1] = -delta[D] * tangent_velocity1 + delta[T1];
      amplitude[2] = -delta[D] * tangent_velocity2 + delta[T2];

      const Real gm1_over_sound_squared = gm1 / sound_squared;
      amplitude[3] = delta[D] * (1.0 - normalization * gm1 * velocity_squared);
      amplitude[3] += delta[N] * gm1_over_sound_squared * normal;
      amplitude[3] += delta[T1] * gm1_over_sound_squared * tangent_velocity1;
      amplitude[3] += delta[T2] * gm1_over_sound_squared * tangent_velocity2;
      amplitude[3] -= delta[E] * gm1_over_sound_squared;

      amplitude[4] = delta[D] * (0.5 * gm1 * velocity_squared - normal * sound);
      amplitude[4] -= delta[N] * (gm1 * normal - sound);
      amplitude[4] -= delta[T1] * gm1 * tangent_velocity1;
      amplitude[4] -= delta[T2] * gm1 * tangent_velocity2;
      amplitude[4] += delta[E] * gm1;
      amplitude[4] *= normalization;

      Real coefficient[5];
      for (int n = 0; n < 5; ++n)
        coefficient[n] = -0.5 * fabs(eigenvalues[n]) * amplitude[n];

      Real intermediate_density = primitive_left[D] + amplitude[0];
      if (intermediate_density < 0.0)
        llf_fallback = 1;
      intermediate_density += amplitude[3];
      if (intermediate_density < 0.0)
        llf_fallback = 1;

      local_flux[D] += coefficient[0] + coefficient[3] + coefficient[4];
      local_flux[N] += coefficient[0] * (normal - sound) + coefficient[3] * normal +
                       coefficient[4] * (normal + sound);
      local_flux[T1] += coefficient[0] * tangent_velocity1 + coefficient[1] +
                        coefficient[3] * tangent_velocity1 + coefficient[4] * tangent_velocity1;
      local_flux[T2] += coefficient[0] * tangent_velocity2 + coefficient[2] +
                        coefficient[3] * tangent_velocity2 + coefficient[4] * tangent_velocity2;
      local_flux[E] += coefficient[0] * (enthalpy - normal * sound) +
                       coefficient[1] * tangent_velocity1 + coefficient[2] * tangent_velocity2 +
                       coefficient[3] * 0.5 * velocity_squared +
                       coefficient[4] * (enthalpy + normal * sound);
    } else {
      const Real normal = roe_state[N];
      const Real tangent_velocity1 = roe_state[T1];
      const Real tangent_velocity2 = roe_state[T2];
      const Real sound = eos.iso_sound_speed;
      eigenvalues[0] = normal - sound;
      eigenvalues[1] = normal;
      eigenvalues[2] = normal;
      eigenvalues[3] = normal + sound;

      Real amplitude[4];
      amplitude[0] = delta[D] * (0.5 + 0.5 * normal / sound) - delta[N] * 0.5 / sound;
      amplitude[1] = -delta[D] * tangent_velocity1 + delta[T1];
      amplitude[2] = -delta[D] * tangent_velocity2 + delta[T2];
      amplitude[3] = delta[D] * (0.5 - 0.5 * normal / sound) + delta[N] * 0.5 / sound;

      Real coefficient[4];
      for (int n = 0; n < 4; ++n)
        coefficient[n] = -0.5 * fabs(eigenvalues[n]) * amplitude[n];

      Real intermediate_density = primitive_left[D] + amplitude[0];
      if (intermediate_density < 0.0)
        llf_fallback = 1;
      intermediate_density += amplitude[3];
      if (intermediate_density < 0.0)
        llf_fallback = 1;

      local_flux[D] += coefficient[0] + coefficient[3];
      local_flux[N] += coefficient[0] * (normal - sound) + coefficient[3] * (normal + sound);
      local_flux[T1] +=
          coefficient[0] * tangent_velocity1 + coefficient[1] + coefficient[3] * tangent_velocity1;
      local_flux[T2] +=
          coefficient[0] * tangent_velocity2 + coefficient[2] + coefficient[3] * tangent_velocity2;
    }

    if (eigenvalues[0] >= 0.0) {
      for (int n = 0; n < components; ++n)
        local_flux[n] = left_flux[n];
    }
    const int last_wave = eos.HasEnergy() ? 4 : 3;
    if (eigenvalues[last_wave] <= 0.0) {
      for (int n = 0; n < components; ++n)
        local_flux[n] = right_flux[n];
    }

    if (llf_fallback != 0) {
      const Real dissipation =
          0.5 * fmax(fabs(primitive_left[N]) + left_sound, fabs(primitive_right[N]) + right_sound);
      for (int n = 0; n < components; ++n) {
        local_flux[n] = 0.5 * (left_flux[n] + right_flux[n]) - dissipation * delta[n];
      }
    }

    interface_flux[hydro::IDN] = local_flux[D];
    interface_flux[hydro::MomentumIndex(direction)] = local_flux[N];
    interface_flux[hydro::IM1 + tangent1] = local_flux[T1];
    interface_flux[hydro::IM1 + tangent2] = local_flux[T2];
    if (eos.HasEnergy())
      interface_flux[hydro::IEN] = local_flux[E];
  }
};

} // namespace pangu::riemann::newtonian_hydro

#endif
