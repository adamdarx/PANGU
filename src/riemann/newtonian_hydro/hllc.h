#ifndef PANGU_RIEMANN_NEWTONIAN_HYDRO_HLLC_H_
#define PANGU_RIEMANN_NEWTONIAN_HYDRO_HLLC_H_

#include <cmath>

#include "eos/newtonian_eos.h"

namespace pangu::riemann::newtonian_hydro {

using hydro::Primitive;
using parthenon::Real;

// HLLC flux restoring the contact wave; it requires an energy equation.
struct HLLC {
  KOKKOS_INLINE_FUNCTION static void Solve(const Primitive& left, const Primitive& right,
                                           const eos::NewtonianEOS& eos, const int direction,
                                           Real* interface_flux) {
    const Real left_sound = eos.SoundSpeed(left);
    const Real right_sound = eos.SoundSpeed(right);
    const Real left_normal = left.velocity[direction];
    const Real right_normal = right.velocity[direction];
    const int tangent1 = (direction + 1) % 3;
    const int tangent2 = (direction + 2) % 3;
    const Real gm1 = eos.gamma - 1.0;
    const Real alpha = (eos.gamma + 1.0) / (2.0 * eos.gamma);
    const Real left_energy = left.pressure / gm1 + 0.5 * left.density *
                                                       (left.velocity[0] * left.velocity[0] +
                                                        left.velocity[1] * left.velocity[1] +
                                                        left.velocity[2] * left.velocity[2]);
    const Real right_energy = right.pressure / gm1 + 0.5 * right.density *
                                                         (right.velocity[0] * right.velocity[0] +
                                                          right.velocity[1] * right.velocity[1] +
                                                          right.velocity[2] * right.velocity[2]);

    const Real impedance = 0.25 * (left.density + right.density) * (left_sound + right_sound);
    const Real middle_pressure =
        0.5 * (left.pressure + right.pressure + (left_normal - right_normal) * impedance);
    const Real left_factor = middle_pressure <= left.pressure
                                 ? 1.0
                                 : sqrt(1.0 + alpha * (middle_pressure / left.pressure - 1.0));
    const Real right_factor = middle_pressure <= right.pressure
                                  ? 1.0
                                  : sqrt(1.0 + alpha * (middle_pressure / right.pressure - 1.0));
    const Real outer_left = left_normal - left_sound * left_factor;
    const Real outer_right = right_normal + right_sound * right_factor;
    const Real positive_speed = outer_right > 0.0 ? outer_right : 1.0e-20;
    const Real negative_speed = outer_left < 0.0 ? outer_left : -1.0e-20;

    const Real left_relative = left_normal - outer_left;
    const Real right_relative = right_normal - outer_right;
    const Real left_pressure_momentum = left.pressure + left_relative * left.density * left_normal;
    const Real right_pressure_momentum =
        right.pressure + right_relative * right.density * right_normal;
    const Real left_mass = left.density * left_relative;
    const Real right_mass = -right.density * right_relative;
    const Real contact =
        (left_pressure_momentum - right_pressure_momentum) / (left_mass + right_mass);
    Real contact_pressure =
        (left_mass * right_pressure_momentum + right_mass * left_pressure_momentum) /
        (left_mass + right_mass);
    contact_pressure = contact_pressure > 0.0 ? contact_pressure : 0.0;

    const Real left_shifted_mass = left.density * (left_normal - negative_speed);
    const Real right_shifted_mass = right.density * (right_normal - positive_speed);
    Real left_shifted[hydro::kIdealComponents]{};
    Real right_shifted[hydro::kIdealComponents]{};
    left_shifted[hydro::IDN] = left_shifted_mass;
    right_shifted[hydro::IDN] = right_shifted_mass;
    left_shifted[hydro::MomentumIndex(direction)] = left_shifted_mass * left_normal + left.pressure;
    right_shifted[hydro::MomentumIndex(direction)] =
        right_shifted_mass * right_normal + right.pressure;
    left_shifted[hydro::IM1 + tangent1] = left_shifted_mass * left.velocity[tangent1];
    right_shifted[hydro::IM1 + tangent1] = right_shifted_mass * right.velocity[tangent1];
    left_shifted[hydro::IM1 + tangent2] = left_shifted_mass * left.velocity[tangent2];
    right_shifted[hydro::IM1 + tangent2] = right_shifted_mass * right.velocity[tangent2];
    left_shifted[hydro::IEN] =
        left_energy * (left_normal - negative_speed) + left.pressure * left_normal;
    right_shifted[hydro::IEN] =
        right_energy * (right_normal - positive_speed) + right.pressure * right_normal;

    Real left_weight;
    Real right_weight;
    Real pressure_weight;
    if (contact >= 0.0) {
      left_weight = contact / (contact - negative_speed);
      right_weight = 0.0;
      pressure_weight = -negative_speed / (contact - negative_speed);
    } else {
      left_weight = 0.0;
      right_weight = -contact / (positive_speed - contact);
      pressure_weight = positive_speed / (positive_speed - contact);
    }
    for (int n = 0; n < hydro::kIdealComponents; ++n) {
      interface_flux[n] = left_weight * left_shifted[n] + right_weight * right_shifted[n];
    }
    interface_flux[hydro::MomentumIndex(direction)] += pressure_weight * contact_pressure;
    interface_flux[hydro::IEN] += pressure_weight * contact_pressure * contact;
  }
};

} // namespace pangu::riemann::newtonian_hydro

#endif
