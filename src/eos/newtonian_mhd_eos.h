#ifndef PANGU_EOS_NEWTONIAN_MHD_EOS_H_
#define PANGU_EOS_NEWTONIAN_MHD_EOS_H_

#include <cmath>

#include "eos/ideal_gas.h"
#include "mhd/mhd_types.h"

namespace pangu::eos {

struct NewtonianMHDEOS {
  mhd::EosMode mode;
  parthenon::Real gamma;
  parthenon::Real iso_sound_speed;
  parthenon::Real density_floor;
  parthenon::Real pressure_floor;

  KOKKOS_INLINE_FUNCTION
  bool HasEnergy() const { return mode == mhd::EosMode::ideal; }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real FastSpeed(const mhd::Primitive& state, const int direction) const {
    const int tangent1 = (direction + 1) % 3;
    const int tangent2 = (direction + 2) % 3;
    const parthenon::Real acoustic =
        HasEnergy() ? gamma * state.pressure : iso_sound_speed * iso_sound_speed * state.density;
    const parthenon::Real normal2 = state.magnetic[direction] * state.magnetic[direction];
    const parthenon::Real transverse2 = state.magnetic[tangent1] * state.magnetic[tangent1] +
                                        state.magnetic[tangent2] * state.magnetic[tangent2];
    const parthenon::Real sum = normal2 + transverse2 + acoustic;
    const parthenon::Real difference = normal2 + transverse2 - acoustic;
    return sqrt(0.5 * (sum + sqrt(difference * difference + 4.0 * acoustic * transverse2)) /
                state.density);
  }

  KOKKOS_INLINE_FUNCTION
  mhd::ConservedFlux PrimitiveToConservedAndFlux(const mhd::Primitive& state,
                                                 const int direction) const {
    mhd::ConservedFlux result{};
    const int tangent1 = (direction + 1) % 3;
    const int tangent2 = (direction + 2) % 3;
    const parthenon::Real velocity2 = state.velocity[0] * state.velocity[0] +
                                      state.velocity[1] * state.velocity[1] +
                                      state.velocity[2] * state.velocity[2];
    const parthenon::Real magnetic2 = state.magnetic[0] * state.magnetic[0] +
                                      state.magnetic[1] * state.magnetic[1] +
                                      state.magnetic[2] * state.magnetic[2];
    const parthenon::Real normal_velocity = state.velocity[direction];
    const parthenon::Real normal_magnetic = state.magnetic[direction];
    const parthenon::Real pressure =
        HasEnergy() ? state.pressure : iso_sound_speed * iso_sound_speed * state.density;
    const parthenon::Real total_pressure = pressure + 0.5 * magnetic2;
    const parthenon::Real velocity_dot_magnetic = state.velocity[0] * state.magnetic[0] +
                                                  state.velocity[1] * state.magnetic[1] +
                                                  state.velocity[2] * state.magnetic[2];

    result.conserved[mhd::IDN] = state.density;
    for (int axis = 0; axis < 3; ++axis)
      result.conserved[mhd::IM1 + axis] = state.density * state.velocity[axis];
    result.flux[mhd::IDN] = state.density * normal_velocity;
    result.flux[mhd::MomentumIndex(direction)] = state.density * normal_velocity * normal_velocity +
                                                 total_pressure - normal_magnetic * normal_magnetic;
    result.flux[mhd::IM1 + tangent1] = state.density * normal_velocity * state.velocity[tangent1] -
                                       normal_magnetic * state.magnetic[tangent1];
    result.flux[mhd::IM1 + tangent2] = state.density * normal_velocity * state.velocity[tangent2] -
                                       normal_magnetic * state.magnetic[tangent2];
    if (HasEnergy()) {
      const parthenon::Real energy =
          IdealGas{gamma}.InternalEnergyDensity(pressure) + 0.5 * state.density * velocity2 +
          0.5 * magnetic2;
      result.conserved[mhd::IEN] = energy;
      result.flux[mhd::IEN] =
          normal_velocity * (energy + total_pressure) - normal_magnetic * velocity_dot_magnetic;
    }
    return result;
  }
};

} // namespace pangu::eos

#endif
