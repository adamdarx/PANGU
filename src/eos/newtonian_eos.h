#ifndef PANGU_EOS_NEWTONIAN_EOS_H_
#define PANGU_EOS_NEWTONIAN_EOS_H_

#include <cmath>

#include "eos/ideal_gas.h"
#include "hydro/hydro_types.h"

namespace pangu::eos {

using hydro::EosMode;
using hydro::FluxState;
using hydro::Primitive;
using parthenon::Real;

struct NewtonianEOS {
  EosMode mode;
  Real gamma;
  Real iso_sound_speed;
  Real density_floor;
  Real pressure_floor;

  KOKKOS_INLINE_FUNCTION
  bool HasEnergy() const { return mode == EosMode::ideal; }

  KOKKOS_INLINE_FUNCTION
  Real SoundSpeed(const Primitive& primitive) const {
    if (HasEnergy()) {
      return sqrt(IdealGas{gamma}.NewtonianSoundSpeedSquared(primitive.density,
                                                              primitive.pressure));
    }
    return iso_sound_speed;
  }

  KOKKOS_INLINE_FUNCTION
  FluxState PrimitiveToConservedAndFlux(const Primitive& primitive, const int direction) const {
    FluxState result{};
    const Real density = primitive.density;
    const Real normal_velocity = primitive.velocity[direction];
    const Real pressure =
        HasEnergy() ? primitive.pressure : iso_sound_speed * iso_sound_speed * density;
    const Real velocity_squared = primitive.velocity[0] * primitive.velocity[0] +
                                  primitive.velocity[1] * primitive.velocity[1] +
                                  primitive.velocity[2] * primitive.velocity[2];
    result.conserved[hydro::IDN] = density;
    result.conserved[hydro::IM1] = density * primitive.velocity[0];
    result.conserved[hydro::IM2] = density * primitive.velocity[1];
    result.conserved[hydro::IM3] = density * primitive.velocity[2];
    result.flux[hydro::IDN] = density * normal_velocity;
    result.flux[hydro::IM1] = density * primitive.velocity[0] * normal_velocity;
    result.flux[hydro::IM2] = density * primitive.velocity[1] * normal_velocity;
    result.flux[hydro::IM3] = density * primitive.velocity[2] * normal_velocity;
    result.flux[hydro::MomentumIndex(direction)] += pressure;
    if (HasEnergy()) {
      const Real energy =
          IdealGas{gamma}.InternalEnergyDensity(pressure) + 0.5 * density * velocity_squared;
      result.conserved[hydro::IEN] = energy;
      result.flux[hydro::IEN] = (energy + pressure) * normal_velocity;
    }
    return result;
  }
};

} // namespace pangu::eos

#endif
