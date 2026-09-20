#ifndef PANGU_EOS_IDEAL_GAS_H_
#define PANGU_EOS_IDEAL_GAS_H_

#include <parthenon/parthenon.hpp>

namespace pangu::eos {

// Thermodynamic closure shared by the Newtonian, SR, and GR ideal-gas paths.
// Kinematic and metric operations deliberately remain in their owning solvers.
struct IdealGas {
  parthenon::Real gamma;

  KOKKOS_INLINE_FUNCTION
  parthenon::Real InternalEnergyDensity(const parthenon::Real pressure) const {
    return pressure / (gamma - 1.0);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real PressureFromInternalEnergyDensity(
      const parthenon::Real internal_energy_density) const {
    return (gamma - 1.0) * internal_energy_density;
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real SpecificInternalEnergy(const parthenon::Real density,
                                         const parthenon::Real pressure) const {
    return InternalEnergyDensity(pressure) / density;
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real PressureFromSpecificInternalEnergy(
      const parthenon::Real density, const parthenon::Real specific_internal_energy) const {
    return PressureFromInternalEnergyDensity(density * specific_internal_energy);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real EnthalpyDensity(const parthenon::Real density,
                                  const parthenon::Real pressure) const {
    return density + gamma * InternalEnergyDensity(pressure);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real SpecificEnthalpyFromSpecificInternalEnergy(
      const parthenon::Real specific_internal_energy) const {
    return 1.0 + gamma * specific_internal_energy;
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real RelativisticSoundSpeedSquared(const parthenon::Real density,
                                                const parthenon::Real pressure) const {
    return gamma * pressure / EnthalpyDensity(density, pressure);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real NewtonianSoundSpeedSquared(const parthenon::Real density,
                                             const parthenon::Real pressure) const {
    return gamma * pressure / density;
  }
};

} // namespace pangu::eos

#endif
