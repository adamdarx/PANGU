#ifndef PANGU_EOS_RELATIVISTIC_EOS_H_
#define PANGU_EOS_RELATIVISTIC_EOS_H_

#include <cmath>
#include <limits>

#include <parthenon/parthenon.hpp>

#include "eos/ideal_gas.h"

namespace pangu::eos {

// Ideal-gas closure and thermodynamic limits of the SR, fixed-background GR and
// dynamical-spacetime paths.  Kinematics, metric contractions and the conserved
// inversion iteration stay with their solvers; those solvers take the closure
// relations and every thermodynamic floor from here.
//
// The Lorentz and magnetization ceilings are recovery limits rather than
// thermodynamics and remain arguments of the recovery routines.
struct RelativisticEOS : IdealGas {
  parthenon::Real density_floor;
  parthenon::Real pressure_floor;
  // Only magnetized recovery consults the entropy floor; the default disables it.
  parthenon::Real entropy_floor = static_cast<parthenon::Real>(std::numeric_limits<float>::min());

  KOKKOS_INLINE_FUNCTION
  parthenon::Real SoundSpeedSquared(const parthenon::Real density,
                                    const parthenon::Real pressure) const {
    return RelativisticSoundSpeedSquared(density, pressure);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real InternalEnergyDensityFloor() const {
    return InternalEnergyDensity(pressure_floor);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real SpecificInternalEnergyFloor(const parthenon::Real density) const {
    return SpecificInternalEnergy(density, pressure_floor);
  }

  // Magnetized recovery additionally imposes the entropy floor.  Its pressure
  // term divides by density and gamma - 1 in AthenaK's order, which differs in
  // the last ulp from the unmagnetized floor above; both orders are kept so the
  // two recovery paths reproduce their references bit for bit.
  KOKKOS_INLINE_FUNCTION
  parthenon::Real MagnetisedSpecificInternalEnergyFloor(const parthenon::Real density) const {
    const parthenon::Real gamma_minus_one = gamma - 1.0;
    return fmax(pressure_floor / (density * gamma_minus_one),
                entropy_floor * pow(density, gamma_minus_one) / gamma_minus_one);
  }
};

// Fluid packages register the thermodynamic parameters under these names.  The
// entropy floor belongs to the magnetized packages only, so the two readers stay
// separate instead of silently tolerating a missing parameter.
inline RelativisticEOS ReadRelativistic(const parthenon::StateDescriptor& package) {
  return {{package.Param<parthenon::Real>("gamma")},
          package.Param<parthenon::Real>("density_floor"),
          package.Param<parthenon::Real>("pressure_floor")};
}

inline RelativisticEOS ReadMagnetised(const parthenon::StateDescriptor& package) {
  auto eos = ReadRelativistic(package);
  eos.entropy_floor = package.Param<parthenon::Real>("entropy_floor");
  return eos;
}

} // namespace pangu::eos

#endif
