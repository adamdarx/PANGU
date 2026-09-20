#ifndef PANGU_UNITS_UNITS_H_
#define PANGU_UNITS_UNITS_H_

#include <memory>

#include <parthenon/parthenon.hpp>

namespace pangu::units {

// Code-unit scales follow AthenaK's M-L-T convention.  Keeping the conversion
// algebra in this trivially-copyable object lets device kernels consume units
// without consulting ParameterInput or host-only package state.
struct Units {
  static constexpr parthenon::Real pc_cgs = 3.0856775809623245e18;
  static constexpr parthenon::Real kpc_cgs = 3.0856775809623245e21;
  static constexpr parthenon::Real msun_cgs = 1.98841586e33;
  static constexpr parthenon::Real atomic_mass_unit_cgs = 1.660538921e-24;
  static constexpr parthenon::Real year_cgs = 3.15576e7;
  static constexpr parthenon::Real myr_cgs = 3.15576e13;
  static constexpr parthenon::Real boltzmann_cgs = 1.3806488e-16;
  static constexpr parthenon::Real gravitational_constant_cgs = 6.67408e-8;
  static constexpr parthenon::Real speed_of_light_cgs = 2.99792458e10;
  static constexpr parthenon::Real radiation_constant_cgs = 7.56573325e-15;
  parthenon::Real length = 1.0;
  parthenon::Real mass = 1.0;
  parthenon::Real time = 1.0;
  parthenon::Real mean_molecular_weight = 1.0;

  KOKKOS_INLINE_FUNCTION parthenon::Real Velocity() const { return length / time; }
  KOKKOS_INLINE_FUNCTION parthenon::Real Density() const {
    return mass / (length * length * length);
  }
  KOKKOS_INLINE_FUNCTION parthenon::Real Energy() const { return mass * Velocity() * Velocity(); }
  KOKKOS_INLINE_FUNCTION parthenon::Real Pressure() const {
    return Energy() / (length * length * length);
  }
  KOKKOS_INLINE_FUNCTION parthenon::Real Temperature() const {
    return Velocity() * Velocity() * mean_molecular_weight * atomic_mass_unit_cgs / boltzmann_cgs;
  }
  KOKKOS_INLINE_FUNCTION parthenon::Real NumberDensity() const {
    return Density() / (mean_molecular_weight * atomic_mass_unit_cgs);
  }
  KOKKOS_INLINE_FUNCTION parthenon::Real Centimeter() const { return 1.0 / length; }
  KOKKOS_INLINE_FUNCTION parthenon::Real Parsec() const { return pc_cgs / length; }
  KOKKOS_INLINE_FUNCTION parthenon::Real Kiloparsec() const { return kpc_cgs / length; }
  KOKKOS_INLINE_FUNCTION parthenon::Real Gram() const { return 1.0 / mass; }
  KOKKOS_INLINE_FUNCTION parthenon::Real SolarMass() const { return msun_cgs / mass; }
  KOKKOS_INLINE_FUNCTION parthenon::Real Second() const { return 1.0 / time; }
  KOKKOS_INLINE_FUNCTION parthenon::Real Year() const { return year_cgs / time; }
  KOKKOS_INLINE_FUNCTION parthenon::Real Megayear() const { return myr_cgs / time; }
  KOKKOS_INLINE_FUNCTION parthenon::Real KilometerPerSecond() const { return 1.0e5 / Velocity(); }
  KOKKOS_INLINE_FUNCTION parthenon::Real Boltzmann() const {
    return boltzmann_cgs / (Energy() / Temperature());
  }
  KOKKOS_INLINE_FUNCTION parthenon::Real GravitationalConstant() const {
    return gravitational_constant_cgs * Density() * time * time;
  }
  KOKKOS_INLINE_FUNCTION parthenon::Real SpeedOfLight() const {
    return speed_of_light_cgs / Velocity();
  }
};

std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput* pin);
Units FromPackage(const std::shared_ptr<parthenon::StateDescriptor>& package);

} // namespace pangu::units

#endif
