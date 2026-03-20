#pragma once

#include <basic_types.hpp>

#include "../initialize/mnemonic.hpp"

KOKKOS_INLINE_FUNCTION
void CalculateAlfvenVelocitySRMHD(
  const parthenon::Real adiabatic_index,
  const parthenon::Real primitive[PrimitiveVariableNumber], const int direction,
  parthenon::Real &greater_velocity, parthenon::Real &less_velocity) {
  const auto squared_weighted_velocity =
    Kokkos::pow(primitive[WeightedVelocityX1], 2) +
    Kokkos::pow(primitive[WeightedVelocityX2], 2) +
    Kokkos::pow(primitive[WeightedVelocityX3], 2);
  const auto squared_lorentz_factor = 1 + squared_weighted_velocity;
  const auto squared_magnetic_field_three_vector =
    Kokkos::pow(primitive[MagneticFieldX1], 2) +
    Kokkos::pow(primitive[MagneticFieldX2], 2) +
    Kokkos::pow(primitive[MagneticFieldX3], 2);
  const auto magnetic_field_three_vector_dot_weighted_velocity =
    primitive[WeightedVelocityX1] * primitive[MagneticFieldX1] +
    primitive[WeightedVelocityX2] * primitive[MagneticFieldX2] +
    primitive[WeightedVelocityX3] * primitive[MagneticFieldX3];
  const auto squared_magnetic_field_four_vector =
    (squared_magnetic_field_three_vector +
     Kokkos::pow(magnetic_field_three_vector_dot_weighted_velocity, 2)) /
    squared_lorentz_factor;
  const auto enthalpy =
    primitive[DensityIndex] + adiabatic_index * primitive[EnergyIndex];
  const auto energy = squared_magnetic_field_four_vector + enthalpy;
  const auto squared_alfven_velocity = squared_magnetic_field_four_vector / energy;
  const auto squared_sound_speed =
    adiabatic_index * (adiabatic_index - 1.) * primitive[EnergyIndex] / enthalpy;
  auto speed_of_fast_magnetosonic_wave =
    squared_sound_speed + squared_alfven_velocity -
    squared_sound_speed * squared_alfven_velocity;

  if (speed_of_fast_magnetosonic_wave < 0.) {
  speed_of_fast_magnetosonic_wave = 1e-10;
  } else if (speed_of_fast_magnetosonic_wave > 1.) {
  speed_of_fast_magnetosonic_wave = 1.;
  }

  const int time_factor = -1 * (direction == 0);
  const parthenon::Real squared_directed_contravariant_velocity =
    squared_lorentz_factor * (direction == 0) +
    Kokkos::pow(primitive[1 + direction], 2) * (direction != 0);
  const parthenon::Real weighted_directed_contravariant_velocity =
    squared_lorentz_factor * (direction == 0) +
    Kokkos::sqrt(squared_lorentz_factor) * primitive[1 + direction] *
      (direction != 0);
  const parthenon::Real coefficient_of_quadratic_item =
    squared_lorentz_factor -
    (squared_lorentz_factor - 1) * speed_of_fast_magnetosonic_wave;
  const parthenon::Real coefficient_of_linear_item =
    2 * (weighted_directed_contravariant_velocity -
       (time_factor + weighted_directed_contravariant_velocity) *
         speed_of_fast_magnetosonic_wave);
  const parthenon::Real asq = -1 * (direction == 0) + 1 * (direction != 0);
  const parthenon::Real constant_item =
    squared_directed_contravariant_velocity -
    (asq + squared_directed_contravariant_velocity) *
      speed_of_fast_magnetosonic_wave;
  auto discriminant =
    coefficient_of_linear_item * coefficient_of_linear_item -
    4. * coefficient_of_quadratic_item * constant_item;

  if ((discriminant < 0.0) && (discriminant > -1.e-10)) {
  discriminant = 0.0;
  } else if (discriminant < -1.e-10) {
  discriminant = 0.;
  }

  const auto velocity_with_plus =
    -(-coefficient_of_linear_item + Kokkos::sqrt(discriminant)) /
    (2. * coefficient_of_quadratic_item);
  const auto velocity_with_minus =
    -(-coefficient_of_linear_item - Kokkos::sqrt(discriminant)) /
    (2. * coefficient_of_quadratic_item);

  if (velocity_with_plus > velocity_with_minus) {
  greater_velocity = velocity_with_plus;
  less_velocity = velocity_with_minus;
  } else {
  greater_velocity = velocity_with_minus;
  less_velocity = velocity_with_plus;
  }
}

KOKKOS_INLINE_FUNCTION
void CalculateAlfvenVelocityGRMHD(const parthenon::Real adiabatic_index,
                  const parthenon::Real primitive[PrimitiveVariableNumber],
                                  const parthenon::Real gcov[4][4],
                                  const parthenon::Real gcon[4][4],
                  const int direction,
                  parthenon::Real &greater_velocity,
                  parthenon::Real &less_velocity) {
  (void)gcov;
  (void)gcon;
  // Use SR estimate as a robust fallback until full GR wave-speed polynomial is added.
  CalculateAlfvenVelocitySRMHD(adiabatic_index, primitive, direction,
                 greater_velocity, less_velocity);
}

KOKKOS_INLINE_FUNCTION
void CalculateAlfvenVelocityGRMHD(const parthenon::Real adiabatic_index,
                  const parthenon::Real primitive[PrimitiveVariableNumber],
                  const int direction,
                  parthenon::Real &greater_velocity,
                  parthenon::Real &less_velocity) {
  CalculateAlfvenVelocitySRMHD(adiabatic_index, primitive, direction,
                 greater_velocity, less_velocity);
}
