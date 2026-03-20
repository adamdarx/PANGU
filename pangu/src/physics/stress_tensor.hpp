#pragma once

#include <basic_types.hpp>

#include "../initialize/mnemonic.hpp"

KOKKOS_INLINE_FUNCTION
void CalculateEnergyMomentumTensor(
    const parthenon::Real adiabatic_index,
    const parthenon::Real primitive[PrimitiveVariableNumber],
    parthenon::Real energy_momentum_tensor[4][4]) {
  const auto squared_weighted_velocity =
      Kokkos::pow(primitive[WeightedVelocityX1], 2) +
      Kokkos::pow(primitive[WeightedVelocityX2], 2) +
      Kokkos::pow(primitive[WeightedVelocityX3], 2);
  const auto squared_lorentz_factor = 1 + squared_weighted_velocity;
  const auto lorentz_factor = Kokkos::sqrt(1 + squared_weighted_velocity);
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
  const auto enthalpy = primitive[DensityIndex] + adiabatic_index * primitive[EnergyIndex];
  const auto energy = squared_magnetic_field_four_vector + enthalpy;
  const auto gas_pressure = (adiabatic_index - 1.) * primitive[EnergyIndex];
  const auto total_pressure = gas_pressure + 0.5 * squared_magnetic_field_four_vector;

  const parthenon::Real contravariant_velocity[4] = {
      lorentz_factor, primitive[WeightedVelocityX1], primitive[WeightedVelocityX2],
      primitive[WeightedVelocityX3]};
  const parthenon::Real covariant_velocity[4] = {
      -lorentz_factor, primitive[WeightedVelocityX1], primitive[WeightedVelocityX2],
      primitive[WeightedVelocityX3]};
  const parthenon::Real contravariant_magnetic_field[4] = {
      magnetic_field_three_vector_dot_weighted_velocity,
      (primitive[MagneticFieldX1] +
       magnetic_field_three_vector_dot_weighted_velocity * primitive[WeightedVelocityX1]) /
          lorentz_factor,
      (primitive[MagneticFieldX2] +
       magnetic_field_three_vector_dot_weighted_velocity * primitive[WeightedVelocityX2]) /
          lorentz_factor,
      (primitive[MagneticFieldX3] +
       magnetic_field_three_vector_dot_weighted_velocity * primitive[WeightedVelocityX3]) /
          lorentz_factor,
  };
  const parthenon::Real covariant_magnetic_field[4] = {
      -magnetic_field_three_vector_dot_weighted_velocity,
      (primitive[MagneticFieldX1] +
       magnetic_field_three_vector_dot_weighted_velocity * primitive[WeightedVelocityX1]) /
          lorentz_factor,
      (primitive[MagneticFieldX2] +
       magnetic_field_three_vector_dot_weighted_velocity * primitive[WeightedVelocityX2]) /
          lorentz_factor,
      (primitive[MagneticFieldX3] +
       magnetic_field_three_vector_dot_weighted_velocity * primitive[WeightedVelocityX3]) /
          lorentz_factor,
  };

  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      energy_momentum_tensor[mu][nu] =
          energy * contravariant_velocity[mu] * covariant_velocity[nu] +
          total_pressure * (mu == nu) -
          contravariant_magnetic_field[mu] * covariant_magnetic_field[nu];
    }
  }
}

KOKKOS_INLINE_FUNCTION
void CalculateEnergyMomentumTensorInDir(
    const parthenon::Real adiabatic_index,
    const parthenon::Real primitive[PrimitiveVariableNumber], const int direction,
    parthenon::Real directed_energy_momentum_tensor[4]) {
  parthenon::Real energy_momentum_tensor[4][4];
  CalculateEnergyMomentumTensor(adiabatic_index, primitive, energy_momentum_tensor);
  for (int index = 0; index < 4; ++index) {
    directed_energy_momentum_tensor[index] = energy_momentum_tensor[direction][index];
  }
}
