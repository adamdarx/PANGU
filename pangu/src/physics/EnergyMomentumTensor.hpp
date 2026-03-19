#pragma once
#include <basic_types.hpp>
#include "../initialize/mnemonic.hpp"

KOKKOS_INLINE_FUNCTION
void CalculateEnergyMomentumTensor(const parthenon::Real AdiabaticIndex, const parthenon::Real Primitive[PrimitiveVariableNumber], const int Direction, parthenon::Real directedEnergyMomentumTensor[4]) {
    const auto SquaredWeightedVelocity = Kokkos::pow(Primitive[WeightedVelocityX1], 2) + Kokkos::pow(Primitive[WeightedVelocityX2], 2) + Kokkos::pow(Primitive[WeightedVelocityX3], 2);
    const auto SquaredLorentzFactor = 1 + SquaredWeightedVelocity;
    const auto LorentzFactor = Kokkos::sqrt(1 + SquaredWeightedVelocity);
    const auto SquaredMagneticFieldThreeVector = Kokkos::pow(Primitive[MagneticFieldX1], 2) + Kokkos::pow(Primitive[MagneticFieldX2], 2) + Kokkos::pow(Primitive[MagneticFieldX3], 2);
    const auto MagneticFieldThreeVectorDotWeightedVelocity = Primitive[WeightedVelocityX1] * Primitive[MagneticFieldX1] + Primitive[WeightedVelocityX2] * Primitive[MagneticFieldX2] + Primitive[WeightedVelocityX3] * Primitive[MagneticFieldX3];
    const auto SquaredMagneticFieldFourVector = (SquaredMagneticFieldThreeVector + Kokkos::pow(MagneticFieldThreeVectorDotWeightedVelocity, 2)) / SquaredLorentzFactor;
    const auto Enthalpy = Primitive[DensityIndex] + AdiabaticIndex * Primitive[EnergyIndex];     
    const auto Energy = SquaredMagneticFieldFourVector + Enthalpy;
    const auto GasPressure = (AdiabaticIndex - 1.) * Primitive[EnergyIndex];
    const auto TotalPressure = GasPressure + 0.5 * SquaredMagneticFieldFourVector;
    
    const parthenon::Real ContravariantVelocity[4] = {LorentzFactor, Primitive[WeightedVelocityX1], Primitive[WeightedVelocityX2], Primitive[WeightedVelocityX3]};
    const parthenon::Real CovariantVelocity[4] = {-LorentzFactor, Primitive[WeightedVelocityX1], Primitive[WeightedVelocityX2], Primitive[WeightedVelocityX3]};
    const parthenon::Real ContravariantMagneticField[4] = {
        MagneticFieldThreeVectorDotWeightedVelocity,
        (Primitive[MagneticFieldX1] + MagneticFieldThreeVectorDotWeightedVelocity * Primitive[WeightedVelocityX1]) / LorentzFactor,
        (Primitive[MagneticFieldX2] + MagneticFieldThreeVectorDotWeightedVelocity * Primitive[WeightedVelocityX2]) / LorentzFactor,
        (Primitive[MagneticFieldX3] + MagneticFieldThreeVectorDotWeightedVelocity * Primitive[WeightedVelocityX3]) / LorentzFactor,
    };
    const parthenon::Real CovariantMagneticField[4] = {
        -MagneticFieldThreeVectorDotWeightedVelocity,
        (Primitive[MagneticFieldX1] + MagneticFieldThreeVectorDotWeightedVelocity * Primitive[WeightedVelocityX1]) / LorentzFactor,
        (Primitive[MagneticFieldX2] + MagneticFieldThreeVectorDotWeightedVelocity * Primitive[WeightedVelocityX2]) / LorentzFactor,
        (Primitive[MagneticFieldX3] + MagneticFieldThreeVectorDotWeightedVelocity * Primitive[WeightedVelocityX3]) / LorentzFactor,
    };
    
    for(int index = 0; index < 4; index++) 
        directedEnergyMomentumTensor[index] = Energy * ContravariantVelocity[Direction] * CovariantVelocity[index] + TotalPressure * (Direction == index) - ContravariantMagneticField[Direction] * CovariantMagneticField[index];
}
