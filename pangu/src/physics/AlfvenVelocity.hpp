#pragma once
#include <basic_types.hpp>
#include "../initialize/mnemonic.hpp"

KOKKOS_INLINE_FUNCTION
void CalculateAlfvenVelocity(const parthenon::Real AdiabaticIndex, const parthenon::Real Primitive[PrimitiveVariableNumber], const int Direction, parthenon::Real& greaterVelocity, parthenon::Real& lessVelocity) {
    const auto SquaredWeightedVelocity = Kokkos::pow(Primitive[WeightedVelocityX1], 2) + Kokkos::pow(Primitive[WeightedVelocityX2], 2) + Kokkos::pow(Primitive[WeightedVelocityX3], 2);
    const auto SquaredLorentzFactor = 1 + SquaredWeightedVelocity;
    const auto SquaredMagneticFieldThreeVector = Kokkos::pow(Primitive[MagneticFieldX1], 2) + Kokkos::pow(Primitive[MagneticFieldX2], 2) + Kokkos::pow(Primitive[MagneticFieldX3], 2);
    const auto MagneticFieldThreeVectorDotWeightedVelocity = Primitive[WeightedVelocityX1] * Primitive[MagneticFieldX1] + Primitive[WeightedVelocityX2] * Primitive[MagneticFieldX2] + Primitive[WeightedVelocityX3] * Primitive[MagneticFieldX3];
    const auto SquaredMagneticFieldFourVector = (SquaredMagneticFieldThreeVector + Kokkos::pow(MagneticFieldThreeVectorDotWeightedVelocity, 2)) / SquaredLorentzFactor;
    const auto Enthalpy = Primitive[DensityIndex] + AdiabaticIndex * Primitive[EnergyIndex];     
    const auto Energy = SquaredMagneticFieldFourVector + Enthalpy;           
    const auto SquaredAlfvenVelocity = SquaredMagneticFieldFourVector / Energy;          
    const auto SquaredSoundSpeed = AdiabaticIndex * (AdiabaticIndex - 1.) * Primitive[EnergyIndex] / Enthalpy;
    auto SpeedOfFastMagnetosonicWave = SquaredSoundSpeed + SquaredAlfvenVelocity - SquaredSoundSpeed * SquaredAlfvenVelocity;
    
    if (SpeedOfFastMagnetosonicWave < 0.) {
        SpeedOfFastMagnetosonicWave = 1e-10;
    }
    else if (SpeedOfFastMagnetosonicWave > 1.) {
        SpeedOfFastMagnetosonicWave = 1.;
    }

    const int TimeFactor = -1 * (Direction == 0);
    const parthenon::Real SquaredDirectedContravariantVelocity = SquaredLorentzFactor * (Direction == 0) + Kokkos::pow(Primitive[1 + Direction], 2) * (Direction != 0);
    const parthenon::Real WeightedDirectedContravariantVelocity = SquaredLorentzFactor * (Direction == 0) + Kokkos::sqrt(SquaredLorentzFactor) * Primitive[1 + Direction] * (Direction != 0);
    const parthenon::Real CoefficientOfQuadraticItem = SquaredLorentzFactor - (SquaredLorentzFactor - 1) * SpeedOfFastMagnetosonicWave;
    const parthenon::Real CoefficientOfLinearItem = 2 * (WeightedDirectedContravariantVelocity - (TimeFactor + WeightedDirectedContravariantVelocity) * SpeedOfFastMagnetosonicWave);
    const parthenon::Real Asq = -1 * (Direction == 0) + 1 * (Direction != 0);
    const parthenon::Real ConstantItem = SquaredDirectedContravariantVelocity - (Asq + SquaredDirectedContravariantVelocity) * SpeedOfFastMagnetosonicWave;
    auto Discriminant = CoefficientOfLinearItem * CoefficientOfLinearItem - 4. * CoefficientOfQuadraticItem * ConstantItem;
    
    if ((Discriminant < 0.0) && (Discriminant > -1.e-10)) {
        Discriminant = 0.0;
    }
    else if (Discriminant < -1.e-10) {
        Discriminant = 0.;
    }

    const auto VelocityWithPlus = -(-CoefficientOfLinearItem + Kokkos::sqrt(Discriminant)) / (2. * CoefficientOfQuadraticItem);
    const auto VelocityWithMinus = -(-CoefficientOfLinearItem - Kokkos::sqrt(Discriminant)) / (2. * CoefficientOfQuadraticItem);

    if (VelocityWithPlus > VelocityWithMinus) {
        greaterVelocity = VelocityWithPlus;
        lessVelocity = VelocityWithMinus;
    }
    else {
        greaterVelocity = VelocityWithMinus;
        lessVelocity = VelocityWithPlus;
    }
}
