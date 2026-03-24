#pragma once
#include <basic_types.hpp>
#include "../initialize/mnemonic.hpp"
#include "../metric/linear_algebra.hpp"
#include "state.hpp"

KOKKOS_INLINE_FUNCTION
void CalculateEnergyMomentumTensorSRMHD(const parthenon::Real AdiabaticIndex, const parthenon::Real Primitive[PrimitiveVariableNumber], const int Direction, parthenon::Real directedEnergyMomentumTensor[4]) {
    State StateSRMHD;
    CalculateSRMHDState(Primitive, StateSRMHD);

    const auto SquaredMagneticFieldFourVector =
        dot4(StateSRMHD.ContravariantMagneticField, StateSRMHD.CovariantMagneticField);
    const auto Enthalpy = Primitive[DensityIndex] + AdiabaticIndex * Primitive[EnergyIndex];
    const auto Energy = SquaredMagneticFieldFourVector + Enthalpy;
    const auto GasPressure = (AdiabaticIndex - 1.) * Primitive[EnergyIndex];
    const auto TotalPressure = GasPressure + 0.5 * SquaredMagneticFieldFourVector;

    for(int index = 0; index < 4; index++) 
        directedEnergyMomentumTensor[index] =
            Energy * StateSRMHD.ContravariantVelocity[Direction] * StateSRMHD.CovariantVelocity[index] +
            TotalPressure * (Direction == index) -
            StateSRMHD.ContravariantMagneticField[Direction] * StateSRMHD.CovariantMagneticField[index];
}

KOKKOS_INLINE_FUNCTION
void CalculateEnergyMomentumTensorGRMHD(const parthenon::Real AdiabaticIndex,
                                        const parthenon::Real Primitive[PrimitiveVariableNumber],
                                        const parthenon::Real gcov[4][4],
                                        const parthenon::Real gcon[4][4],
                                        const int Direction,
                                        parthenon::Real directedEnergyMomentumTensor[4]) {
    State StateGRMHD;
    CalculateGRMHDState(Primitive, gcov, gcon, StateGRMHD);

    const auto SquaredMagneticFieldFourVector =
        dot4(StateGRMHD.ContravariantMagneticField, StateGRMHD.CovariantMagneticField);
    const auto GasPressure = (AdiabaticIndex - 1.) * Primitive[EnergyIndex];
    const auto Enthalpy = Primitive[DensityIndex] + AdiabaticIndex * Primitive[EnergyIndex];
    const auto Energy = Enthalpy + SquaredMagneticFieldFourVector;
    const auto TotalPressure = GasPressure + 0.5 * SquaredMagneticFieldFourVector;

    for (int index = 0; index < 4; ++index) {
        directedEnergyMomentumTensor[index] =
            Energy * StateGRMHD.ContravariantVelocity[Direction] * StateGRMHD.CovariantVelocity[index] +
            TotalPressure * (Direction == index) -
            StateGRMHD.ContravariantMagneticField[Direction] * StateGRMHD.CovariantMagneticField[index];
    }
}

KOKKOS_INLINE_FUNCTION
void CalculateEnergyMomentumTensorGRMHD(const parthenon::Real AdiabaticIndex,
                                        const parthenon::Real Primitive[PrimitiveVariableNumber],
                                        const parthenon::Real gcov[4][4],
                                        const parthenon::Real gcon[4][4],
                                        parthenon::Real energyMomentumTensor[4][4]) {
    State StateGRMHD;
    CalculateGRMHDState(Primitive, gcov, gcon, StateGRMHD);

    const auto SquaredMagneticFieldFourVector =
        dot4(StateGRMHD.ContravariantMagneticField, StateGRMHD.CovariantMagneticField);
    const auto GasPressure = (AdiabaticIndex - 1.) * Primitive[EnergyIndex];
    const auto Enthalpy = Primitive[DensityIndex] + AdiabaticIndex * Primitive[EnergyIndex];
    const auto Energy = Enthalpy + SquaredMagneticFieldFourVector;
    const auto TotalPressure = GasPressure + 0.5 * SquaredMagneticFieldFourVector;

    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            energyMomentumTensor[row][col] =
                Energy * StateGRMHD.ContravariantVelocity[row] * StateGRMHD.ContravariantVelocity[col] +
                TotalPressure * gcon[row][col] -
                StateGRMHD.ContravariantMagneticField[row] * StateGRMHD.ContravariantMagneticField[col];
        }
    }
}
