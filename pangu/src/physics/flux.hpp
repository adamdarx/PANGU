#pragma once

#include <basic_types.hpp>

#include "../initialize/mnemonic.hpp"
#include "EnergyMomentumTensor.hpp"
#include "state.hpp"

KOKKOS_INLINE_FUNCTION
void CalculateContravariantFluxSRMHD(const parthenon::Real AdiabaticIndex,
                                     const parthenon::Real Primitive[PrimitiveVariableNumber],
                                     const int Direction,
                                     parthenon::Real FluxState[PrimitiveVariableNumber]) {
    State StateSRMHD;
    CalculateSRMHDState(Primitive, StateSRMHD);

    parthenon::Real directedEnergyMomentumTensorSpace[4];
    CalculateEnergyMomentumTensorSRMHD(AdiabaticIndex, Primitive, Direction,
                                       directedEnergyMomentumTensorSpace);

    FluxState[DensityIndex] = Primitive[DensityIndex] * StateSRMHD.ContravariantVelocity[Direction];
    FluxState[EnergyIndex] = directedEnergyMomentumTensorSpace[0] + FluxState[DensityIndex];
    FluxState[WeightedVelocityX1] = directedEnergyMomentumTensorSpace[1];
    FluxState[WeightedVelocityX2] = directedEnergyMomentumTensorSpace[2];
    FluxState[WeightedVelocityX3] = directedEnergyMomentumTensorSpace[3];
    FluxState[MagneticFieldX1] =
        StateSRMHD.ContravariantMagneticField[1] * StateSRMHD.ContravariantVelocity[Direction] -
        StateSRMHD.ContravariantMagneticField[Direction] * StateSRMHD.ContravariantVelocity[1];
    FluxState[MagneticFieldX2] =
        StateSRMHD.ContravariantMagneticField[2] * StateSRMHD.ContravariantVelocity[Direction] -
        StateSRMHD.ContravariantMagneticField[Direction] * StateSRMHD.ContravariantVelocity[2];
    FluxState[MagneticFieldX3] =
        StateSRMHD.ContravariantMagneticField[3] * StateSRMHD.ContravariantVelocity[Direction] -
        StateSRMHD.ContravariantMagneticField[Direction] * StateSRMHD.ContravariantVelocity[3];
}

KOKKOS_INLINE_FUNCTION
void CalculateContravariantFluxGRMHD(const parthenon::Real AdiabaticIndex,
                                     const parthenon::Real Primitive[PrimitiveVariableNumber],
                                     const parthenon::Real gcov[4][4],
                                     const parthenon::Real gcon[4][4],
                                     const parthenon::Real gdet,
                                     const int Direction,
                                     parthenon::Real FluxState[PrimitiveVariableNumber]) {
    State StateGRMHD;
    CalculateGRMHDState(Primitive, gcov, gcon, StateGRMHD);

    parthenon::Real directedEnergyMomentumTensorSpace[4];
    CalculateEnergyMomentumTensorGRMHD(AdiabaticIndex, Primitive, gcov, gcon, Direction,
                                       directedEnergyMomentumTensorSpace);

    FluxState[DensityIndex] = Primitive[DensityIndex] * StateGRMHD.ContravariantVelocity[Direction];
    FluxState[EnergyIndex] = directedEnergyMomentumTensorSpace[0] + FluxState[DensityIndex];
    FluxState[WeightedVelocityX1] = directedEnergyMomentumTensorSpace[1];
    FluxState[WeightedVelocityX2] = directedEnergyMomentumTensorSpace[2];
    FluxState[WeightedVelocityX3] = directedEnergyMomentumTensorSpace[3];
    FluxState[MagneticFieldX1] =
        StateGRMHD.ContravariantMagneticField[1] * StateGRMHD.ContravariantVelocity[Direction] -
        StateGRMHD.ContravariantMagneticField[Direction] * StateGRMHD.ContravariantVelocity[1];
    FluxState[MagneticFieldX2] =
        StateGRMHD.ContravariantMagneticField[2] * StateGRMHD.ContravariantVelocity[Direction] -
        StateGRMHD.ContravariantMagneticField[Direction] * StateGRMHD.ContravariantVelocity[2];
    FluxState[MagneticFieldX3] =
        StateGRMHD.ContravariantMagneticField[3] * StateGRMHD.ContravariantVelocity[Direction] -
        StateGRMHD.ContravariantMagneticField[Direction] * StateGRMHD.ContravariantVelocity[3];

    const auto SqrtAbsMetricDeterminant = Kokkos::sqrt(Kokkos::abs(gdet));
    for (int index = 0; index < PrimitiveVariableNumber; ++index) {
        FluxState[index] *= SqrtAbsMetricDeterminant;
    }
}
