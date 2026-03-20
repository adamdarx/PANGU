#pragma once

#include <basic_types.hpp>

#include "../initialize/mnemonic.hpp"

KOKKOS_INLINE_FUNCTION
void TransformConservativeToSRMHD(
    const parthenon::Real conservative_in[PrimitiveVariableNumber],
    const parthenon::Real gcov[4][4], const parthenon::Real gcon[4][4],
    parthenon::Real conservative_sr[PrimitiveVariableNumber]) {
    const parthenon::Real alpha = Kokkos::sqrt(-1.0 / gcon[0][0]);

    for (int n = 0; n < PrimitiveVariableNumber; ++n) {
    conservative_sr[n] = conservative_in[n];
    }

    conservative_sr[DensityIndex] = conservative_in[DensityIndex] * alpha;
    conservative_sr[EnergyIndex] =
        gcon[0][0] * (conservative_in[EnergyIndex] - conservative_in[DensityIndex]) +
        gcon[0][1] * conservative_in[WeightedVelocityX1] +
        gcon[0][2] * conservative_in[WeightedVelocityX2] +
        gcon[0][3] * conservative_in[WeightedVelocityX3];
    conservative_sr[EnergyIndex] *= (-1.0 / gcon[0][0]);
    conservative_sr[EnergyIndex] -= conservative_sr[DensityIndex];

    conservative_sr[WeightedVelocityX1] =
        conservative_in[WeightedVelocityX1] * alpha;
    conservative_sr[WeightedVelocityX2] =
        conservative_in[WeightedVelocityX2] * alpha;
    conservative_sr[WeightedVelocityX3] =
        conservative_in[WeightedVelocityX3] * alpha;

    conservative_sr[MagneticFieldX1] = conservative_in[MagneticFieldX1] * alpha;
    conservative_sr[MagneticFieldX2] = conservative_in[MagneticFieldX2] * alpha;
    conservative_sr[MagneticFieldX3] = conservative_in[MagneticFieldX3] * alpha;

    (void)gcov;
}

KOKKOS_INLINE_FUNCTION
void TransformConservativeToGRMHD(
    const parthenon::Real conservative_sr[PrimitiveVariableNumber],
    const parthenon::Real gcov[4][4], const parthenon::Real gcon[4][4],
    parthenon::Real conservative_out[PrimitiveVariableNumber]) {
    const parthenon::Real alpha = Kokkos::sqrt(-1.0 / gcon[0][0]);

    for (int n = 0; n < PrimitiveVariableNumber; ++n) {
    conservative_out[n] = conservative_sr[n];
    }

    conservative_out[DensityIndex] = conservative_sr[DensityIndex] / alpha;
    conservative_out[WeightedVelocityX1] =
        conservative_sr[WeightedVelocityX1] / alpha;
    conservative_out[WeightedVelocityX2] =
        conservative_sr[WeightedVelocityX2] / alpha;
    conservative_out[WeightedVelocityX3] =
        conservative_sr[WeightedVelocityX3] / alpha;
    conservative_out[MagneticFieldX1] = conservative_sr[MagneticFieldX1] / alpha;
    conservative_out[MagneticFieldX2] = conservative_sr[MagneticFieldX2] / alpha;
    conservative_out[MagneticFieldX3] = conservative_sr[MagneticFieldX3] / alpha;

    conservative_out[EnergyIndex] = conservative_sr[EnergyIndex];
    (void)gcov;
}
