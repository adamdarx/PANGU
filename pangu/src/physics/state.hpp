#pragma once

#include <basic_types.hpp>

#include "../initialize/mnemonic.hpp"
#include "../metric/linear_algebra.hpp"

struct State {
    parthenon::Real ContravariantVelocity[4];
    parthenon::Real CovariantVelocity[4];
    parthenon::Real ContravariantMagneticField[4];
    parthenon::Real CovariantMagneticField[4];
};

KOKKOS_INLINE_FUNCTION
void CalculateSRMHDState(const parthenon::Real Primitive[PrimitiveVariableNumber],
                         State &StateGRMHD) {
    const auto SquaredWeightedVelocity =
        Kokkos::pow(Primitive[WeightedVelocityX1], 2) +
        Kokkos::pow(Primitive[WeightedVelocityX2], 2) +
        Kokkos::pow(Primitive[WeightedVelocityX3], 2);
    const auto LorentzFactor = Kokkos::sqrt(1.0 + SquaredWeightedVelocity);

    StateGRMHD.ContravariantVelocity[0] = LorentzFactor;
    StateGRMHD.ContravariantVelocity[1] = Primitive[WeightedVelocityX1];
    StateGRMHD.ContravariantVelocity[2] = Primitive[WeightedVelocityX2];
    StateGRMHD.ContravariantVelocity[3] = Primitive[WeightedVelocityX3];

    StateGRMHD.CovariantVelocity[0] = -LorentzFactor;
    StateGRMHD.CovariantVelocity[1] = Primitive[WeightedVelocityX1];
    StateGRMHD.CovariantVelocity[2] = Primitive[WeightedVelocityX2];
    StateGRMHD.CovariantVelocity[3] = Primitive[WeightedVelocityX3];

    const auto MagneticFieldThreeVectorDotWeightedVelocity =
        Primitive[WeightedVelocityX1] * Primitive[MagneticFieldX1] +
        Primitive[WeightedVelocityX2] * Primitive[MagneticFieldX2] +
        Primitive[WeightedVelocityX3] * Primitive[MagneticFieldX3];

    StateGRMHD.ContravariantMagneticField[0] = MagneticFieldThreeVectorDotWeightedVelocity;
    StateGRMHD.ContravariantMagneticField[1] =
        (Primitive[MagneticFieldX1] + MagneticFieldThreeVectorDotWeightedVelocity * Primitive[WeightedVelocityX1]) /
        LorentzFactor;
    StateGRMHD.ContravariantMagneticField[2] =
        (Primitive[MagneticFieldX2] + MagneticFieldThreeVectorDotWeightedVelocity * Primitive[WeightedVelocityX2]) /
        LorentzFactor;
    StateGRMHD.ContravariantMagneticField[3] =
        (Primitive[MagneticFieldX3] + MagneticFieldThreeVectorDotWeightedVelocity * Primitive[WeightedVelocityX3]) /
        LorentzFactor;

    StateGRMHD.CovariantMagneticField[0] = -MagneticFieldThreeVectorDotWeightedVelocity;
    StateGRMHD.CovariantMagneticField[1] = StateGRMHD.ContravariantMagneticField[1];
    StateGRMHD.CovariantMagneticField[2] = StateGRMHD.ContravariantMagneticField[2];
    StateGRMHD.CovariantMagneticField[3] = StateGRMHD.ContravariantMagneticField[3];
}

KOKKOS_INLINE_FUNCTION
void CalculateGRMHDState(const parthenon::Real Primitive[PrimitiveVariableNumber],
                         const parthenon::Real gcov[4][4],
                         const parthenon::Real gcon[4][4],
                         State &StateGRMHD) {
    const auto Lapse = 1.0 / Kokkos::sqrt(-gcon[0][0]);

    parthenon::Real SpatialMetric[3][3] = {{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};
    for (int row = 1; row < 4; ++row) {
        for (int col = 1; col < 4; ++col) {
            SpatialMetric[row - 1][col - 1] = gcov[row][col];
        }
    }

    const parthenon::Real WeightedVelocity[3] = {
        Primitive[WeightedVelocityX1],
        Primitive[WeightedVelocityX2],
        Primitive[WeightedVelocityX3]
    };
    const auto SquaredLorentzFactor = 1.0 + square3(WeightedVelocity, SpatialMetric);
    const auto LorentzFactor = Kokkos::sqrt(SquaredLorentzFactor);

    StateGRMHD.ContravariantVelocity[0] = LorentzFactor / Lapse;
    StateGRMHD.ContravariantVelocity[1] = Primitive[WeightedVelocityX1] - StateGRMHD.ContravariantVelocity[0] * gcon[0][1] * Lapse * Lapse;
    StateGRMHD.ContravariantVelocity[2] = Primitive[WeightedVelocityX2] - StateGRMHD.ContravariantVelocity[0] * gcon[0][2] * Lapse * Lapse;
    StateGRMHD.ContravariantVelocity[3] = Primitive[WeightedVelocityX3] - StateGRMHD.ContravariantVelocity[0] * gcon[0][3] * Lapse * Lapse;

    for (int row = 0; row < 4; ++row) {
        StateGRMHD.CovariantVelocity[row] = 0.0;
        for (int col = 0; col < 4; ++col) {
            StateGRMHD.CovariantVelocity[row] += gcov[row][col] * StateGRMHD.ContravariantVelocity[col];
        }
    }

    StateGRMHD.ContravariantMagneticField[0] =
        Primitive[MagneticFieldX1] * StateGRMHD.CovariantVelocity[1] +
        Primitive[MagneticFieldX2] * StateGRMHD.CovariantVelocity[2] +
        Primitive[MagneticFieldX3] * StateGRMHD.CovariantVelocity[3];

    StateGRMHD.ContravariantMagneticField[1] =
        (Primitive[MagneticFieldX1] + StateGRMHD.ContravariantMagneticField[0] * StateGRMHD.ContravariantVelocity[1]) /
        StateGRMHD.ContravariantVelocity[0];
    StateGRMHD.ContravariantMagneticField[2] =
        (Primitive[MagneticFieldX2] + StateGRMHD.ContravariantMagneticField[0] * StateGRMHD.ContravariantVelocity[2]) /
        StateGRMHD.ContravariantVelocity[0];
    StateGRMHD.ContravariantMagneticField[3] =
        (Primitive[MagneticFieldX3] + StateGRMHD.ContravariantMagneticField[0] * StateGRMHD.ContravariantVelocity[3]) /
        StateGRMHD.ContravariantVelocity[0];

    for (int row = 0; row < 4; ++row) {
        StateGRMHD.CovariantMagneticField[row] = 0.0;
        for (int col = 0; col < 4; ++col) {
            StateGRMHD.CovariantMagneticField[row] += gcov[row][col] * StateGRMHD.ContravariantMagneticField[col];
        }
    }
}
