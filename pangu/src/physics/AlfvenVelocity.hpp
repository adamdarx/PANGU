#pragma once
#include <basic_types.hpp>
#include "../initialize/mnemonic.hpp"
#include "../metric/linear_algebra.hpp"
#include "state.hpp"

KOKKOS_INLINE_FUNCTION
void CalculateAlfvenVelocitySRMHD(const parthenon::Real AdiabaticIndex, const parthenon::Real Primitive[PrimitiveVariableNumber], const int Direction, parthenon::Real& greaterVelocity, parthenon::Real& lessVelocity) {
    State StateSRMHD;
    CalculateSRMHDState(Primitive, StateSRMHD);

    const auto SquaredMagneticFieldFourVector =
        dot4(StateSRMHD.ContravariantMagneticField, StateSRMHD.CovariantMagneticField);
    const auto FluidInertia = Primitive[DensityIndex] + AdiabaticIndex * Primitive[EnergyIndex];
    const auto TotalInertia = SquaredMagneticFieldFourVector + FluidInertia;
    const auto SquaredAlfvenVelocity = SquaredMagneticFieldFourVector / TotalInertia;
    const auto SquaredSoundSpeed = AdiabaticIndex * (AdiabaticIndex - 1.) * Primitive[EnergyIndex] / FluidInertia;
    auto SpeedOfFastMagnetosonicWave = SquaredSoundSpeed + SquaredAlfvenVelocity - SquaredSoundSpeed * SquaredAlfvenVelocity;
    
    if (SpeedOfFastMagnetosonicWave < 0.) {
        SpeedOfFastMagnetosonicWave = 1e-10;
    }
    else if (SpeedOfFastMagnetosonicWave > 1.) {
        SpeedOfFastMagnetosonicWave = 1.;
    }

    parthenon::Real DirectedContravariantVector[4] = {0., 0., 0., 0.};
    parthenon::Real DirectedCovariantVector[4] = {0., 0., 0., 0.};
    parthenon::Real TimeContravariantVector[4] = {-1., 0., 0., 0.};
    parthenon::Real TimeCovariantVector[4] = {1., 0., 0., 0.};

    DirectedCovariantVector[Direction] = 1.0;
    DirectedContravariantVector[Direction] = (Direction == 0) ? -1.0 : 1.0;

    const auto SquaredDirectedContravariantVector = dot4(DirectedContravariantVector, DirectedCovariantVector);
    const auto SquaredTimeContravariantVector = dot4(TimeContravariantVector, TimeCovariantVector);
    const auto DirectedContravariantVectorDotFluidContravariantVelocity = dot4(DirectedCovariantVector, StateSRMHD.ContravariantVelocity);
    const auto TimeCovariantVectorDotFluidContravariantVelocity = dot4(TimeCovariantVector, StateSRMHD.ContravariantVelocity);
    const auto DirectedContravariantVectorDotTimeCovariantVector = dot4(DirectedContravariantVector, TimeCovariantVector);

    const auto SquaredDirectedContravariantVectorDotFluidContravariantVelocity =
        DirectedContravariantVectorDotFluidContravariantVelocity * DirectedContravariantVectorDotFluidContravariantVelocity;
    const auto SquaredTimeCovariantVectorDotFluidContravariantVelocity =
        TimeCovariantVectorDotFluidContravariantVelocity * TimeCovariantVectorDotFluidContravariantVelocity;
    const auto DirectedContravariantVectorDotFluidContravariantVelocityMulTimeCovariantVectorDotFluidContravariantVelocity =
        DirectedContravariantVectorDotFluidContravariantVelocity * TimeCovariantVectorDotFluidContravariantVelocity;

    const auto CoefficientOfQuadraticItem =
        SquaredTimeCovariantVectorDotFluidContravariantVelocity -
        (SquaredTimeContravariantVector + SquaredTimeCovariantVectorDotFluidContravariantVelocity) * SpeedOfFastMagnetosonicWave;
    const auto CoefficientOfLinearItem =
        2. * (DirectedContravariantVectorDotFluidContravariantVelocityMulTimeCovariantVectorDotFluidContravariantVelocity -
              (DirectedContravariantVectorDotTimeCovariantVector + DirectedContravariantVectorDotFluidContravariantVelocityMulTimeCovariantVectorDotFluidContravariantVelocity) *
                  SpeedOfFastMagnetosonicWave);
    const auto ConstantItem =
        SquaredDirectedContravariantVectorDotFluidContravariantVelocity -
        (SquaredDirectedContravariantVector + SquaredDirectedContravariantVectorDotFluidContravariantVelocity) * SpeedOfFastMagnetosonicWave;

    auto Discriminant = CoefficientOfLinearItem * CoefficientOfLinearItem -
                        4. * CoefficientOfQuadraticItem * ConstantItem;
    
    if ((Discriminant < 0.0) && (Discriminant > -1.e-10)) {
        Discriminant = 0.0;
    }
    else if (Discriminant < -1.e-10) {
        Discriminant = 0.;
    }

    Discriminant = Kokkos::sqrt(Discriminant);
    const auto VelocityWithPlus = -(-CoefficientOfLinearItem + Discriminant) / (2. * CoefficientOfQuadraticItem);
    const auto VelocityWithMinus = -(-CoefficientOfLinearItem - Discriminant) / (2. * CoefficientOfQuadraticItem);

    if (VelocityWithPlus > VelocityWithMinus) {
        greaterVelocity = VelocityWithPlus;
        lessVelocity = VelocityWithMinus;
    }
    else {
        greaterVelocity = VelocityWithMinus;
        lessVelocity = VelocityWithPlus;
    }
}

KOKKOS_INLINE_FUNCTION
void CalculateAlfvenVelocityGRMHD(const parthenon::Real AdiabaticIndex,
                                  const parthenon::Real Primitive[PrimitiveVariableNumber],
                                  const parthenon::Real gcov[4][4],
                                  const parthenon::Real gcon[4][4],
                                  const int Direction,
                                  parthenon::Real &greaterVelocity,
                                  parthenon::Real &lessVelocity) {
    parthenon::Real DirectedContravariantVector[4] = {0., 0., 0., 0.};
    parthenon::Real DirectedCovariantVector[4] = {0., 0., 0., 0.};
    parthenon::Real TimeContravariantVector[4] = {0., 0., 0., 0.};
    parthenon::Real TimeCovariantVector[4] = {0., 0., 0., 0.};

    DirectedCovariantVector[Direction] = 1.0;
    TimeCovariantVector[0] = 1.0;

    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            DirectedContravariantVector[row] += gcon[row][col] * DirectedCovariantVector[col];
            TimeContravariantVector[row] += gcon[row][col] * TimeCovariantVector[col];
        }
    }

    State StateGRMHD;
    CalculateGRMHDState(Primitive, gcov, gcon, StateGRMHD);

    const auto SquaredMagneticFieldFourVector =
        dot4(StateGRMHD.ContravariantMagneticField, StateGRMHD.CovariantMagneticField);
    const auto FluidInertia = Primitive[DensityIndex] + AdiabaticIndex * Primitive[EnergyIndex];
    const auto TotalInertia = SquaredMagneticFieldFourVector + FluidInertia;
    const auto SquaredAlfvenVelocity = SquaredMagneticFieldFourVector / TotalInertia;
    const auto SquaredSoundSpeed = AdiabaticIndex * (AdiabaticIndex - 1.) * Primitive[EnergyIndex] / FluidInertia;
    auto SpeedOfFastMagnetosonicWave = SquaredSoundSpeed + SquaredAlfvenVelocity - SquaredSoundSpeed * SquaredAlfvenVelocity;

    if (SpeedOfFastMagnetosonicWave < 0.) {
        SpeedOfFastMagnetosonicWave = 1e-10;
    }
    if (SpeedOfFastMagnetosonicWave > 1.) {
        SpeedOfFastMagnetosonicWave = 1.;
    }

    const auto SquaredDirectedContravariantVector = dot4(DirectedContravariantVector, DirectedCovariantVector);
    const auto SquaredTimeContravariantVector = dot4(TimeContravariantVector, TimeCovariantVector);
    const auto DirectedContravariantVectorDotFluidContravariantVelocity = dot4(DirectedCovariantVector, StateGRMHD.ContravariantVelocity);
    const auto TimeCovariantVectorDotFluidContravariantVelocity = dot4(TimeCovariantVector, StateGRMHD.ContravariantVelocity);
    const auto DirectedContravariantVectorDotTimeCovariantVector = dot4(DirectedContravariantVector, TimeCovariantVector);

    const auto SquaredDirectedContravariantVectorDotFluidContravariantVelocity =
        DirectedContravariantVectorDotFluidContravariantVelocity * DirectedContravariantVectorDotFluidContravariantVelocity;
    const auto SquaredTimeCovariantVectorDotFluidContravariantVelocity =
        TimeCovariantVectorDotFluidContravariantVelocity * TimeCovariantVectorDotFluidContravariantVelocity;
    const auto DirectedContravariantVectorDotFluidContravariantVelocityMulTimeCovariantVectorDotFluidContravariantVelocity =
        DirectedContravariantVectorDotFluidContravariantVelocity * TimeCovariantVectorDotFluidContravariantVelocity;

    const auto CoefficientOfQuadraticItem =
        SquaredTimeCovariantVectorDotFluidContravariantVelocity -
        (SquaredTimeContravariantVector + SquaredTimeCovariantVectorDotFluidContravariantVelocity) * SpeedOfFastMagnetosonicWave;
    const auto CoefficientOfLinearItem =
        2. * (DirectedContravariantVectorDotFluidContravariantVelocityMulTimeCovariantVectorDotFluidContravariantVelocity -
              (DirectedContravariantVectorDotTimeCovariantVector + DirectedContravariantVectorDotFluidContravariantVelocityMulTimeCovariantVectorDotFluidContravariantVelocity) *
                  SpeedOfFastMagnetosonicWave);
    const auto ConstantItem =
        SquaredDirectedContravariantVectorDotFluidContravariantVelocity -
        (SquaredDirectedContravariantVector + SquaredDirectedContravariantVectorDotFluidContravariantVelocity) * SpeedOfFastMagnetosonicWave;

    auto Discriminant = CoefficientOfLinearItem * CoefficientOfLinearItem -
                        4. * CoefficientOfQuadraticItem * ConstantItem;
    if ((Discriminant < 0.0) && (Discriminant > -1.e-10)) {
        Discriminant = 0.0;
    }
    else if (Discriminant < -1.e-10) {
        Discriminant = 0.;
    }

    Discriminant = Kokkos::sqrt(Discriminant);
    const auto VelocityWithPlus = -(-CoefficientOfLinearItem + Discriminant) / (2. * CoefficientOfQuadraticItem);
    const auto VelocityWithMinus = -(-CoefficientOfLinearItem - Discriminant) / (2. * CoefficientOfQuadraticItem);

    if (VelocityWithPlus > VelocityWithMinus) {
        greaterVelocity = VelocityWithPlus;
        lessVelocity = VelocityWithMinus;
    }
    else {
        greaterVelocity = VelocityWithMinus;
        lessVelocity = VelocityWithPlus;
    }
}
