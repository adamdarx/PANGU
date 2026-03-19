#pragma once
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <basic_types.hpp>
#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"
#include "../physics/AlfvenVelocity.hpp"
#include "../physics/EnergyMomentumTensor.hpp"
#include "../reconstruct/InterpolaterMC.hpp"

parthenon::TaskStatus CalculateFluxes(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    using namespace parthenon;
    PARTHENON_INSTRUMENT

    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto Package = MeshblockPointer->packages.Get("PANGU");
    const auto &AdiabaticIndex = Package->Param<Real>("AdiabaticIndex");

    const auto BoundX1 = MeshblockPointer->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto BoundX2 = MeshblockPointer->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto BoundX3 = MeshblockPointer->cellbounds.GetBoundsK(IndexDomain::interior);

    PackIndexMap primitiveIndexMap;
    const std::vector<std::string> PrimitiveTags = {"Density", "Energy", "WeightedVelocity", "MagneticField"};
    const auto Primitive = resource->PackVariables(PrimitiveTags, primitiveIndexMap);
    PackIndexMap conservativeIndexMap;
    const std::vector<std::string> ConservativeTags = {"Conservative"};
    auto conservative = resource->PackVariablesAndFluxes(ConservativeTags, conservativeIndexMap);
    
    const int ScratchLevel = 1;
    const auto MeshgridSizeX1 = MeshblockPointer->cellbounds.ncellsi(IndexDomain::entire);
    const auto MeshgridSizeX2 = MeshblockPointer->cellbounds.ncellsj(IndexDomain::entire);
    const auto MeshgridSizeX3 = MeshblockPointer->cellbounds.ncellsk(IndexDomain::entire);
    
    const size_t ScratchSizeInBytesX1 = ScratchPad2D<Real>::shmem_size(PrimitiveVariableNumber, MeshgridSizeX1);
    const size_t ScratchSizeInBytesX2 = ScratchPad2D<Real>::shmem_size(PrimitiveVariableNumber, MeshgridSizeX2);
    const size_t ScratchSizeInBytesX3 = ScratchPad2D<Real>::shmem_size(PrimitiveVariableNumber, MeshgridSizeX3);

    const int OffsetX1 = (MeshgridSizeX1 > 1) ? 1 : 0;
    const int OffsetX2 = (MeshgridSizeX2 > 1) ? 1 : 0;
    const int OffsetX3 = (MeshgridSizeX3 > 1) ? 1 : 0;

    MeshblockPointer->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX1, ScratchLevel, BoundX3.s - OffsetX3, BoundX3.e + OffsetX3, BoundX2.s - OffsetX2, BoundX2.e + OffsetX2,
        KOKKOS_LAMBDA(team_mbr_t member, const int k, const int j) {
            ScratchPad2D<Real> primitiveLeft(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX1);
            ScratchPad2D<Real> primitiveRight(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX1);

            par_for_inner(member, 0, PrimitiveVariableNumber - 1, BoundX1.s, BoundX1.e + 1, [&](const int n, const int i) {
                primitiveLeft(n, i) = Primitive(n, k, j, i - 1) + 0.5 * InterpolateMC(Primitive(n, k, j, i - 2), Primitive(n, k, j, i - 1), Primitive(n, k, j, i));
                primitiveRight(n, i) = Primitive(n, k, j, i) - 0.5 * InterpolateMC(Primitive(n, k, j, i - 1), Primitive(n, k, j, i), Primitive(n, k, j, i + 1));
            });
            
            member.team_barrier();

            par_for_inner(member, BoundX1.s, BoundX1.e + 1, [&](const int i) {
                Real directedEnergyMomentumTensorLeft[4];
                Real directedEnergyMomentumTensorRight[4];
                Real conservativeLeft[PrimitiveVariableNumber];
                Real conservativeRight[PrimitiveVariableNumber];
                Real fluxLeft[PrimitiveVariableNumber];
                Real fluxRight[PrimitiveVariableNumber];
                const Real PrimitiveLeftCArray[PrimitiveVariableNumber] = {
                    primitiveLeft(DensityIndex, i),
                    primitiveLeft(EnergyIndex, i),
                    primitiveLeft(WeightedVelocityX1, i),
                    primitiveLeft(WeightedVelocityX2, i),
                    primitiveLeft(WeightedVelocityX3, i),
                    primitiveLeft(MagneticFieldX1, i),
                    primitiveLeft(MagneticFieldX2, i),
                    primitiveLeft(MagneticFieldX3, i),
                };
                const Real PrimitiveRightCArray[PrimitiveVariableNumber] = {
                    primitiveRight(DensityIndex, i),
                    primitiveRight(EnergyIndex, i),
                    primitiveRight(WeightedVelocityX1, i),
                    primitiveRight(WeightedVelocityX2, i),
                    primitiveRight(WeightedVelocityX3, i),
                    primitiveRight(MagneticFieldX1, i),
                    primitiveRight(MagneticFieldX2, i),
                    primitiveRight(MagneticFieldX3, i),
                };

                const auto SquaredWeightedVelocityLeft = Kokkos::pow(PrimitiveLeftCArray[WeightedVelocityX1], 2) + Kokkos::pow(PrimitiveLeftCArray[WeightedVelocityX2], 2) + Kokkos::pow(PrimitiveLeftCArray[WeightedVelocityX3], 2);
                const auto SquaredLorentzFactorLeft = 1 + SquaredWeightedVelocityLeft;
                const auto LorentzFactorLeft = Kokkos::sqrt(SquaredLorentzFactorLeft);
                const auto MagneticFieldThreeVectorDotWeightedVelocityLeft = PrimitiveLeftCArray[WeightedVelocityX1] * PrimitiveLeftCArray[MagneticFieldX1] + PrimitiveLeftCArray[WeightedVelocityX2] * PrimitiveLeftCArray[MagneticFieldX2] + PrimitiveLeftCArray[WeightedVelocityX3] * PrimitiveLeftCArray[MagneticFieldX3];

                const auto SquaredWeightedVelocityRight = Kokkos::pow(PrimitiveRightCArray[WeightedVelocityX1], 2) + Kokkos::pow(PrimitiveRightCArray[WeightedVelocityX2], 2) + Kokkos::pow(PrimitiveRightCArray[WeightedVelocityX3], 2);
                const auto SquaredLorentzFactorRight = 1 + SquaredWeightedVelocityRight;
                const auto LorentzFactorRight = Kokkos::sqrt(SquaredLorentzFactorRight);
                const auto MagneticFieldThreeVectorDotWeightedVelocityRight = PrimitiveRightCArray[WeightedVelocityX1] * PrimitiveRightCArray[MagneticFieldX1] + PrimitiveRightCArray[WeightedVelocityX2] * PrimitiveRightCArray[MagneticFieldX2] + PrimitiveRightCArray[WeightedVelocityX3] * PrimitiveRightCArray[MagneticFieldX3];

                Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
                Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
                CalculateAlfvenVelocity(AdiabaticIndex, PrimitiveLeftCArray, X1DIR, maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
                CalculateAlfvenVelocity(AdiabaticIndex, PrimitiveRightCArray, X1DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
                const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft), maximumAlfvenVelocityRight));
                const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft), -minimumAlfvenVelocityRight));
                const auto AlfvenVelocityCenter = Kokkos::max(MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);

                CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveLeftCArray, X0DIR, directedEnergyMomentumTensorLeft);
                conservativeLeft[DensityIndex] = PrimitiveLeftCArray[DensityIndex] * LorentzFactorLeft;
                conservativeLeft[EnergyIndex] = directedEnergyMomentumTensorLeft[0] + conservativeLeft[DensityIndex];
                conservativeLeft[WeightedVelocityX1] = directedEnergyMomentumTensorLeft[1];
                conservativeLeft[WeightedVelocityX2] = directedEnergyMomentumTensorLeft[2];
                conservativeLeft[WeightedVelocityX3] = directedEnergyMomentumTensorLeft[3];
                conservativeLeft[MagneticFieldX1] = PrimitiveLeftCArray[MagneticFieldX1];
                conservativeLeft[MagneticFieldX2] = PrimitiveLeftCArray[MagneticFieldX2];
                conservativeLeft[MagneticFieldX3] = PrimitiveLeftCArray[MagneticFieldX3];

                CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveLeftCArray, X1DIR, directedEnergyMomentumTensorLeft);
                fluxLeft[DensityIndex] = PrimitiveLeftCArray[DensityIndex] * PrimitiveLeftCArray[WeightedVelocityX1];
                fluxLeft[EnergyIndex] = directedEnergyMomentumTensorLeft[0] + fluxLeft[DensityIndex];
                fluxLeft[WeightedVelocityX1] = directedEnergyMomentumTensorLeft[1];
                fluxLeft[WeightedVelocityX2] = directedEnergyMomentumTensorLeft[2];
                fluxLeft[WeightedVelocityX3] = directedEnergyMomentumTensorLeft[3];
                fluxLeft[MagneticFieldX1] = 0;
                fluxLeft[MagneticFieldX2] = ((PrimitiveLeftCArray[MagneticFieldX2] + MagneticFieldThreeVectorDotWeightedVelocityLeft * PrimitiveLeftCArray[WeightedVelocityX2]) * PrimitiveLeftCArray[WeightedVelocityX1] - (PrimitiveLeftCArray[MagneticFieldX1] + MagneticFieldThreeVectorDotWeightedVelocityLeft * PrimitiveLeftCArray[WeightedVelocityX1]) * PrimitiveLeftCArray[WeightedVelocityX2]) / LorentzFactorLeft;
                fluxLeft[MagneticFieldX3] = ((PrimitiveLeftCArray[MagneticFieldX3] + MagneticFieldThreeVectorDotWeightedVelocityLeft * PrimitiveLeftCArray[WeightedVelocityX3]) * PrimitiveLeftCArray[WeightedVelocityX1] - (PrimitiveLeftCArray[MagneticFieldX1] + MagneticFieldThreeVectorDotWeightedVelocityLeft * PrimitiveLeftCArray[WeightedVelocityX1]) * PrimitiveLeftCArray[WeightedVelocityX3]) / LorentzFactorLeft;

                CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveRightCArray, X0DIR, directedEnergyMomentumTensorRight);
                conservativeRight[DensityIndex] = PrimitiveRightCArray[DensityIndex] * LorentzFactorRight;
                conservativeRight[EnergyIndex] = directedEnergyMomentumTensorRight[0] + conservativeRight[DensityIndex];
                conservativeRight[WeightedVelocityX1] = directedEnergyMomentumTensorRight[1];
                conservativeRight[WeightedVelocityX2] = directedEnergyMomentumTensorRight[2];
                conservativeRight[WeightedVelocityX3] = directedEnergyMomentumTensorRight[3];
                conservativeRight[MagneticFieldX1] = PrimitiveRightCArray[MagneticFieldX1];
                conservativeRight[MagneticFieldX2] = PrimitiveRightCArray[MagneticFieldX2];
                conservativeRight[MagneticFieldX3] = PrimitiveRightCArray[MagneticFieldX3];

                CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveRightCArray, X1DIR, directedEnergyMomentumTensorRight);
                fluxRight[DensityIndex] = PrimitiveRightCArray[DensityIndex] * PrimitiveRightCArray[WeightedVelocityX1];
                fluxRight[EnergyIndex] = directedEnergyMomentumTensorRight[0] + fluxRight[DensityIndex];
                fluxRight[WeightedVelocityX1] = directedEnergyMomentumTensorRight[1];
                fluxRight[WeightedVelocityX2] = directedEnergyMomentumTensorRight[2];
                fluxRight[WeightedVelocityX3] = directedEnergyMomentumTensorRight[3];
                fluxRight[MagneticFieldX1] = 0;
                fluxRight[MagneticFieldX2] = ((PrimitiveRightCArray[MagneticFieldX2] + MagneticFieldThreeVectorDotWeightedVelocityRight * PrimitiveRightCArray[WeightedVelocityX2]) * PrimitiveRightCArray[WeightedVelocityX1] - (PrimitiveRightCArray[MagneticFieldX1] + MagneticFieldThreeVectorDotWeightedVelocityRight * PrimitiveRightCArray[WeightedVelocityX1]) * PrimitiveRightCArray[WeightedVelocityX2]) / LorentzFactorRight;
                fluxRight[MagneticFieldX3] = ((PrimitiveRightCArray[MagneticFieldX3] + MagneticFieldThreeVectorDotWeightedVelocityRight * PrimitiveRightCArray[WeightedVelocityX3]) * PrimitiveRightCArray[WeightedVelocityX1] - (PrimitiveRightCArray[MagneticFieldX1] + MagneticFieldThreeVectorDotWeightedVelocityRight * PrimitiveRightCArray[WeightedVelocityX1]) * PrimitiveRightCArray[WeightedVelocityX3]) / LorentzFactorRight;
                
                for(int index = 0; index < PrimitiveVariableNumber; index++)
                    conservative.flux(X1DIR, index, k, j, i) = 0.5 * (fluxLeft[index] + fluxRight[index] - AlfvenVelocityCenter * (conservativeRight[index] - conservativeLeft[index]));
            });
        });

    if (MeshblockPointer->pmy_mesh->ndim >= 2)
    MeshblockPointer->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX2, ScratchLevel, BoundX3.s - OffsetX3, BoundX3.e + OffsetX3, BoundX1.s - OffsetX1, BoundX1.e + OffsetX1,
        KOKKOS_LAMBDA(team_mbr_t member, const int k, const int i) {
            ScratchPad2D<Real> primitiveLeft(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX2);
            ScratchPad2D<Real> primitiveRight(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX2);

            par_for_inner(member, 0, PrimitiveVariableNumber - 1, BoundX2.s, BoundX2.e + 1, [&](const int n, const int j) {
                primitiveLeft(n, j) = Primitive(n, k, j - 1, i) + 0.5 * InterpolateMC(Primitive(n, k, j - 2, i), Primitive(n, k, j - 1, i), Primitive(n, k, j, i));
                primitiveRight(n, j) = Primitive(n, k, j, i) - 0.5 * InterpolateMC(Primitive(n, k, j - 1, i), Primitive(n, k, j, i), Primitive(n, k, j + 1, i));
            });
            
            member.team_barrier();

            par_for_inner(member, BoundX2.s, BoundX2.e + 1, [&](const int j) {
                Real directedEnergyMomentumTensorLeft[4];
                Real directedEnergyMomentumTensorRight[4];
                Real conservativeLeft[PrimitiveVariableNumber];
                Real conservativeRight[PrimitiveVariableNumber];
                Real fluxLeft[PrimitiveVariableNumber];
                Real fluxRight[PrimitiveVariableNumber];
                const Real PrimitiveLeftCArray[PrimitiveVariableNumber] = {
                    primitiveLeft(DensityIndex, j),
                    primitiveLeft(EnergyIndex, j),
                    primitiveLeft(WeightedVelocityX1, j),
                    primitiveLeft(WeightedVelocityX2, j),
                    primitiveLeft(WeightedVelocityX3, j),
                    primitiveLeft(MagneticFieldX1, j),
                    primitiveLeft(MagneticFieldX2, j),
                    primitiveLeft(MagneticFieldX3, j),
                };
                const Real PrimitiveRightCArray[PrimitiveVariableNumber] = {
                    primitiveRight(DensityIndex, j),
                    primitiveRight(EnergyIndex, j),
                    primitiveRight(WeightedVelocityX1, j),
                    primitiveRight(WeightedVelocityX2, j),
                    primitiveRight(WeightedVelocityX3, j),
                    primitiveRight(MagneticFieldX1, j),
                    primitiveRight(MagneticFieldX2, j),
                    primitiveRight(MagneticFieldX3, j),
                };

                const auto SquaredWeightedVelocityLeft = Kokkos::pow(PrimitiveLeftCArray[WeightedVelocityX1], 2) + Kokkos::pow(PrimitiveLeftCArray[WeightedVelocityX2], 2) + Kokkos::pow(PrimitiveLeftCArray[WeightedVelocityX3], 2);
                const auto SquaredLorentzFactorLeft = 1 + SquaredWeightedVelocityLeft;
                const auto LorentzFactorLeft = Kokkos::sqrt(SquaredLorentzFactorLeft);
                const auto MagneticFieldThreeVectorDotWeightedVelocityLeft = PrimitiveLeftCArray[WeightedVelocityX1] * PrimitiveLeftCArray[MagneticFieldX1] + PrimitiveLeftCArray[WeightedVelocityX2] * PrimitiveLeftCArray[MagneticFieldX2] + PrimitiveLeftCArray[WeightedVelocityX3] * PrimitiveLeftCArray[MagneticFieldX3];

                const auto SquaredWeightedVelocityRight = Kokkos::pow(PrimitiveRightCArray[WeightedVelocityX1], 2) + Kokkos::pow(PrimitiveRightCArray[WeightedVelocityX2], 2) + Kokkos::pow(PrimitiveRightCArray[WeightedVelocityX3], 2);
                const auto SquaredLorentzFactorRight = 1 + SquaredWeightedVelocityRight;
                const auto LorentzFactorRight = Kokkos::sqrt(SquaredLorentzFactorRight);
                const auto MagneticFieldThreeVectorDotWeightedVelocityRight = PrimitiveRightCArray[WeightedVelocityX1] * PrimitiveRightCArray[MagneticFieldX1] + PrimitiveRightCArray[WeightedVelocityX2] * PrimitiveRightCArray[MagneticFieldX2] + PrimitiveRightCArray[WeightedVelocityX3] * PrimitiveRightCArray[MagneticFieldX3];

                Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
                Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
                CalculateAlfvenVelocity(AdiabaticIndex, PrimitiveLeftCArray, X2DIR, maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
                CalculateAlfvenVelocity(AdiabaticIndex, PrimitiveRightCArray, X2DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
                const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft), maximumAlfvenVelocityRight));
                const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft), -minimumAlfvenVelocityRight));
                const auto AlfvenVelocityCenter = Kokkos::max(MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);

                CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveLeftCArray, X0DIR, directedEnergyMomentumTensorLeft);
                conservativeLeft[DensityIndex] = PrimitiveLeftCArray[DensityIndex] * LorentzFactorLeft;
                conservativeLeft[EnergyIndex] = directedEnergyMomentumTensorLeft[0] + conservativeLeft[DensityIndex];
                conservativeLeft[WeightedVelocityX1] = directedEnergyMomentumTensorLeft[1];
                conservativeLeft[WeightedVelocityX2] = directedEnergyMomentumTensorLeft[2];
                conservativeLeft[WeightedVelocityX3] = directedEnergyMomentumTensorLeft[3];
                conservativeLeft[MagneticFieldX1] = PrimitiveLeftCArray[MagneticFieldX1];
                conservativeLeft[MagneticFieldX2] = PrimitiveLeftCArray[MagneticFieldX2];
                conservativeLeft[MagneticFieldX3] = PrimitiveLeftCArray[MagneticFieldX3];

                CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveLeftCArray, X2DIR, directedEnergyMomentumTensorLeft);
                fluxLeft[DensityIndex] = PrimitiveLeftCArray[DensityIndex] * PrimitiveLeftCArray[WeightedVelocityX2];
                fluxLeft[EnergyIndex] = directedEnergyMomentumTensorLeft[0] + fluxLeft[DensityIndex];
                fluxLeft[WeightedVelocityX1] = directedEnergyMomentumTensorLeft[1];
                fluxLeft[WeightedVelocityX2] = directedEnergyMomentumTensorLeft[2];
                fluxLeft[WeightedVelocityX3] = directedEnergyMomentumTensorLeft[3];
                fluxLeft[MagneticFieldX1] = ((PrimitiveLeftCArray[MagneticFieldX1] + MagneticFieldThreeVectorDotWeightedVelocityLeft * PrimitiveLeftCArray[WeightedVelocityX1]) * PrimitiveLeftCArray[WeightedVelocityX2] - (PrimitiveLeftCArray[MagneticFieldX2] + MagneticFieldThreeVectorDotWeightedVelocityLeft * PrimitiveLeftCArray[WeightedVelocityX2]) * PrimitiveLeftCArray[WeightedVelocityX1]) / LorentzFactorLeft;
                fluxLeft[MagneticFieldX2] = 0;
                fluxLeft[MagneticFieldX3] = ((PrimitiveLeftCArray[MagneticFieldX3] + MagneticFieldThreeVectorDotWeightedVelocityLeft * PrimitiveLeftCArray[WeightedVelocityX3]) * PrimitiveLeftCArray[WeightedVelocityX2] - (PrimitiveLeftCArray[MagneticFieldX2] + MagneticFieldThreeVectorDotWeightedVelocityLeft * PrimitiveLeftCArray[WeightedVelocityX2]) * PrimitiveLeftCArray[WeightedVelocityX3]) / LorentzFactorLeft;

                CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveRightCArray, X0DIR, directedEnergyMomentumTensorRight);
                conservativeRight[DensityIndex] = PrimitiveRightCArray[DensityIndex] * LorentzFactorRight;
                conservativeRight[EnergyIndex] = directedEnergyMomentumTensorRight[0] + conservativeRight[DensityIndex];
                conservativeRight[WeightedVelocityX1] = directedEnergyMomentumTensorRight[1];
                conservativeRight[WeightedVelocityX2] = directedEnergyMomentumTensorRight[2];
                conservativeRight[WeightedVelocityX3] = directedEnergyMomentumTensorRight[3];
                conservativeRight[MagneticFieldX1] = PrimitiveRightCArray[MagneticFieldX1];
                conservativeRight[MagneticFieldX2] = PrimitiveRightCArray[MagneticFieldX2];
                conservativeRight[MagneticFieldX3] = PrimitiveRightCArray[MagneticFieldX3];

                CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveRightCArray, X2DIR, directedEnergyMomentumTensorRight);
                fluxRight[DensityIndex] = PrimitiveRightCArray[DensityIndex] * PrimitiveRightCArray[WeightedVelocityX2];
                fluxRight[EnergyIndex] = directedEnergyMomentumTensorRight[0] + fluxRight[DensityIndex];
                fluxRight[WeightedVelocityX1] = directedEnergyMomentumTensorRight[1];
                fluxRight[WeightedVelocityX2] = directedEnergyMomentumTensorRight[2];
                fluxRight[WeightedVelocityX3] = directedEnergyMomentumTensorRight[3];
                fluxRight[MagneticFieldX1] = ((PrimitiveRightCArray[MagneticFieldX1] + MagneticFieldThreeVectorDotWeightedVelocityRight * PrimitiveRightCArray[WeightedVelocityX1]) * PrimitiveRightCArray[WeightedVelocityX2] - (PrimitiveRightCArray[MagneticFieldX2] + MagneticFieldThreeVectorDotWeightedVelocityRight * PrimitiveRightCArray[WeightedVelocityX2]) * PrimitiveRightCArray[WeightedVelocityX1]) / LorentzFactorRight;
                fluxRight[MagneticFieldX2] = 0;
                fluxRight[MagneticFieldX3] = ((PrimitiveRightCArray[MagneticFieldX3] + MagneticFieldThreeVectorDotWeightedVelocityRight * PrimitiveRightCArray[WeightedVelocityX3]) * PrimitiveRightCArray[WeightedVelocityX2] - (PrimitiveRightCArray[MagneticFieldX2] + MagneticFieldThreeVectorDotWeightedVelocityRight * PrimitiveRightCArray[WeightedVelocityX2]) * PrimitiveRightCArray[WeightedVelocityX3]) / LorentzFactorRight;

                for(int index = 0; index < PrimitiveVariableNumber; index++)
                    conservative.flux(X2DIR, index, k, j, i) = 0.5 * (fluxLeft[index] + fluxRight[index] - AlfvenVelocityCenter * (conservativeRight[index] - conservativeLeft[index]));
            });
        });
    
    if (MeshblockPointer->pmy_mesh->ndim == 3)
    MeshblockPointer->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX3, ScratchLevel, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(team_mbr_t member, const int j, const int i) {
            ScratchPad2D<Real> primitiveLeft(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX3);
            ScratchPad2D<Real> primitiveRight(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX3);

            par_for_inner(member, 0, PrimitiveVariableNumber - 1, BoundX3.s, BoundX3.e + 1, [&](const int n, const int k) {
                primitiveLeft(n, k) = Primitive(n, k - 1, j, i) + 0.5 * InterpolateMC(Primitive(n, k - 2, j, i), Primitive(n, k - 1, j, i), Primitive(n, k, j, i));
                primitiveRight(n, k) = Primitive(n, k, j, i) - 0.5 * InterpolateMC(Primitive(n, k - 1, j, i), Primitive(n, k, j, i), Primitive(n, k + 1, j, i));
            });
            
            member.team_barrier();

            par_for_inner(member, BoundX3.s, BoundX3.e + 1, [&](const int k) {
                Real directedEnergyMomentumTensorLeft[4];
                Real directedEnergyMomentumTensorRight[4];
                Real conservativeLeft[PrimitiveVariableNumber];
                Real conservativeRight[PrimitiveVariableNumber];
                Real fluxLeft[PrimitiveVariableNumber];
                Real fluxRight[PrimitiveVariableNumber];
                const Real PrimitiveLeftCArray[PrimitiveVariableNumber] = {
                    primitiveLeft(DensityIndex, k),
                    primitiveLeft(EnergyIndex, k),
                    primitiveLeft(WeightedVelocityX1, k),
                    primitiveLeft(WeightedVelocityX2, k),
                    primitiveLeft(WeightedVelocityX3, k),
                    primitiveLeft(MagneticFieldX1, k),
                    primitiveLeft(MagneticFieldX2, k),
                    primitiveLeft(MagneticFieldX3, k),
                };
                const Real PrimitiveRightCArray[PrimitiveVariableNumber] = {
                    primitiveRight(DensityIndex, k),
                    primitiveRight(EnergyIndex, k),
                    primitiveRight(WeightedVelocityX1, k),
                    primitiveRight(WeightedVelocityX2, k),
                    primitiveRight(WeightedVelocityX3, k),
                    primitiveRight(MagneticFieldX1, k),
                    primitiveRight(MagneticFieldX2, k),
                    primitiveRight(MagneticFieldX3, k),
                };

                const auto SquaredWeightedVelocityLeft = Kokkos::pow(PrimitiveLeftCArray[WeightedVelocityX1], 2) + Kokkos::pow(PrimitiveLeftCArray[WeightedVelocityX2], 2) + Kokkos::pow(PrimitiveLeftCArray[WeightedVelocityX3], 2);
                const auto SquaredLorentzFactorLeft = 1 + SquaredWeightedVelocityLeft;
                const auto LorentzFactorLeft = Kokkos::sqrt(SquaredLorentzFactorLeft);
                const auto MagneticFieldThreeVectorDotWeightedVelocityLeft = PrimitiveLeftCArray[WeightedVelocityX1] * PrimitiveLeftCArray[MagneticFieldX1] + PrimitiveLeftCArray[WeightedVelocityX2] * PrimitiveLeftCArray[MagneticFieldX2] + PrimitiveLeftCArray[WeightedVelocityX3] * PrimitiveLeftCArray[MagneticFieldX3];

                const auto SquaredWeightedVelocityRight = Kokkos::pow(PrimitiveRightCArray[WeightedVelocityX1], 2) + Kokkos::pow(PrimitiveRightCArray[WeightedVelocityX2], 2) + Kokkos::pow(PrimitiveRightCArray[WeightedVelocityX3], 2);
                const auto SquaredLorentzFactorRight = 1 + SquaredWeightedVelocityRight;
                const auto LorentzFactorRight = Kokkos::sqrt(SquaredLorentzFactorRight);
                const auto MagneticFieldThreeVectorDotWeightedVelocityRight = PrimitiveRightCArray[WeightedVelocityX1] * PrimitiveRightCArray[MagneticFieldX1] + PrimitiveRightCArray[WeightedVelocityX2] * PrimitiveRightCArray[MagneticFieldX2] + PrimitiveRightCArray[WeightedVelocityX3] * PrimitiveRightCArray[MagneticFieldX3];

                Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
                Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
                CalculateAlfvenVelocity(AdiabaticIndex, PrimitiveLeftCArray, X3DIR, maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
                CalculateAlfvenVelocity(AdiabaticIndex, PrimitiveRightCArray, X3DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
                const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft), maximumAlfvenVelocityRight));
                const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft), -minimumAlfvenVelocityRight));
                const auto AlfvenVelocityCenter = Kokkos::max(MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);

                CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveLeftCArray, X0DIR, directedEnergyMomentumTensorLeft);
                conservativeLeft[DensityIndex] = PrimitiveLeftCArray[DensityIndex] * LorentzFactorLeft;
                conservativeLeft[EnergyIndex] = directedEnergyMomentumTensorLeft[0] + conservativeLeft[DensityIndex];
                conservativeLeft[WeightedVelocityX1] = directedEnergyMomentumTensorLeft[1];
                conservativeLeft[WeightedVelocityX2] = directedEnergyMomentumTensorLeft[2];
                conservativeLeft[WeightedVelocityX3] = directedEnergyMomentumTensorLeft[3];
                conservativeLeft[MagneticFieldX1] = PrimitiveLeftCArray[MagneticFieldX1];
                conservativeLeft[MagneticFieldX2] = PrimitiveLeftCArray[MagneticFieldX2];
                conservativeLeft[MagneticFieldX3] = PrimitiveLeftCArray[MagneticFieldX3];

                CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveLeftCArray, X3DIR, directedEnergyMomentumTensorLeft);
                fluxLeft[DensityIndex] = PrimitiveLeftCArray[DensityIndex] * PrimitiveLeftCArray[WeightedVelocityX3];
                fluxLeft[EnergyIndex] = directedEnergyMomentumTensorLeft[0] + fluxLeft[DensityIndex];
                fluxLeft[WeightedVelocityX1] = directedEnergyMomentumTensorLeft[1];
                fluxLeft[WeightedVelocityX2] = directedEnergyMomentumTensorLeft[2];
                fluxLeft[WeightedVelocityX3] = directedEnergyMomentumTensorLeft[3];
                fluxLeft[MagneticFieldX1] = ((PrimitiveLeftCArray[MagneticFieldX1] + MagneticFieldThreeVectorDotWeightedVelocityLeft * PrimitiveLeftCArray[WeightedVelocityX1]) * PrimitiveLeftCArray[WeightedVelocityX3] - (PrimitiveLeftCArray[MagneticFieldX3] + MagneticFieldThreeVectorDotWeightedVelocityLeft * PrimitiveLeftCArray[WeightedVelocityX3]) * PrimitiveLeftCArray[WeightedVelocityX1]) / LorentzFactorLeft;
                fluxLeft[MagneticFieldX2] = ((PrimitiveLeftCArray[MagneticFieldX2] + MagneticFieldThreeVectorDotWeightedVelocityLeft * PrimitiveLeftCArray[WeightedVelocityX2]) * PrimitiveLeftCArray[WeightedVelocityX3] - (PrimitiveLeftCArray[MagneticFieldX3] + MagneticFieldThreeVectorDotWeightedVelocityLeft * PrimitiveLeftCArray[WeightedVelocityX3]) * PrimitiveLeftCArray[WeightedVelocityX2]) / LorentzFactorLeft;
                fluxLeft[MagneticFieldX3] = 0;

                CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveRightCArray, X0DIR, directedEnergyMomentumTensorRight);
                conservativeRight[DensityIndex] = PrimitiveRightCArray[DensityIndex] * LorentzFactorRight;
                conservativeRight[EnergyIndex] = directedEnergyMomentumTensorRight[0] + conservativeRight[DensityIndex];
                conservativeRight[WeightedVelocityX1] = directedEnergyMomentumTensorRight[1];
                conservativeRight[WeightedVelocityX2] = directedEnergyMomentumTensorRight[2];
                conservativeRight[WeightedVelocityX3] = directedEnergyMomentumTensorRight[3];
                conservativeRight[MagneticFieldX1] = PrimitiveRightCArray[MagneticFieldX1];
                conservativeRight[MagneticFieldX2] = PrimitiveRightCArray[MagneticFieldX2];
                conservativeRight[MagneticFieldX3] = PrimitiveRightCArray[MagneticFieldX3];

                CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveRightCArray, X3DIR, directedEnergyMomentumTensorRight);
                fluxRight[DensityIndex] = PrimitiveRightCArray[DensityIndex] * PrimitiveRightCArray[WeightedVelocityX3];
                fluxRight[EnergyIndex] = directedEnergyMomentumTensorRight[0] + fluxRight[DensityIndex];
                fluxRight[WeightedVelocityX1] = directedEnergyMomentumTensorRight[1];
                fluxRight[WeightedVelocityX2] = directedEnergyMomentumTensorRight[2];
                fluxRight[WeightedVelocityX3] = directedEnergyMomentumTensorRight[3];
                fluxRight[MagneticFieldX1] = ((PrimitiveRightCArray[MagneticFieldX1] + MagneticFieldThreeVectorDotWeightedVelocityRight * PrimitiveRightCArray[WeightedVelocityX1]) * PrimitiveRightCArray[WeightedVelocityX3] - (PrimitiveRightCArray[MagneticFieldX3] + MagneticFieldThreeVectorDotWeightedVelocityRight * PrimitiveRightCArray[WeightedVelocityX3]) * PrimitiveRightCArray[WeightedVelocityX1]) / LorentzFactorRight;
                fluxRight[MagneticFieldX2] = ((PrimitiveRightCArray[MagneticFieldX2] + MagneticFieldThreeVectorDotWeightedVelocityRight * PrimitiveRightCArray[WeightedVelocityX2]) * PrimitiveRightCArray[WeightedVelocityX3] - (PrimitiveRightCArray[MagneticFieldX3] + MagneticFieldThreeVectorDotWeightedVelocityRight * PrimitiveRightCArray[WeightedVelocityX3]) * PrimitiveRightCArray[WeightedVelocityX2]) / LorentzFactorRight;
                fluxRight[MagneticFieldX3] = 0;

                for(int index = 0; index < PrimitiveVariableNumber; index++)
                    conservative.flux(X3DIR, index, k, j, i) = 0.5 * (fluxLeft[index] + fluxRight[index] - AlfvenVelocityCenter * (conservativeRight[index] - conservativeLeft[index]));
            });
        });

    return TaskStatus::complete;
}
