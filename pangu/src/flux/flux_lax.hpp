#pragma once
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <basic_types.hpp>
#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"
#include "../physics/AlfvenVelocity.hpp"
#include "../physics/flux.hpp"
#include "../reconstruct/InterpolaterMC.hpp"

parthenon::TaskStatus CalculateFluxes(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    using namespace parthenon;
    PARTHENON_INSTRUMENT

    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto PackageCORE = MeshblockPointer->packages.Get("CORE");
    const auto &AdiabaticIndex = PackageCORE->Param<Real>("AdiabaticIndex");

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

                Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
                Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
                CalculateAlfvenVelocitySRMHD(AdiabaticIndex, PrimitiveLeftCArray, X1DIR, maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
                CalculateAlfvenVelocitySRMHD(AdiabaticIndex, PrimitiveRightCArray, X1DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
                const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft), maximumAlfvenVelocityRight));
                const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft), -minimumAlfvenVelocityRight));
                const auto AlfvenVelocityCenter = Kokkos::max(MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);

                CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveLeftCArray, X0DIR,
                                                conservativeLeft);
                CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveRightCArray, X0DIR,
                                                conservativeRight);
                CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveLeftCArray, X1DIR,
                                                fluxLeft);
                CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveRightCArray, X1DIR,
                                                fluxRight);
                
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

                Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
                Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
                CalculateAlfvenVelocitySRMHD(AdiabaticIndex, PrimitiveLeftCArray, X2DIR, maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
                CalculateAlfvenVelocitySRMHD(AdiabaticIndex, PrimitiveRightCArray, X2DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
                const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft), maximumAlfvenVelocityRight));
                const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft), -minimumAlfvenVelocityRight));
                const auto AlfvenVelocityCenter = Kokkos::max(MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);

                CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveLeftCArray, X0DIR,
                                                conservativeLeft);
                CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveRightCArray, X0DIR,
                                                conservativeRight);
                CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveLeftCArray, X2DIR,
                                                fluxLeft);
                CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveRightCArray, X2DIR,
                                                fluxRight);

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

                Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
                Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
                CalculateAlfvenVelocitySRMHD(AdiabaticIndex, PrimitiveLeftCArray, X3DIR, maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
                CalculateAlfvenVelocitySRMHD(AdiabaticIndex, PrimitiveRightCArray, X3DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
                const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft), maximumAlfvenVelocityRight));
                const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft), -minimumAlfvenVelocityRight));
                const auto AlfvenVelocityCenter = Kokkos::max(MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);

                CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveLeftCArray, X0DIR,
                                                conservativeLeft);
                CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveRightCArray, X0DIR,
                                                conservativeRight);
                CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveLeftCArray, X3DIR,
                                                fluxLeft);
                CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveRightCArray, X3DIR,
                                                fluxRight);

                for(int index = 0; index < PrimitiveVariableNumber; index++)
                    conservative.flux(X3DIR, index, k, j, i) = 0.5 * (fluxLeft[index] + fluxRight[index] - AlfvenVelocityCenter * (conservativeRight[index] - conservativeLeft[index]));
            });
        });

    return TaskStatus::complete;
}

parthenon::TaskStatus CalculateFluxesGRMHD(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    using namespace parthenon;
    PARTHENON_INSTRUMENT

    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto PackageCORE = MeshblockPointer->packages.Get("CORE");
    const auto PackageMETRIC = MeshblockPointer->packages.Get("METRIC");
    const auto &AdiabaticIndex = PackageCORE->Param<Real>("AdiabaticIndex");
    (void)PackageMETRIC;

    const auto BoundX1 = MeshblockPointer->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto BoundX2 = MeshblockPointer->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto BoundX3 = MeshblockPointer->cellbounds.GetBoundsK(IndexDomain::interior);

    PackIndexMap primitiveIndexMap;
    const std::vector<std::string> PrimitiveTags = {"Density", "Energy", "WeightedVelocity", "MagneticField"};
    const auto Primitive = resource->PackVariables(PrimitiveTags, primitiveIndexMap);
    PackIndexMap covariantMetricIndexMap;
    const std::vector<std::string> CovariantMetricTags = {"CovariantMetric"};
    const auto CovariantMetric = resource->PackVariables(CovariantMetricTags, covariantMetricIndexMap);
    PackIndexMap contravariantMetricIndexMap;
    const std::vector<std::string> ContravariantMetricTags = {"ContravariantMetric"};
    const auto ContravariantMetric = resource->PackVariables(ContravariantMetricTags, contravariantMetricIndexMap);
    PackIndexMap metricDeterminantIndexMap;
    const std::vector<std::string> MetricDeterminantTags = {"MetricDeterminant"};
    const auto MetricDeterminant = resource->PackVariables(MetricDeterminantTags, metricDeterminantIndexMap);
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
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX1, ScratchLevel,
        BoundX3.s - OffsetX3, BoundX3.e + OffsetX3, BoundX2.s - OffsetX2, BoundX2.e + OffsetX2,
        KOKKOS_LAMBDA(team_mbr_t member, const int k, const int j) {
            ScratchPad2D<Real> primitiveLeft(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX1);
            ScratchPad2D<Real> primitiveRight(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX1);

            par_for_inner(member, 0, PrimitiveVariableNumber - 1, BoundX1.s, BoundX1.e + 1, [&](const int n, const int i) {
                primitiveLeft(n, i) = Primitive(n, k, j, i - 1) + 0.5 * InterpolateMC(Primitive(n, k, j, i - 2), Primitive(n, k, j, i - 1), Primitive(n, k, j, i));
                primitiveRight(n, i) = Primitive(n, k, j, i) - 0.5 * InterpolateMC(Primitive(n, k, j, i - 1), Primitive(n, k, j, i), Primitive(n, k, j, i + 1));
            });

            member.team_barrier();

            par_for_inner(member, BoundX1.s, BoundX1.e + 1, [&](const int i) {
                Real conservativeLeft[PrimitiveVariableNumber];
                Real conservativeRight[PrimitiveVariableNumber];
                Real fluxLeft[PrimitiveVariableNumber];
                Real fluxRight[PrimitiveVariableNumber];

                const Real PrimitiveLeftCArray[PrimitiveVariableNumber] = {
                    primitiveLeft(DensityIndex, i), primitiveLeft(EnergyIndex, i), primitiveLeft(WeightedVelocityX1, i),
                    primitiveLeft(WeightedVelocityX2, i), primitiveLeft(WeightedVelocityX3, i), primitiveLeft(MagneticFieldX1, i),
                    primitiveLeft(MagneticFieldX2, i), primitiveLeft(MagneticFieldX3, i),
                };
                const Real PrimitiveRightCArray[PrimitiveVariableNumber] = {
                    primitiveRight(DensityIndex, i), primitiveRight(EnergyIndex, i), primitiveRight(WeightedVelocityX1, i),
                    primitiveRight(WeightedVelocityX2, i), primitiveRight(WeightedVelocityX3, i), primitiveRight(MagneticFieldX1, i),
                    primitiveRight(MagneticFieldX2, i), primitiveRight(MagneticFieldX3, i),
                };

                Real gcovLeft[4][4], gconLeft[4][4], gcovRight[4][4], gconRight[4][4];
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        gcovLeft[row][col] = CovariantMetric(row * 4 + col, k, j, i - 1);
                        gconLeft[row][col] = ContravariantMetric(row * 4 + col, k, j, i - 1);
                        gcovRight[row][col] = CovariantMetric(row * 4 + col, k, j, i);
                        gconRight[row][col] = ContravariantMetric(row * 4 + col, k, j, i);
                    }
                }

                Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
                Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
                CalculateAlfvenVelocityGRMHD(AdiabaticIndex, PrimitiveLeftCArray, gcovLeft, gconLeft, X1DIR, maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
                CalculateAlfvenVelocityGRMHD(AdiabaticIndex, PrimitiveRightCArray, gcovRight, gconRight, X1DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
                const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft), maximumAlfvenVelocityRight));
                const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft), -minimumAlfvenVelocityRight));
                const auto AlfvenVelocityCenter = Kokkos::max(MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);

                const auto MetricDeterminantFace = MetricDeterminant(0, k, j, i);
                CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveLeftCArray, gcovLeft,
                                                gconLeft, MetricDeterminantFace, X0DIR,
                                                conservativeLeft);
                CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveRightCArray, gcovRight,
                                                gconRight, MetricDeterminantFace, X0DIR,
                                                conservativeRight);
                CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveLeftCArray, gcovLeft,
                                                gconLeft, MetricDeterminantFace, X1DIR,
                                                fluxLeft);
                CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveRightCArray, gcovRight,
                                                gconRight, MetricDeterminantFace, X1DIR,
                                                fluxRight);

                for (int index = 0; index < PrimitiveVariableNumber; ++index) {
                    conservative.flux(X1DIR, index, k, j, i) =
                        0.5 * (fluxLeft[index] + fluxRight[index] - AlfvenVelocityCenter * (conservativeRight[index] - conservativeLeft[index]));
                }
            });
        });

    if (MeshblockPointer->pmy_mesh->ndim >= 2)
    MeshblockPointer->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX2, ScratchLevel,
        BoundX3.s - OffsetX3, BoundX3.e + OffsetX3, BoundX1.s - OffsetX1, BoundX1.e + OffsetX1,
        KOKKOS_LAMBDA(team_mbr_t member, const int k, const int i) {
            ScratchPad2D<Real> primitiveLeft(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX2);
            ScratchPad2D<Real> primitiveRight(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX2);

            par_for_inner(member, 0, PrimitiveVariableNumber - 1, BoundX2.s, BoundX2.e + 1, [&](const int n, const int j) {
                primitiveLeft(n, j) = Primitive(n, k, j - 1, i) + 0.5 * InterpolateMC(Primitive(n, k, j - 2, i), Primitive(n, k, j - 1, i), Primitive(n, k, j, i));
                primitiveRight(n, j) = Primitive(n, k, j, i) - 0.5 * InterpolateMC(Primitive(n, k, j - 1, i), Primitive(n, k, j, i), Primitive(n, k, j + 1, i));
            });

            member.team_barrier();

            par_for_inner(member, BoundX2.s, BoundX2.e + 1, [&](const int j) {
                Real conservativeLeft[PrimitiveVariableNumber];
                Real conservativeRight[PrimitiveVariableNumber];
                Real fluxLeft[PrimitiveVariableNumber];
                Real fluxRight[PrimitiveVariableNumber];

                const Real PrimitiveLeftCArray[PrimitiveVariableNumber] = {
                    primitiveLeft(DensityIndex, j), primitiveLeft(EnergyIndex, j), primitiveLeft(WeightedVelocityX1, j),
                    primitiveLeft(WeightedVelocityX2, j), primitiveLeft(WeightedVelocityX3, j), primitiveLeft(MagneticFieldX1, j),
                    primitiveLeft(MagneticFieldX2, j), primitiveLeft(MagneticFieldX3, j),
                };
                const Real PrimitiveRightCArray[PrimitiveVariableNumber] = {
                    primitiveRight(DensityIndex, j), primitiveRight(EnergyIndex, j), primitiveRight(WeightedVelocityX1, j),
                    primitiveRight(WeightedVelocityX2, j), primitiveRight(WeightedVelocityX3, j), primitiveRight(MagneticFieldX1, j),
                    primitiveRight(MagneticFieldX2, j), primitiveRight(MagneticFieldX3, j),
                };

                Real gcovLeft[4][4], gconLeft[4][4], gcovRight[4][4], gconRight[4][4];
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        gcovLeft[row][col] = CovariantMetric(row * 4 + col, k, j - 1, i);
                        gconLeft[row][col] = ContravariantMetric(row * 4 + col, k, j - 1, i);
                        gcovRight[row][col] = CovariantMetric(row * 4 + col, k, j, i);
                        gconRight[row][col] = ContravariantMetric(row * 4 + col, k, j, i);
                    }
                }

                Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
                Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
                CalculateAlfvenVelocityGRMHD(AdiabaticIndex, PrimitiveLeftCArray, gcovLeft, gconLeft, X2DIR, maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
                CalculateAlfvenVelocityGRMHD(AdiabaticIndex, PrimitiveRightCArray, gcovRight, gconRight, X2DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
                const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft), maximumAlfvenVelocityRight));
                const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft), -minimumAlfvenVelocityRight));
                const auto AlfvenVelocityCenter = Kokkos::max(MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);

                const auto MetricDeterminantFace = MetricDeterminant(0, k, j, i);
                CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveLeftCArray, gcovLeft,
                                                gconLeft, MetricDeterminantFace, X0DIR,
                                                conservativeLeft);
                CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveRightCArray, gcovRight,
                                                gconRight, MetricDeterminantFace, X0DIR,
                                                conservativeRight);
                CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveLeftCArray, gcovLeft,
                                                gconLeft, MetricDeterminantFace, X2DIR,
                                                fluxLeft);
                CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveRightCArray, gcovRight,
                                                gconRight, MetricDeterminantFace, X2DIR,
                                                fluxRight);

                for (int index = 0; index < PrimitiveVariableNumber; ++index) {
                    conservative.flux(X2DIR, index, k, j, i) =
                        0.5 * (fluxLeft[index] + fluxRight[index] - AlfvenVelocityCenter * (conservativeRight[index] - conservativeLeft[index]));
                }
            });
        });

    if (MeshblockPointer->pmy_mesh->ndim == 3)
    MeshblockPointer->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX3, ScratchLevel,
        BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(team_mbr_t member, const int j, const int i) {
            ScratchPad2D<Real> primitiveLeft(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX3);
            ScratchPad2D<Real> primitiveRight(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX3);

            par_for_inner(member, 0, PrimitiveVariableNumber - 1, BoundX3.s, BoundX3.e + 1, [&](const int n, const int k) {
                primitiveLeft(n, k) = Primitive(n, k - 1, j, i) + 0.5 * InterpolateMC(Primitive(n, k - 2, j, i), Primitive(n, k - 1, j, i), Primitive(n, k, j, i));
                primitiveRight(n, k) = Primitive(n, k, j, i) - 0.5 * InterpolateMC(Primitive(n, k - 1, j, i), Primitive(n, k, j, i), Primitive(n, k + 1, j, i));
            });

            member.team_barrier();

            par_for_inner(member, BoundX3.s, BoundX3.e + 1, [&](const int k) {
                Real conservativeLeft[PrimitiveVariableNumber];
                Real conservativeRight[PrimitiveVariableNumber];
                Real fluxLeft[PrimitiveVariableNumber];
                Real fluxRight[PrimitiveVariableNumber];

                const Real PrimitiveLeftCArray[PrimitiveVariableNumber] = {
                    primitiveLeft(DensityIndex, k), primitiveLeft(EnergyIndex, k), primitiveLeft(WeightedVelocityX1, k),
                    primitiveLeft(WeightedVelocityX2, k), primitiveLeft(WeightedVelocityX3, k), primitiveLeft(MagneticFieldX1, k),
                    primitiveLeft(MagneticFieldX2, k), primitiveLeft(MagneticFieldX3, k),
                };
                const Real PrimitiveRightCArray[PrimitiveVariableNumber] = {
                    primitiveRight(DensityIndex, k), primitiveRight(EnergyIndex, k), primitiveRight(WeightedVelocityX1, k),
                    primitiveRight(WeightedVelocityX2, k), primitiveRight(WeightedVelocityX3, k), primitiveRight(MagneticFieldX1, k),
                    primitiveRight(MagneticFieldX2, k), primitiveRight(MagneticFieldX3, k),
                };

                Real gcovLeft[4][4], gconLeft[4][4], gcovRight[4][4], gconRight[4][4];
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        gcovLeft[row][col] = CovariantMetric(row * 4 + col, k - 1, j, i);
                        gconLeft[row][col] = ContravariantMetric(row * 4 + col, k - 1, j, i);
                        gcovRight[row][col] = CovariantMetric(row * 4 + col, k, j, i);
                        gconRight[row][col] = ContravariantMetric(row * 4 + col, k, j, i);
                    }
                }

                Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
                Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
                CalculateAlfvenVelocityGRMHD(AdiabaticIndex, PrimitiveLeftCArray, gcovLeft, gconLeft, X3DIR, maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
                CalculateAlfvenVelocityGRMHD(AdiabaticIndex, PrimitiveRightCArray, gcovRight, gconRight, X3DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
                const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft), maximumAlfvenVelocityRight));
                const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft), -minimumAlfvenVelocityRight));
                const auto AlfvenVelocityCenter = Kokkos::max(MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);

                const auto MetricDeterminantFace = MetricDeterminant(0, k, j, i);
                CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveLeftCArray, gcovLeft,
                                                gconLeft, MetricDeterminantFace, X0DIR,
                                                conservativeLeft);
                CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveRightCArray, gcovRight,
                                                gconRight, MetricDeterminantFace, X0DIR,
                                                conservativeRight);
                CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveLeftCArray, gcovLeft,
                                                gconLeft, MetricDeterminantFace, X3DIR,
                                                fluxLeft);
                CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveRightCArray, gcovRight,
                                                gconRight, MetricDeterminantFace, X3DIR,
                                                fluxRight);

                for (int index = 0; index < PrimitiveVariableNumber; ++index) {
                    conservative.flux(X3DIR, index, k, j, i) =
                        0.5 * (fluxLeft[index] + fluxRight[index] - AlfvenVelocityCenter * (conservativeRight[index] - conservativeLeft[index]));
                }
            });
        });

    return TaskStatus::complete;
}
