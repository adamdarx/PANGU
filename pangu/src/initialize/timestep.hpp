#pragma once
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"
#include "../physics/AlfvenVelocity.hpp"
#include "../reconstruct/InterpolaterMC.hpp"

parthenon::Real EstimateTimestepBlock(parthenon::MeshBlockData<parthenon::Real> *resource) {
    using namespace parthenon;
    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto PackageCORE = MeshblockPointer->packages.Get("CORE");
    const auto CFLNumber = PackageCORE->Param<Real>("CFLNumber");
    const auto AdiabaticIndex = PackageCORE->Param<Real>("AdiabaticIndex");

    const auto BoundX1 = MeshblockPointer->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto BoundX2 = MeshblockPointer->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto BoundX3 = MeshblockPointer->cellbounds.GetBoundsK(IndexDomain::interior);

    const auto &Coords = MeshblockPointer->coords;

    Real minimumOfTimestepX1 = std::numeric_limits<double>::max();
    Real minimumOfTimestepX2 = std::numeric_limits<double>::max();
    Real minimumOfTimestepX3 = std::numeric_limits<double>::max();

    PackIndexMap primitiveIndexMap;
    const std::vector<std::string> PrimitiveTags = {"Density", "Energy", "WeightedVelocity", "MagneticField"};
    const auto Primitive = resource->PackVariables(PrimitiveTags, primitiveIndexMap);
    PackIndexMap alfvenVelocityIndexMap;
    const std::vector<std::string> AlfvenTags = {"Alfven"};
    auto AlfvenVelocity = resource->PackVariables(AlfvenTags, alfvenVelocityIndexMap);

    const int ScratchLevel = 1;
    const int MeshgridSizeX1 = MeshblockPointer->cellbounds.ncellsi(IndexDomain::entire);
    const int MeshgridSizeX2 = MeshblockPointer->cellbounds.ncellsj(IndexDomain::entire);
    const int MeshgridSizeX3 = MeshblockPointer->cellbounds.ncellsk(IndexDomain::entire);

    const size_t ScratchSizeInBytesX1 = ScratchPad2D<Real>::shmem_size(PrimitiveVariableNumber, MeshgridSizeX1);
    const size_t ScratchSizeInBytesX2 = ScratchPad2D<Real>::shmem_size(PrimitiveVariableNumber, MeshgridSizeX2);
    const size_t ScratchSizeInBytesX3 = ScratchPad2D<Real>::shmem_size(PrimitiveVariableNumber, MeshgridSizeX3);

    const int OffsetX1 = (MeshgridSizeX1 > 1) ? 1 : 0;
    const int OffsetX2 = (MeshgridSizeX2 > 1) ? 1 : 0;
    const int OffsetX3 = (MeshgridSizeX3 > 1) ? 1 : 0;

    if(OffsetX1)
    MeshblockPointer->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX1, ScratchLevel, BoundX3.s - OffsetX3, BoundX3.e + OffsetX3, BoundX2.s - OffsetX2, BoundX2.e + OffsetX2,
        KOKKOS_LAMBDA(team_mbr_t member, const int k, const int j) {
            ScratchPad2D<Real> PrimitiveLeft(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX1);
            ScratchPad2D<Real> PrimitiveRight(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX1);

            par_for_inner(member, 0, PrimitiveVariableNumber - 1, BoundX1.s, BoundX1.e + 1, [&](const int n, const int i) {
                PrimitiveLeft(n, i) = Primitive(n, k, j, i - 1) + 0.5 * InterpolateMC(Primitive(n, k, j, i - 2), Primitive(n, k, j, i - 1), Primitive(n, k, j, i));
                PrimitiveRight(n, i) = Primitive(n, k, j, i) - 0.5 * InterpolateMC(Primitive(n, k, j, i - 1), Primitive(n, k, j, i), Primitive(n, k, j, i + 1));
            });
            
            member.team_barrier();

            par_for_inner(member, BoundX1.s, BoundX1.e + 1, [&](const int i) {
                const Real PrimitveLeftCArray[PrimitiveVariableNumber] = {
                    PrimitiveLeft(DensityIndex, i),
                    PrimitiveLeft(EnergyIndex, i),
                    PrimitiveLeft(WeightedVelocityX1, i),
                    PrimitiveLeft(WeightedVelocityX2, i),
                    PrimitiveLeft(WeightedVelocityX3, i),
                    PrimitiveLeft(MagneticFieldX1, i),
                    PrimitiveLeft(MagneticFieldX2, i),
                    PrimitiveLeft(MagneticFieldX3, i),
                };

                const Real PrimitveRightCArray[PrimitiveVariableNumber] = {
                    PrimitiveRight(DensityIndex, i),
                    PrimitiveRight(EnergyIndex, i),
                    PrimitiveRight(WeightedVelocityX1, i),
                    PrimitiveRight(WeightedVelocityX2, i),
                    PrimitiveRight(WeightedVelocityX3, i),
                    PrimitiveRight(MagneticFieldX1, i),
                    PrimitiveRight(MagneticFieldX2, i),
                    PrimitiveRight(MagneticFieldX3, i),
                };

                Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
                Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
                CalculateAlfvenVelocitySRMHD(AdiabaticIndex, PrimitveLeftCArray, X1DIR, maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
                CalculateAlfvenVelocitySRMHD(AdiabaticIndex, PrimitveRightCArray, X1DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
                const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft), maximumAlfvenVelocityRight));
                const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft), -minimumAlfvenVelocityRight));
                AlfvenVelocity(Vector3D::X1, k, j, i) = Kokkos::max(MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);
            });
        });

    if(OffsetX2)
    MeshblockPointer->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX2, ScratchLevel, BoundX3.s - OffsetX3, BoundX3.e + OffsetX3, BoundX1.s - OffsetX1, BoundX1.e + OffsetX1,
        KOKKOS_LAMBDA(team_mbr_t member, const int k, const int i) {
            ScratchPad2D<Real> PrimitiveLeft(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX2);
            ScratchPad2D<Real> PrimitiveRight(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX2);

            par_for_inner(member, 0, PrimitiveVariableNumber - 1, BoundX2.s, BoundX2.e + 1, [&](const int n, const int j) {
                PrimitiveLeft(n, i) = Primitive(n, k, j - 1, i) + 0.5 * InterpolateMC(Primitive(n, k, j - 2, i), Primitive(n, k, j - 1, i), Primitive(n, k, j, i));
                PrimitiveRight(n, i) = Primitive(n, k, j, i) - 0.5 * InterpolateMC(Primitive(n, k, j - 1, i), Primitive(n, k, j, i), Primitive(n, k, j + 1, i));
            });
            
            member.team_barrier();

            par_for_inner(member, BoundX2.s, BoundX2.e + 1, [&](const int j) {
                Real PrimitveLeftCArray[PrimitiveVariableNumber] = {
                    PrimitiveLeft(DensityIndex, j),
                    PrimitiveLeft(EnergyIndex, j),
                    PrimitiveLeft(WeightedVelocityX1, j),
                    PrimitiveLeft(WeightedVelocityX2, j),
                    PrimitiveLeft(WeightedVelocityX3, j),
                    PrimitiveLeft(MagneticFieldX1, j),
                    PrimitiveLeft(MagneticFieldX2, j),
                    PrimitiveLeft(MagneticFieldX3, j),
                };

                Real PrimitveRightCArray[PrimitiveVariableNumber] = {
                    PrimitiveRight(DensityIndex, j),
                    PrimitiveRight(EnergyIndex, j),
                    PrimitiveRight(WeightedVelocityX1, j),
                    PrimitiveRight(WeightedVelocityX2, j),
                    PrimitiveRight(WeightedVelocityX3, j),
                    PrimitiveRight(MagneticFieldX1, j),
                    PrimitiveRight(MagneticFieldX2, j),
                    PrimitiveRight(MagneticFieldX3, j),
                };

                Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
                Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
                CalculateAlfvenVelocitySRMHD(AdiabaticIndex, PrimitveLeftCArray, X2DIR, maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
                CalculateAlfvenVelocitySRMHD(AdiabaticIndex, PrimitveRightCArray, X2DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
                const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft), maximumAlfvenVelocityRight));
                const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft), -minimumAlfvenVelocityRight));
                AlfvenVelocity(Vector3D::X2, k, j, i) = Kokkos::max(MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);
            });
        });
    
    if(OffsetX3)
    MeshblockPointer->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX3, ScratchLevel, BoundX2.s - OffsetX2, BoundX2.e + OffsetX2, BoundX1.s - OffsetX1, BoundX1.e + OffsetX1,
        KOKKOS_LAMBDA(team_mbr_t member, const int j, const int i) {
            ScratchPad2D<Real> PrimitiveLeft(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX3);
            ScratchPad2D<Real> PrimitiveRight(member.team_scratch(ScratchLevel), PrimitiveVariableNumber, MeshgridSizeX3);

            par_for_inner(member, 0, PrimitiveVariableNumber - 1, BoundX3.s, BoundX3.e + 1, [&](const int n, const int k) {
                PrimitiveLeft(n, i) = Primitive(n, k - 1, j, i) + 0.5 * InterpolateMC(Primitive(n, k - 2, j, i), Primitive(n, k - 1, j, i), Primitive(n, k, j, i));
                PrimitiveRight(n, i) = Primitive(n, k, j, i) - 0.5 * InterpolateMC(Primitive(n, k - 1, j, i), Primitive(n, k, j, i), Primitive(n, k + 1, j, i));
            });
            
            member.team_barrier();

            par_for_inner(member, BoundX3.s, BoundX3.e + 1, [&](const int k) {
                Real PrimitveLeftCArray[PrimitiveVariableNumber] = {
                    PrimitiveLeft(DensityIndex, k),
                    PrimitiveLeft(EnergyIndex, k),
                    PrimitiveLeft(WeightedVelocityX1, k),
                    PrimitiveLeft(WeightedVelocityX2, k),
                    PrimitiveLeft(WeightedVelocityX3, k),
                    PrimitiveLeft(MagneticFieldX1, k),
                    PrimitiveLeft(MagneticFieldX2, k),
                    PrimitiveLeft(MagneticFieldX3, k),
                };

                Real PrimitveRightCArray[PrimitiveVariableNumber] = {
                    PrimitiveRight(DensityIndex, k),
                    PrimitiveRight(EnergyIndex, k),
                    PrimitiveRight(WeightedVelocityX1, k),
                    PrimitiveRight(WeightedVelocityX2, k),
                    PrimitiveRight(WeightedVelocityX3, k),
                    PrimitiveRight(MagneticFieldX1, k),
                    PrimitiveRight(MagneticFieldX2, k),
                    PrimitiveRight(MagneticFieldX3, k),
                };

                Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
                Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
                CalculateAlfvenVelocitySRMHD(AdiabaticIndex, PrimitveLeftCArray, X3DIR, maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
                CalculateAlfvenVelocitySRMHD(AdiabaticIndex, PrimitveRightCArray, X3DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
                const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft), maximumAlfvenVelocityRight));
                const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft), -minimumAlfvenVelocityRight));
                AlfvenVelocity(Vector3D::X3, k, j, i) = Kokkos::max(MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);
            });
        });

    if(OffsetX1)
    MeshblockPointer->par_reduce(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real& timestepX1) {
            if(timestepX1 > Coords.Dx<X1DIR>() / AlfvenVelocity(Vector3D::X1, k, j, i))
                timestepX1 = Coords.Dx<X1DIR>() / AlfvenVelocity(Vector3D::X1, k, j, i);
        },
        Kokkos::Min<Real>(minimumOfTimestepX1));

    if(OffsetX2)
    MeshblockPointer->par_reduce(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real& timestepX2) {
            if(timestepX2 > Coords.Dx<X2DIR>() / AlfvenVelocity(Vector3D::X2, k, j, i))
                timestepX2 = Coords.Dx<X2DIR>() / AlfvenVelocity(Vector3D::X2, k, j, i);
        },
        Kokkos::Min<Real>(minimumOfTimestepX2));
    
    if(OffsetX3)
    MeshblockPointer->par_reduce(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real& timestepX3) {
            if(timestepX3 > Coords.Dx<X3DIR>() / AlfvenVelocity(Vector3D::X3, k, j, i))
                timestepX3 = Coords.Dx<X3DIR>() / AlfvenVelocity(Vector3D::X3, k, j, i);
        },
        Kokkos::Min<Real>(minimumOfTimestepX3));

    const Real MinimumOfTimestep = 1 / (1 / minimumOfTimestepX1 + 1 / minimumOfTimestepX2 + 1 / minimumOfTimestepX3);
    return CFLNumber * MinimumOfTimestep;
}
