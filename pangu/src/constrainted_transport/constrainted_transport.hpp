#pragma once
#include <memory>
#include <vector>

#include <basic_types.hpp>
#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"

parthenon::TaskStatus ConstraintedTransport(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    using namespace parthenon;
    PARTHENON_INSTRUMENT

    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto PackageCORE = MeshblockPointer->packages.Get("CORE");
    const auto &AdiabaticIndex = PackageCORE->Param<parthenon::Real>("AdiabaticIndex");

    const auto BoundX1 = MeshblockPointer->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
    const auto BoundX2 = MeshblockPointer->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
    const auto BoundX3 = MeshblockPointer->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);

    const auto &Coords = MeshblockPointer->coords;

    parthenon::Real minimumOfTimestepX1 = std::numeric_limits<double>::max();
    parthenon::Real minimumOfTimestepX2 = std::numeric_limits<double>::max();
    parthenon::Real minimumOfTimestepX3 = std::numeric_limits<double>::max();

    parthenon::PackIndexMap conservativeIndexMap;
    const std::vector<std::string> ConservativeTags = {"Conservative"};
    auto conservative = resource->PackVariablesAndFluxes(ConservativeTags, conservativeIndexMap);
    parthenon::PackIndexMap electricFieldIndexMap;
    const std::vector<std::string> ElectricFieldTags = {"ElectricField"};
    auto electricField = resource->PackVariables(ElectricFieldTags, electricFieldIndexMap);

    const int MeshgridSizeX1 = MeshblockPointer->cellbounds.ncellsi(parthenon::IndexDomain::entire);
    const int MeshgridSizeX2 = MeshblockPointer->cellbounds.ncellsj(parthenon::IndexDomain::entire);
    const int MeshgridSizeX3 = MeshblockPointer->cellbounds.ncellsk(parthenon::IndexDomain::entire);

    const int OffsetX1 = (MeshgridSizeX1 > 1) ? 1 : 0;
    const int OffsetX2 = (MeshgridSizeX2 > 1) ? 1 : 0;
    const int OffsetX3 = (MeshgridSizeX3 > 1) ? 1 : 0;

    const int CalculateElectricFieldX1 = OffsetX2 && OffsetX3;
    const int CalculateElectricFieldX2 = OffsetX3 && OffsetX1;
    const int CalculateElectricFieldX3 = OffsetX1 && OffsetX2;

    MeshblockPointer->par_for(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if(CalculateElectricFieldX1)
                electricField(Vector3D::X1, k, j, i) = 0.25 * ((conservative.flux(X2DIR, MagneticFieldX3, k, j, i) + conservative.flux(X2DIR, MagneticFieldX3, k - 1, j, i)) - (conservative.flux(X3DIR, MagneticFieldX2, k, j, i) + conservative.flux(X3DIR, MagneticFieldX2, k, j - 1, i)));
            if(CalculateElectricFieldX2)
                electricField(Vector3D::X2, k, j, i) = 0.25 * ((conservative.flux(X3DIR, MagneticFieldX1, k, j, i) + conservative.flux(X3DIR, MagneticFieldX1, k, j, i - 1)) - (conservative.flux(X1DIR, MagneticFieldX3, k, j, i) + conservative.flux(X1DIR, MagneticFieldX3, k - 1, j, i)));
            if(CalculateElectricFieldX3)
                electricField(Vector3D::X3, k, j, i) = 0.25 * ((conservative.flux(X1DIR, MagneticFieldX2, k, j, i) + conservative.flux(X1DIR, MagneticFieldX2, k, j - 1, i)) - (conservative.flux(X2DIR, MagneticFieldX1, k, j, i) + conservative.flux(X2DIR, MagneticFieldX1, k, j, i - 1)));
        });
    
    MeshblockPointer->par_for(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e + OffsetX1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if(OffsetX1)
                conservative.flux(X1DIR, MagneticFieldX1, k, j, i) = 0;
            if(CalculateElectricFieldX2)
                conservative.flux(X1DIR, MagneticFieldX3, k, j, i) = -0.5 * (electricField(Vector3D::X2, k, j, i) + electricField(Vector3D::X2, k + 1, j, i));
            if(CalculateElectricFieldX3)
                conservative.flux(X1DIR, MagneticFieldX2, k, j, i) = 0.5 * (electricField(Vector3D::X3, k, j, i) + electricField(Vector3D::X3, k, j + 1, i));  
        });
    
    MeshblockPointer->par_for(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e + OffsetX2, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if(CalculateElectricFieldX1)
                conservative.flux(X2DIR, MagneticFieldX3, k, j, i) = 0.5 * (electricField(Vector3D::X1, k, j, i) + electricField(Vector3D::X1, k + 1, j, i));
            if(OffsetX2)
                conservative.flux(X2DIR, MagneticFieldX2, k, j, i) = 0.;
            if(CalculateElectricFieldX3)
                conservative.flux(X2DIR, MagneticFieldX1, k, j, i) = -0.5 * (electricField(Vector3D::X3, k, j, i) + electricField(Vector3D::X3, k, j, i + 1));  
        });

    MeshblockPointer->par_for(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e + OffsetX3, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if(CalculateElectricFieldX1)
                conservative.flux(X3DIR, MagneticFieldX2, k, j, i) = -0.5 * (electricField(Vector3D::X1, k, j, i) + electricField(Vector3D::X1, k, j + 1, i));
            if(CalculateElectricFieldX2)
                conservative.flux(X3DIR, MagneticFieldX1, k, j, i) = 0.5 * (electricField(Vector3D::X2, k, j, i) + electricField(Vector3D::X2, k, j, i + 1));
            if(OffsetX3)
                conservative.flux(X3DIR, MagneticFieldX3, k, j, i) = 0;
        });
    
    return parthenon::TaskStatus::complete;
}
