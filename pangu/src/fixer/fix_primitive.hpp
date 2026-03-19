#pragma once
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"

parthenon::TaskStatus FixPrimitive(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto Package = MeshblockPointer->packages.Get("PANGU");
    const auto AdiabaticIndex = Package->Param<Real>("AdiabaticIndex");

    const auto BoundX1 = MeshblockPointer->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto BoundX2 = MeshblockPointer->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto BoundX3 = MeshblockPointer->cellbounds.GetBoundsK(IndexDomain::interior);

    parthenon::PackIndexMap primitiveIndexMap;
    const std::vector<std::string> PrimitiveTags = {"Density", "Energy", "WeightedVelocity", "MagneticField"};
    auto primitive = resource->PackVariables(PrimitiveTags, primitiveIndexMap);
    
    MeshblockPointer->par_for(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if(primitive(DensityIndex, k, j, i) < 1e-6)
                primitive(DensityIndex, k, j, i) = 1e-6;
            if(primitive(EnergyIndex, k, j, i) < Kokkos::pow(1e-6, AdiabaticIndex))
                primitive(EnergyIndex, k, j, i) = Kokkos::pow(1e-6, AdiabaticIndex);
        });

    return parthenon::TaskStatus::complete;
}
