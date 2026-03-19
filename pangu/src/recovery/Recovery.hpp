#pragma once
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"

#include "Scheme1D.hpp"
#include "Scheme1Dvsq.hpp"
#include "Scheme2D.hpp"

parthenon::TaskStatus TransformConservativeToPrimitive(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    PARTHENON_INSTRUMENT

    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto Package = MeshblockPointer->packages.Get("PANGU");
    const auto AdiabaticIndex = Package->Param<parthenon::Real>("AdiabaticIndex");

    const auto BoundX1 = MeshblockPointer->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
    const auto BoundX2 = MeshblockPointer->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
    const auto BoundX3 = MeshblockPointer->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);

    PackIndexMap primitiveIndexMap;
    const std::vector<std::string> PrimitiveTags = {"Density", "Energy", "WeightedVelocity", "MagneticField"};
    auto primitive = resource->PackVariables(PrimitiveTags, primitiveIndexMap);
    PackIndexMap conservativeIndexMap;
    const std::vector<std::string> ConservativeTags = {"Conservative"};
    const auto Conservative = resource->PackVariables(ConservativeTags, conservativeIndexMap);
    auto &flag = resource->Get("Flag").data;

    MeshblockPointer->par_for(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            parthenon::Real conservativeCArray[PrimitiveVariableNumber], primitiveCArray[PrimitiveVariableNumber];
            for (int index = 0; index < PrimitiveVariableNumber; ++index) {
                conservativeCArray[index] = Conservative(index, k, j, i);
                primitiveCArray[index] = primitive(index, k, j, i);
            }
            
            conservativeCArray[EnergyIndex] -= conservativeCArray[DensityIndex];

            flag(k, j, i) = 0;
            if (
                Scheme2D::restore(conservativeCArray, primitiveCArray, AdiabaticIndex) == 0
                ||
                Scheme1D::restore(conservativeCArray, primitiveCArray, AdiabaticIndex) == 0
                ||
                Scheme1Dvsq::restore(conservativeCArray, primitiveCArray, AdiabaticIndex) == 0
            ) {
                for (int index = 0; index < PrimitiveVariableNumber; ++index)
                    primitive(index, k, j, i) = primitiveCArray[index];
                flag(k, j, i) = 1;
            }
            
        });
    return parthenon::TaskStatus::complete;
}
