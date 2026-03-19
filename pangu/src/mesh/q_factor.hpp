#pragma once
#include <memory>
#include <vector>

#include <globals.hpp>
#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"

TaskStatus CalculateQFactor(std::shared_ptr<MeshBlockData<Real>> &resource) {
    using namespace parthenon::package::prelude;
    
    PARTHENON_INSTRUMENT

    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto Package = MeshblockPointer->packages.Get("PANGU");
    const auto AdiabaticIndex = Package->Param<Real>("AdiabaticIndex");

    const auto BoundX1 = MeshblockPointer->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto BoundX2 = MeshblockPointer->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto BoundX3 = MeshblockPointer->cellbounds.GetBoundsK(IndexDomain::interior);

    const auto &Coords = MeshblockPointer->coords;
    
    PackIndexMap primitiveIndexMap;
    const std::vector<std::string> PrimitiveTags = {"Density", "Energy", "WeightedVelocity", "MagneticField"};
    const auto Primitive = resource->PackVariables(PrimitiveTags, primitiveIndexMap);
    PackIndexMap qFactorIndexMap;
    const std::vector<std::string> QFactorTags = {"QFactor"};
    auto QFactor = resource->PackVariables(QFactorTags, qFactorIndexMap);

    MeshblockPointer->par_for(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const auto SquaredWeightedVelocity = Kokkos::pow(Primitive(WeightedVelocityX1, k, j, i), 2) + Kokkos::pow(Primitive(WeightedVelocityX2, k, j, i), 2) + Kokkos::pow(Primitive(WeightedVelocityX3, k, j, i), 2);
            const auto SquaredLorentzFactor = 1 + SquaredWeightedVelocity;
            const auto LorentzFactor = Kokkos::sqrt(1 + SquaredWeightedVelocity);
            const auto SquaredMagneticFieldThreeVector = Kokkos::pow(Primitive(MagneticFieldX1, k, j, i), 2) + Kokkos::pow(Primitive(MagneticFieldX2, k, j, i), 2) + Kokkos::pow(Primitive(MagneticFieldX3, k, j, i), 2);
            const auto MagneticFieldThreeVectorDotWeightedVelocity = Primitive(WeightedVelocityX1, k, j, i) * Primitive(MagneticFieldX1, k, j, i) + Primitive(WeightedVelocityX2, k, j, i) * Primitive(MagneticFieldX2, k, j, i) + Primitive(WeightedVelocityX3, k, j, i) * Primitive(MagneticFieldX3, k, j, i);
            const auto SquaredMagneticFieldFourVector = (SquaredMagneticFieldThreeVector + Kokkos::pow(MagneticFieldThreeVectorDotWeightedVelocity, 2)) / SquaredLorentzFactor;
            const auto Enthalpy = Primitive(EnergyIndex, k, j, i) + AdiabaticIndex * Primitive(DensityIndex, k, j, i);     
            const auto Energy = SquaredMagneticFieldFourVector + Enthalpy;
            
            QFactor(0, k, j, i) = 2 * M_PI * (LorentzFactor * Primitive(MagneticFieldX2, k, j, i) + MagneticFieldThreeVectorDotWeightedVelocity * Primitive(WeightedVelocityX2, k, j, i)) / (LorentzFactor * Primitive(WeightedVelocityX2, k, j, i) * Kokkos::sqrt(Energy) * Coords.Dxc<X2DIR>());
        }
    );
    return TaskStatus::complete;
}