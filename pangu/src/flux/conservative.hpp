#pragma once
#include <memory>
#include <string>
#include <vector>

#include <basic_types.hpp>
#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"
#include "../physics/EnergyMomentumTensor.hpp"

parthenon::TaskStatus CalculateConservative(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    using namespace parthenon;
    PARTHENON_INSTRUMENT
    
    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto Package = MeshblockPointer->packages.Get("PANGU");
    const auto AdiabaticIndex = Package->Param<Real>("AdiabaticIndex");

    const auto BoundX1 = MeshblockPointer->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto BoundX2 = MeshblockPointer->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto BoundX3 = MeshblockPointer->cellbounds.GetBoundsK(IndexDomain::interior);

    PackIndexMap primitiveIndexMap;
    const std::vector<std::string> PrimitiveTags = {"Density", "Energy", "WeightedVelocity", "MagneticField"};
    const auto Primitive = resource->PackVariables(PrimitiveTags, primitiveIndexMap);
    PackIndexMap conservativeIndexMap;
    const std::vector<std::string> ConservativeTags = {"Conservative"};
    auto conservative = resource->PackVariablesAndFluxes(ConservativeTags, conservativeIndexMap);
    
    MeshblockPointer->par_for(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const auto SquaredWeightedVelocity = Kokkos::pow(Primitive(WeightedVelocityX1, k, j, i), 2) + Kokkos::pow(Primitive(WeightedVelocityX2, k, j, i), 2) + Kokkos::pow(Primitive(WeightedVelocityX3, k, j, i), 2);
            const auto SquaredLorentzFactor = 1 + SquaredWeightedVelocity;
            const auto LorentzFactor = Kokkos::sqrt(SquaredLorentzFactor);
            const auto MagneticFieldThreeVectorDotWeightedVelocity = Primitive(WeightedVelocityX1, k, j, i) * Primitive(MagneticFieldX1, k, j, i) + Primitive(WeightedVelocityX2, k, j, i) * Primitive(MagneticFieldX2, k, j, i) + Primitive(WeightedVelocityX3, k, j, i) * Primitive(MagneticFieldX3, k, j, i);
            Real directedEnergyMomentumTensor[4];
            const Real PrimitiveCArray[PrimitiveVariableNumber] = {
                    Primitive(DensityIndex, k, j, i),
                    Primitive(EnergyIndex, k, j, i),
                    Primitive(WeightedVelocityX1, k, j, i),
                    Primitive(WeightedVelocityX2, k, j, i),
                    Primitive(WeightedVelocityX3, k, j, i),
                    Primitive(MagneticFieldX1, k, j, i),
                    Primitive(MagneticFieldX2, k, j, i),
                    Primitive(MagneticFieldX3, k, j, i),
                };
            
            CalculateEnergyMomentumTensor(AdiabaticIndex, PrimitiveCArray, X0DIR, directedEnergyMomentumTensor);
            conservative(DensityIndex, k, j, i) = Primitive(DensityIndex, k, j, i) * LorentzFactor;
            conservative(EnergyIndex, k, j, i) = directedEnergyMomentumTensor[0] + conservative(DensityIndex, k, j, i);
            conservative(WeightedVelocityX1, k, j, i) = directedEnergyMomentumTensor[1];
            conservative(WeightedVelocityX2, k, j, i) = directedEnergyMomentumTensor[2];
            conservative(WeightedVelocityX3, k, j, i) = directedEnergyMomentumTensor[3];

            conservative(MagneticFieldX1, k, j, i) = Primitive(MagneticFieldX1, k, j, i);
            conservative(MagneticFieldX2, k, j, i) = Primitive(MagneticFieldX2, k, j, i);
            conservative(MagneticFieldX3, k, j, i) = Primitive(MagneticFieldX3, k, j, i);
        });

    return TaskStatus::complete;
}