#pragma once
#include <memory>
#include <string>
#include <vector>

#include <basic_types.hpp>
#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"
#include "../physics/flux.hpp"

parthenon::TaskStatus CalculateConservativeSRMHD(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    using namespace parthenon;
    PARTHENON_INSTRUMENT
    
    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto PackageCORE = MeshblockPointer->packages.Get("CORE");
    const auto AdiabaticIndex = PackageCORE->Param<Real>("AdiabaticIndex");

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

            Real ConservativeState[PrimitiveVariableNumber];
            CalculateContravariantFluxSRMHD(AdiabaticIndex, PrimitiveCArray,
                                            X0DIR, ConservativeState);

            for (int index = 0; index < PrimitiveVariableNumber; ++index) {
                conservative(index, k, j, i) = ConservativeState[index];
            }
        });

    return TaskStatus::complete;
}

parthenon::TaskStatus CalculateConservativeGRMHD(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    using namespace parthenon;
    PARTHENON_INSTRUMENT

    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto PackageCORE = MeshblockPointer->packages.Get("CORE");
    const auto PackageMETRIC = MeshblockPointer->packages.Get("METRIC");
    const auto AdiabaticIndex = PackageCORE->Param<Real>("AdiabaticIndex");
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

    MeshblockPointer->par_for(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            Real gcov[4][4];
            Real gcon[4][4];
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    gcov[row][col] = CovariantMetric(row * 4 + col, k, j, i);
                    gcon[row][col] = ContravariantMetric(row * 4 + col, k, j, i);
                }
            }

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

            Real ConservativeState[PrimitiveVariableNumber];
            CalculateContravariantFluxGRMHD(AdiabaticIndex, PrimitiveCArray, gcov, gcon,
                                            MetricDeterminant(0, k, j, i), X0DIR,
                                            ConservativeState);

            for (int index = 0; index < PrimitiveVariableNumber; ++index) {
                conservative(index, k, j, i) = ConservativeState[index];
            }
        });

    return TaskStatus::complete;
}