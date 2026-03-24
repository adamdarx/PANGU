#pragma once
#include <string>
#include <vector>
#include <memory>

#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"

parthenon::TaskStatus FixRecovery(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    PARTHENON_INSTRUMENT
    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto PackageCORE = MeshblockPointer->packages.Get("CORE");

    auto &flag = resource->Get("Flag").data;

    const auto BoundX1 = MeshblockPointer->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
    const auto BoundX2 = MeshblockPointer->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
    const auto BoundX3 = MeshblockPointer->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);

    parthenon::PackIndexMap primitiveIndexMap;
    const std::vector<std::string> PrimitiveTags = {"Density", "Energy", "WeightedVelocity", "MagneticField"};
    auto primitive = resource->PackVariables(PrimitiveTags, primitiveIndexMap);

    const int OffsetX2 = (BoundX2.s != BoundX2.e) ? 1 : 0;
    MeshblockPointer->par_for(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (flag(k, j, i) == 0) {
                int pf1 = flag(k, j + OffsetX2, i - 1);
                int pf2 = flag(k, j + OffsetX2, i);
                int pf3 = flag(k, j + OffsetX2, i + 1);
                int pf4 = flag(k, j, i + 1);
                int pf5 = flag(k, j - OffsetX2, i + 1);
                int pf6 = flag(k, j - OffsetX2, i);
                int pf7 = flag(k, j - OffsetX2, i - 1);
                int pf8 = flag(k, j, i - 1);

                if (pf2 && pf4 && pf6 && pf8) {
                    for (int m = 0; m < PrimitiveVariableNumber; m++)
                        primitive(m, k, j, i) = 0.25 * (primitive(m, k, j + OffsetX2, i) + primitive(m, k, j - OffsetX2, i) + primitive(m, k, j, i - 1) + primitive(m, k, j, i + 1));
                }
                else if (pf1 && pf3 && pf5 && pf7) {
                    for (int m = 0; m < PrimitiveVariableNumber; m++)
                        primitive(m, k, j, i) = 0.25 * (primitive(m, k, j + OffsetX2, i + 1) + primitive(m, k, j + OffsetX2, i - 1) + primitive(m, k, j - OffsetX2, i + 1) + primitive(m, k, j - OffsetX2, i - 1));
                }
                else {
                    for (int m = 0; m < PrimitiveVariableNumber; m++)
                        primitive(m, k, j, i) = 0.125 * (primitive(m, k, j + OffsetX2, i - 1) + primitive(m, k, j + OffsetX2, i) + primitive(m, k, j + OffsetX2, i + 1) + primitive(m, k, j, i + 1) + primitive(m, k, j - OffsetX2, i + 1) + primitive(m, k, j - OffsetX2, i) + primitive(m, k, j - OffsetX2, i - 1) + primitive(m, k, j, i - 1));
                    primitive(WeightedVelocityX1, k, j, i) = 0.;
                    primitive(WeightedVelocityX2, k, j, i) = 0.;
                    primitive(WeightedVelocityX3, k, j, i) = 0.;
                }
                flag(k, j, i) = 1;
            }
        });

    return parthenon::TaskStatus::complete;
}
