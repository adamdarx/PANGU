#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <globals.hpp>
#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"

using namespace parthenon::package::prelude;

AmrTag CheckRefinement(MeshBlockData<Real> *resource) {
    const auto MeshblockPointer = resource->GetBlockPointer();
    const auto Package = MeshblockPointer->packages.Get("PANGU");
    const auto QFactorFloor = Package->Param<Real>("QFactorFloor");
    const auto QFactorCeiling = Package->Param<Real>("QFactorCeiling");
    const auto QFactor = resource->Get("QFactor").data;

    const auto BoundX1 = MeshblockPointer->cellbounds.GetBoundsI(IndexDomain::entire);
    const auto BoundX2 = MeshblockPointer->cellbounds.GetBoundsJ(IndexDomain::entire);
    const auto BoundX3 = MeshblockPointer->cellbounds.GetBoundsK(IndexDomain::entire);

    typename Kokkos::MinMax<Real>::value_type minmax;
    MeshblockPointer->par_reduce(
        PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i,
                        typename Kokkos::MinMax<Real>::value_type &lminmax) {
            lminmax.min_val =
                (QFactor(k, j, i) < lminmax.min_val ? QFactor(k, j, i) : lminmax.min_val);
            lminmax.max_val =
                (QFactor(k, j, i) > lminmax.max_val ? QFactor(k, j, i) : lminmax.max_val);
        },
        Kokkos::MinMax<Real>(minmax));

    if (minmax.max_val > QFactorCeiling && minmax.min_val < QFactorFloor) return AmrTag::refine;
    if (minmax.max_val < QFactorFloor) return AmrTag::derefine;
    return AmrTag::same;
}
