// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/mesh module defines refinement_criteria.cc responsibilities for the
// Pangu runtime. It centers on mesh to express core data flow, keep interfaces readable,
// and preserve predictable behavior across task coordination, recovery paths, and
// performance-sensitive execution.

#include "mesh/refinement_criteria.h"

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

AmrTag CheckRefinement(MeshBlockData<Real> *resource) {
  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
  const auto kQFactorFloor = package_core->Param<Real>("q_factor_floor");
  const auto kQFactorCeiling = package_core->Param<Real>("q_factor_ceiling");
  const auto q_factor = resource->Get("q_factor").data;

  const auto bound_x1_entire =
      pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto bound_x2_entire =
      pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto bound_x3_entire =
      pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  typename Kokkos::MinMax<Real>::value_type minmax;
  pmb->par_reduce(
      PARTHENON_AUTO_LABEL, bound_x3_entire.s, bound_x3_entire.e, bound_x2_entire.s, bound_x2_entire.e,
      bound_x1_entire.s, bound_x1_entire.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i,
                    typename Kokkos::MinMax<Real>::value_type &lminmax) {
        lminmax.min_val =
            (q_factor(k, j, i) < lminmax.min_val ? q_factor(k, j, i)
                                                : lminmax.min_val);
        lminmax.max_val =
            (q_factor(k, j, i) > lminmax.max_val ? q_factor(k, j, i)
                                                : lminmax.max_val);
      },
      Kokkos::MinMax<Real>(minmax));

  if (minmax.max_val > kQFactorCeiling && minmax.min_val < kQFactorFloor)
    return AmrTag::refine;
  if (minmax.max_val < kQFactorFloor) return AmrTag::derefine;
  return AmrTag::same;
}
