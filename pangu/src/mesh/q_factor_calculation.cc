// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/mesh module defines q_factor_calculation.cc responsibilities for the
// Pangu runtime. It centers on mesh to express core data flow, keep interfaces readable,
// and preserve predictable behavior across task coordination, recovery paths, and
// performance-sensitive execution.

#include "mesh/q_factor_calculation.h"

#include <memory>
#include <vector>

#include "globals.hpp"
#include "initialization/variable_mnemonics.h"

parthenon::TaskStatus CalculateQFactor(
        std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
  using namespace parthenon::package::prelude;

  PARTHENON_INSTRUMENT

  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
    const auto kAdiabaticIndex =
            package_core->Param<parthenon::Real>("adiabatic_index");

  const auto bound_x1_interior =
      pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto bound_x2_interior =
      pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto bound_x3_interior =
      pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto &Coords = pmb->coords;

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field"};
  const auto primitive =
      resource->PackVariables(primitive_tags, primitiveIndexMap);
  PackIndexMap qFactorIndexMap;
  const std::vector<std::string> q_factor_tags = {"q_factor"};
  auto q_factor = resource->PackVariables(q_factor_tags, qFactorIndexMap);

  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_interior.s, bound_x3_interior.e, bound_x2_interior.s, bound_x2_interior.e,
      bound_x1_interior.s, bound_x1_interior.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto SquaredWeightedVelocity =
            Kokkos::pow(primitive(UX1, k, j, i), 2) +
            Kokkos::pow(primitive(UX2, k, j, i), 2) +
            Kokkos::pow(primitive(UX3, k, j, i), 2);
        const auto SquaredLorentzFactor = 1 + SquaredWeightedVelocity;
        const auto LorentzFactor = Kokkos::sqrt(1 + SquaredWeightedVelocity);
        const auto SquaredMagneticFieldThreeVector =
            Kokkos::pow(primitive(BX1, k, j, i), 2) +
            Kokkos::pow(primitive(BX2, k, j, i), 2) +
            Kokkos::pow(primitive(BX3, k, j, i), 2);
        const auto MagneticFieldThreeVectorDotWeightedVelocity =
            primitive(UX1, k, j, i) * primitive(BX1, k, j, i) +
            primitive(UX2, k, j, i) * primitive(BX2, k, j, i) +
            primitive(UX3, k, j, i) * primitive(BX3, k, j, i);
        const auto SquaredMagneticFieldFourVector =
            (SquaredMagneticFieldThreeVector +
             Kokkos::pow(MagneticFieldThreeVectorDotWeightedVelocity, 2)) /
            SquaredLorentzFactor;
        const auto Enthalpy =
            primitive(ENY, k, j, i) + kAdiabaticIndex * primitive(RHO, k, j, i);
        const auto Energy = SquaredMagneticFieldFourVector + Enthalpy;

        q_factor(0, k, j, i) = 2 * M_PI *
                              (LorentzFactor * primitive(BX2, k, j, i) +
                               MagneticFieldThreeVectorDotWeightedVelocity *
                                   primitive(UX2, k, j, i)) /
                              (LorentzFactor * primitive(UX2, k, j, i) *
                               Kokkos::sqrt(Energy) * Coords.Dxc<X2DIR>());
      });
    return parthenon::TaskStatus::complete;
}
