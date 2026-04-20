// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/numerical_correction module defines recovery_fixer.cc
// responsibilities for the Pangu runtime. It centers on numerical_correction to express
// core data flow, keep interfaces readable, and preserve predictable behavior across task
// coordination, recovery paths, and performance-sensitive execution.

#include "numerical_correction/recovery_fixer.h"

#include <memory>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"

parthenon::TaskStatus FixRecovery(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
  using namespace parthenon::package::prelude;
  PARTHENON_INSTRUMENT
  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");

  auto &flag = resource->Get("flag").data;

  const auto bound_x1_interior =
      pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
  const auto bound_x2_interior =
      pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
  const auto bound_x3_interior =
      pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);

  parthenon::PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field"};
  auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);

  const int offset_x2 = (bound_x2_interior.s != bound_x2_interior.e) ? 1 : 0;
  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_interior.s, bound_x3_interior.e, bound_x2_interior.s, bound_x2_interior.e,
      bound_x1_interior.s, bound_x1_interior.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (flag(k, j, i) == 0) {
          int pf1 = flag(k, j + offset_x2, i - 1);
          int pf2 = flag(k, j + offset_x2, i);
          int pf3 = flag(k, j + offset_x2, i + 1);
          int pf4 = flag(k, j, i + 1);
          int pf5 = flag(k, j - offset_x2, i + 1);
          int pf6 = flag(k, j - offset_x2, i);
          int pf7 = flag(k, j - offset_x2, i - 1);
          int pf8 = flag(k, j, i - 1);

          if (pf2 && pf4 && pf6 && pf8) {
            for (int m = 0; m < NPRIM; m++)
              primitive(m, k, j, i) = 0.25 * (primitive(m, k, j + offset_x2, i) +
                                              primitive(m, k, j - offset_x2, i) +
                                              primitive(m, k, j, i - 1) +
                                              primitive(m, k, j, i + 1));
          } else if (pf1 && pf3 && pf5 && pf7) {
            for (int m = 0; m < NPRIM; m++)
              primitive(m, k, j, i) =
                  0.25 * (primitive(m, k, j + offset_x2, i + 1) +
                          primitive(m, k, j + offset_x2, i - 1) +
                          primitive(m, k, j - offset_x2, i + 1) +
                          primitive(m, k, j - offset_x2, i - 1));
          } else {
            for (int m = 0; m < NPRIM; m++)
              primitive(m, k, j, i) =
                  0.125 * (primitive(m, k, j + offset_x2, i - 1) +
                           primitive(m, k, j + offset_x2, i) +
                           primitive(m, k, j + offset_x2, i + 1) +
                           primitive(m, k, j, i + 1) +
                           primitive(m, k, j - offset_x2, i + 1) +
                           primitive(m, k, j - offset_x2, i) +
                           primitive(m, k, j - offset_x2, i - 1) +
                           primitive(m, k, j, i - 1));
          }
          primitive(UX1, k, j, i) = 0.;
          primitive(UX2, k, j, i) = 0.;
          primitive(UX3, k, j, i) = 0.;
          flag(k, j, i) = 1;
        }
      });

  return parthenon::TaskStatus::complete;
}
