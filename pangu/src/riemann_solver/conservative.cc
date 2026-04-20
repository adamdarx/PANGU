// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/riemann_solver module defines conservative.cc responsibilities for
// the Pangu runtime. It centers on riemann_solver to express core data flow, keep
// interfaces readable, and preserve predictable behavior across task coordination, recovery
// paths, and performance-sensitive execution.

#include "riemann_solver/conservative.h"

#include <memory>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "initialization/variable_mnemonics.h"
#include "physics_transformation/contravariant_flux.h"

parthenon::TaskStatus CalculateConservativeSRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
  const auto kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");

  const auto bound_x1_interior =
      pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto bound_x2_interior =
      pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto bound_x3_interior =
      pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field"};
  const auto primitive =
      resource->PackVariables(primitive_tags, primitiveIndexMap);
  PackIndexMap conservativeIndexMap;
  const std::vector<std::string> conservative_tags = {"conservative"};
  auto conservative =
      resource->PackVariablesAndFluxes(conservative_tags, conservativeIndexMap);

  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_interior.s, bound_x3_interior.e, bound_x2_interior.s, bound_x2_interior.e,
      bound_x1_interior.s, bound_x1_interior.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real primitive_c_array[NPRIM] = {
            primitive(RHO, k, j, i), primitive(ENY, k, j, i),
            primitive(UX1, k, j, i), primitive(UX2, k, j, i),
            primitive(UX3, k, j, i), primitive(BX1, k, j, i),
            primitive(BX2, k, j, i), primitive(BX3, k, j, i),
        };

        Real ConservativeState[NPRIM];
        CalculateContravariantFluxSRMHD(kAdiabaticIndex, primitive_c_array, X0DIR,
                                        ConservativeState);

        for (int index = 0; index < NPRIM; ++index) {
          conservative(index, k, j, i) = ConservativeState[index];
        }
      });

  return TaskStatus::complete;
}

parthenon::TaskStatus CalculateConservativeGRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
  const auto package_metric = pmb->packages.Get("metric");
  const auto kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");

  const auto bound_x1_interior =
      pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto bound_x2_interior =
      pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto bound_x3_interior =
      pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field"};
  const auto primitive =
      resource->PackVariables(primitive_tags, primitiveIndexMap);

  PackIndexMap conservativeIndexMap;
  const std::vector<std::string> conservative_tags = {"conservative"};
  auto conservative =
      resource->PackVariablesAndFluxes(conservative_tags, conservativeIndexMap);

  auto covariant_metric = geom_resource->Get("covariant_metric").data;
  auto contravariant_metric = geom_resource->Get("contravariant_metric").data;
  auto metric_determinant = geom_resource->Get("metric_determinant").data;

  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_interior.s, bound_x3_interior.e, bound_x2_interior.s, bound_x2_interior.e,
      bound_x1_interior.s, bound_x1_interior.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real gcov[4][4];
        Real gcon[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcov[row][col] = covariant_metric(CENTER, col, row, k, j, i);
            gcon[row][col] = contravariant_metric(CENTER, col, row, k, j, i);
          }
        }

        const Real primitive_c_array[NPRIM] = {
            primitive(RHO, k, j, i), primitive(ENY, k, j, i),
            primitive(UX1, k, j, i), primitive(UX2, k, j, i),
            primitive(UX3, k, j, i), primitive(BX1, k, j, i),
            primitive(BX2, k, j, i), primitive(BX3, k, j, i),
        };

        Real conservative_c_array[NPRIM];
        CalculateContravariantFluxGRMHD(
            kAdiabaticIndex, primitive_c_array, gcov, gcon,
            metric_determinant(CENTER, k, j, i), X0DIR, conservative_c_array);

        for (int index = 0; index < NPRIM; ++index) {
          conservative(index, k, j, i) = conservative_c_array[index];
        }
      });

  return TaskStatus::complete;
}
