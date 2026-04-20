// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/constrained_transport module defines constrained_transport.cc
// responsibilities for the Pangu runtime. It centers on constrained_transport to express
// core data flow, keep interfaces readable, and preserve predictable behavior across task
// coordination, recovery paths, and performance-sensitive execution.

#include "constrained_transport/constrained_transport.h"

#include <memory>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "initialization/variable_mnemonics.h"

parthenon::TaskStatus ConstraintedTransport(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");

  const auto bound_x1_interior =
      pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
  const auto bound_x2_interior =
      pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
  const auto bound_x3_interior =
      pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);

  parthenon::PackIndexMap conservativeIndexMap;
  const std::vector<std::string> conservative_tags = {"conservative"};
  auto conservative =
      resource->PackVariablesAndFluxes(conservative_tags, conservativeIndexMap);
  parthenon::PackIndexMap electricFieldIndexMap;
  const std::vector<std::string> electric_field_tags = {"electric_field"};
  auto electricField =
      resource->PackVariables(electric_field_tags, electricFieldIndexMap);

  const int meshgrid_size_x1 =
      pmb->cellbounds.ncellsi(parthenon::IndexDomain::entire);
  const int meshgrid_size_x2 =
      pmb->cellbounds.ncellsj(parthenon::IndexDomain::entire);
  const int meshgrid_size_x3 =
      pmb->cellbounds.ncellsk(parthenon::IndexDomain::entire);

  const int offset_x1 = (meshgrid_size_x1 > 1) ? 1 : 0;
  const int offset_x2 = (meshgrid_size_x2 > 1) ? 1 : 0;
  const int offset_x3 = (meshgrid_size_x3 > 1) ? 1 : 0;

  const int CalculateElectricFieldX1 = offset_x2 && offset_x3;
  const int CalculateElectricFieldX2 = offset_x3 && offset_x1;
  const int CalculateElectricFieldX3 = offset_x1 && offset_x2;

  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_interior.s, bound_x3_interior.e + offset_x3, bound_x2_interior.s,
      bound_x2_interior.e + offset_x2, bound_x1_interior.s, bound_x1_interior.e + offset_x1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        electricField(Vector3D::X1, k, j, i) = 0.0;
        electricField(Vector3D::X2, k, j, i) = 0.0;
        electricField(Vector3D::X3, k, j, i) = 0.0;

        if (CalculateElectricFieldX1)
          electricField(Vector3D::X1, k, j, i) =
              0.25 * ((conservative.flux(X2DIR, BX3, k, j, i) +
                       conservative.flux(X2DIR, BX3, k - 1, j, i)) -
                      (conservative.flux(X3DIR, BX2, k, j, i) +
                       conservative.flux(X3DIR, BX2, k, j - 1, i)));
        if (CalculateElectricFieldX2)
          electricField(Vector3D::X2, k, j, i) =
              0.25 * ((conservative.flux(X3DIR, BX1, k, j, i) +
                       conservative.flux(X3DIR, BX1, k, j, i - 1)) -
                      (conservative.flux(X1DIR, BX3, k, j, i) +
                       conservative.flux(X1DIR, BX3, k - 1, j, i)));
        if (CalculateElectricFieldX3)
          electricField(Vector3D::X3, k, j, i) =
              0.25 * ((conservative.flux(X1DIR, BX2, k, j, i) +
                       conservative.flux(X1DIR, BX2, k, j - 1, i)) -
                      (conservative.flux(X2DIR, BX1, k, j, i) +
                       conservative.flux(X2DIR, BX1, k, j, i - 1)));
      });

  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_interior.s, bound_x3_interior.e, bound_x2_interior.s, bound_x2_interior.e,
      bound_x1_interior.s, bound_x1_interior.e + offset_x1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (offset_x1)
          conservative.flux(X1DIR, BX1, k, j, i) = 0;
        if (CalculateElectricFieldX2)
          conservative.flux(X1DIR, BX3, k, j, i) =
              -0.5 * (electricField(Vector3D::X2, k, j, i) +
                      electricField(Vector3D::X2, k + 1, j, i));
        if (CalculateElectricFieldX3)
          conservative.flux(X1DIR, BX2, k, j, i) =
              0.5 * (electricField(Vector3D::X3, k, j, i) +
                     electricField(Vector3D::X3, k, j + 1, i));
      });

  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_interior.s, bound_x3_interior.e, bound_x2_interior.s,
      bound_x2_interior.e + offset_x2, bound_x1_interior.s, bound_x1_interior.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (CalculateElectricFieldX1)
          conservative.flux(X2DIR, BX3, k, j, i) =
              0.5 * (electricField(Vector3D::X1, k, j, i) +
                     electricField(Vector3D::X1, k + 1, j, i));
        if (offset_x2)
          conservative.flux(X2DIR, BX2, k, j, i) = 0.;
        if (CalculateElectricFieldX3)
          conservative.flux(X2DIR, BX1, k, j, i) =
              -0.5 * (electricField(Vector3D::X3, k, j, i) +
                      electricField(Vector3D::X3, k, j, i + 1));
      });

  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_interior.s, bound_x3_interior.e + offset_x3, bound_x2_interior.s,
      bound_x2_interior.e, bound_x1_interior.s, bound_x1_interior.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (CalculateElectricFieldX1)
          conservative.flux(X3DIR, BX2, k, j, i) =
              -0.5 * (electricField(Vector3D::X1, k, j, i) +
                      electricField(Vector3D::X1, k, j + 1, i));
        if (CalculateElectricFieldX2)
          conservative.flux(X3DIR, BX1, k, j, i) =
              0.5 * (electricField(Vector3D::X2, k, j, i) +
                     electricField(Vector3D::X2, k, j, i + 1));
        if (offset_x3)
          conservative.flux(X3DIR, BX3, k, j, i) = 0;
      });

  return parthenon::TaskStatus::complete;
}
