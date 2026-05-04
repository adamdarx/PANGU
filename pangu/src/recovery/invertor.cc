// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "recovery/invertor.h"

#include <memory>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"
#include "recovery/constants.h"
#include "recovery/scheme_1d.h"
#include "recovery/scheme_1d_vsq.h"
#include "recovery/scheme_2d.h"

parthenon::TaskStatus Recovery(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource) {
  PARTHENON_INSTRUMENT

  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
  const auto kAdiabaticIndex =
      package_core->Param<parthenon::Real>("adiabatic_index");

  const auto bound_x1_interior =
      pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
  const auto bound_x2_interior =
      pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
  const auto bound_x3_interior =
      pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
      "electron_entropy"
  };
  auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);
  PackIndexMap conservativeIndexMap;
  const std::vector<std::string> conservative_tags = {"conservative"};
  const auto conservative =
      resource->PackVariables(conservative_tags, conservativeIndexMap);

  auto flag = resource->Get("flag").data;
  auto covariant_metric = geom_resource->Get("covariant_metric").data;
  auto contravariant_metric = geom_resource->Get("contravariant_metric").data;
  auto metric_determinant = geom_resource->Get("metric_determinant").data;

  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_interior.s, bound_x3_interior.e, bound_x2_interior.s, bound_x2_interior.e,
      bound_x1_interior.s, bound_x1_interior.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        parthenon::Real conservativeCArray[NPRIM_RECV],
            primitiveCArray[NPRIM_RECV];
        for (int index = 0; index < NPRIM_RECV; ++index) {
          conservativeCArray[index] = conservative(index, k, j, i);
          primitiveCArray[index] = primitive(index, k, j, i);
        }

        Real gcov[4][4], gcon[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcov[row][col] = covariant_metric(CENTER, col, row, k, j, i);
            gcon[row][col] = contravariant_metric(CENTER, col, row, k, j, i);
          }
        }
        Real gdet = metric_determinant(CENTER, k, j, i);

        flag(k, j, i) = 0;

        const parthenon::Real sqrt_abs_g = Kokkos::sqrt(Kokkos::fabs(gdet));
        const parthenon::Real alpha = 1.0 / Kokkos::sqrt(-gcon[0][0]);

        parthenon::Real conservativeHarm[NPRIM_RECV];
        parthenon::Real primitiveHarm[NPRIM_RECV];

        const parthenon::Real inv_sqrt_abs_g = 1.0 / sqrt_abs_g;
        const parthenon::Real alpha_over_sqrt_abs_g = alpha * inv_sqrt_abs_g;

        const parthenon::Real b1_prim =
            conservativeCArray[BX1] * inv_sqrt_abs_g;
        const parthenon::Real b2_prim =
            conservativeCArray[BX2] * inv_sqrt_abs_g;
        const parthenon::Real b3_prim =
            conservativeCArray[BX3] * inv_sqrt_abs_g;

        primitive(BX1, k, j, i) = b1_prim;
        primitive(BX2, k, j, i) = b2_prim;
        primitive(BX3, k, j, i) = b3_prim;

        conservativeHarm[RHO] = alpha_over_sqrt_abs_g * conservativeCArray[RHO];
        conservativeHarm[ENY] =
            alpha_over_sqrt_abs_g *
            (conservativeCArray[ENY] - conservativeCArray[RHO]);
        conservativeHarm[UX1] = alpha_over_sqrt_abs_g * conservativeCArray[UX1];
        conservativeHarm[UX2] = alpha_over_sqrt_abs_g * conservativeCArray[UX2];
        conservativeHarm[UX3] = alpha_over_sqrt_abs_g * conservativeCArray[UX3];
        conservativeHarm[BX1] = alpha_over_sqrt_abs_g * conservativeCArray[BX1];
        conservativeHarm[BX2] = alpha_over_sqrt_abs_g * conservativeCArray[BX2];
        conservativeHarm[BX3] = alpha_over_sqrt_abs_g * conservativeCArray[BX3];

        primitiveHarm[RHO] = primitiveCArray[RHO];
        primitiveHarm[ENY] = primitiveCArray[ENY];
        primitiveHarm[UX1] = primitiveCArray[UX1];
        primitiveHarm[UX2] = primitiveCArray[UX2];
        primitiveHarm[UX3] = primitiveCArray[UX3];
        primitiveHarm[BX1] = alpha * b1_prim;
        primitiveHarm[BX2] = alpha * b2_prim;
        primitiveHarm[BX3] = alpha * b3_prim;

        flag(k, j, i) =
            Scheme2D::invert(conservativeHarm, primitiveHarm, kAdiabaticIndex,
                             gcov, gcon, gdet) == 0;
        if (!flag(k, j, i)) {
          flag(k, j, i) =
              Scheme1Dvsq::invert(conservativeHarm, primitiveHarm,
                                  kAdiabaticIndex, gcov, gcon, gdet) == 0;
          if (!flag(k, j, i)) {
            flag(k, j, i) =
                Scheme1D::invert(conservativeHarm, primitiveHarm,
                                 kAdiabaticIndex, gcov, gcon, gdet) == 0;
          }
        }
        for (int index = 0; index < BX1; ++index)
          primitive(index, k, j, i) = primitiveHarm[index];
        primitive(ENT, k, j, i) = (kAdiabaticIndex - 1.0) * primitive(ENY, k, j, i) * Kokkos::pow(primitive(RHO, k, j, i), -kAdiabaticIndex);
      });

  return parthenon::TaskStatus::complete;
}
