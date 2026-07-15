// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "recovery/invertor.h"

#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"
#include "recovery/constants.h"
#include "recovery/scheme_1d.h"
#include "recovery/scheme_1d_vsq.h"
#include "recovery/scheme_2d.h"

parthenon::TaskStatus Recovery(parthenon::MeshData<parthenon::Real> *md) {
  PARTHENON_INSTRUMENT

  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  const auto package_core = pmb0->packages.Get("core");
  const auto package_metric_inv = pmb0->packages.Get("metric");
  const auto kAdiabaticIndex =
      package_core->Param<parthenon::Real>("adiabatic_index");
  const auto enable_B = package_core->Param<bool>("enable_B");
  const auto enable_heating = package_core->Param<bool>("enable_heating");
  const auto& fnames = package_core->Param<std::vector<std::string>>("primitive_field_names");
  const auto met_type_inv = package_metric_inv->Param<std::string>("metric_type");
  const bool is_cks_inv = (met_type_inv == "cks");
  const Real r_excise_inv = package_metric_inv->Param<Real>("r_excise");
  const Real dexcise_inv = package_metric_inv->Param<Real>("dexcise");
  const Real pexcise_inv = package_metric_inv->Param<Real>("pexcise");
  const Real e_excise_inv = pexcise_inv / (kAdiabaticIndex - 1.0);
  const Real kerr_a_inv = package_metric_inv->Param<Real>("a");
  const Real kerr_a2_inv = kerr_a_inv * kerr_a_inv;

  const auto bound_x1_interior = md->GetBoundsI(IndexDomain::interior);
  const auto bound_x2_interior = md->GetBoundsJ(IndexDomain::interior);
  const auto bound_x3_interior = md->GetBoundsK(IndexDomain::interior);
  auto block = IndexRange{0, md->NumBlocks() - 1};

  PackIndexMap idxMap;
  auto primitive = md->PackVariables(fnames, idxMap);

  const int iRHO = idxMap["density"].first;
  const int iENY = idxMap["energy"].first;
  const int iUX  = idxMap["weighted_velocity"].first;
  const int iENT = idxMap["entropy"].first;
  const int iBX  = enable_B ? idxMap["magnetic_field"].first : -1;
  const int iKEL = enable_heating ? idxMap["electron_entropy"].first : -1;

  PackIndexMap conservativeIndexMap;
  const std::vector<std::string> conservative_tags = {"conservative"};
  const auto conservative =
      md->PackVariables(conservative_tags, conservativeIndexMap);

  auto flag =
      md->PackVariables(std::vector<std::string>{"flag"});
  auto covariant_metric =
      md->PackVariables(std::vector<std::string>{"covariant_metric"});
  auto contravariant_metric =
      md->PackVariables(std::vector<std::string>{"contravariant_metric"});
  auto metric_determinant =
      md->PackVariables(std::vector<std::string>{"metric_determinant"});

  pmb0->par_for(
      PARTHENON_AUTO_LABEL, block.s, block.e,
      bound_x3_interior.s, bound_x3_interior.e, bound_x2_interior.s, bound_x2_interior.e,
      bound_x1_interior.s, bound_x1_interior.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        if (is_cks_inv) {
          const auto &coords_inv = primitive.GetCoords(b);
          const Real x = coords_inv.Xc<X1DIR>(i);
          const Real y = coords_inv.Xc<X2DIR>(j);
          const Real z = coords_inv.Xc<X3DIR>(k);
          const Real rad2 = x*x + y*y + z*z;
          const Real sa = SQR(rad2 - kerr_a2_inv) + 4.0*kerr_a2_inv*z*z;
          const Real r_bl = Kokkos::sqrt(0.5*(rad2 - kerr_a2_inv + Kokkos::sqrt(sa)));
          if (r_bl < r_excise_inv) { flag(b, 0, k, j, i) = 0; return; }
        }

        Real gcov[4][4], gcon[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcov[row][col] =
                covariant_metric(b, CENTER * 16 + col * 4 + row, k, j, i);
            gcon[row][col] =
                contravariant_metric(b, CENTER * 16 + col * 4 + row, k, j, i);
          }
        }
        Real gdet = metric_determinant(b, CENTER, k, j, i);

        flag(b, 0, k, j, i) = 0;

        const parthenon::Real sqrt_abs_g = Kokkos::sqrt(Kokkos::fabs(gdet));
        const parthenon::Real alpha = 1.0 / Kokkos::sqrt(-gcon[0][0]);

        // Load conservative C array from pack (dynamic layout)
        Real conservativeCArray[NPRIM_RECV] = {0};
        int c = 0;
        conservativeCArray[RHO] = conservative(b, c++, k, j, i);
        conservativeCArray[ENY] = conservative(b, c++, k, j, i);
        conservativeCArray[UX1] = conservative(b, c++, k, j, i);
        conservativeCArray[UX2] = conservative(b, c++, k, j, i);
        conservativeCArray[UX3] = conservative(b, c++, k, j, i);
        if (enable_B) {
          conservativeCArray[BX1] = conservative(b, c++, k, j, i);
          conservativeCArray[BX2] = conservative(b, c++, k, j, i);
          conservativeCArray[BX3] = conservative(b, c++, k, j, i);
        }
        conservativeCArray[ENT] = conservative(b, c++, k, j, i);

        // Load primitive C array from pack (dynamic layout)
        Real primitiveCArray[NPRIM_RECV] = {0};
        int p = 0;
        primitiveCArray[RHO] = primitive(b, p++, k, j, i);
        primitiveCArray[ENY] = primitive(b, p++, k, j, i);
        primitiveCArray[UX1] = primitive(b, p++, k, j, i);
        primitiveCArray[UX2] = primitive(b, p++, k, j, i);
        primitiveCArray[UX3] = primitive(b, p++, k, j, i);
        if (enable_B) {
          primitiveCArray[BX1] = primitive(b, p++, k, j, i);
          primitiveCArray[BX2] = primitive(b, p++, k, j, i);
          primitiveCArray[BX3] = primitive(b, p++, k, j, i);
        }
        primitiveCArray[ENT] = primitive(b, p++, k, j, i);

        const parthenon::Real inv_sqrt_abs_g = 1.0 / sqrt_abs_g;
        const parthenon::Real alpha_over_sqrt_abs_g = alpha * inv_sqrt_abs_g;

        // Extract B field and write to primitive pack
        if (enable_B) {
          const parthenon::Real b1_prim =
              conservativeCArray[BX1] * inv_sqrt_abs_g;
          const parthenon::Real b2_prim =
              conservativeCArray[BX2] * inv_sqrt_abs_g;
          const parthenon::Real b3_prim =
              conservativeCArray[BX3] * inv_sqrt_abs_g;
          primitive(b, iBX,   k, j, i) = b1_prim;
          primitive(b, iBX+1, k, j, i) = b2_prim;
          primitive(b, iBX+2, k, j, i) = b3_prim;
        }

        Real conservativeHarm[NPRIM_RECV];
        Real primitiveHarm[NPRIM_RECV];

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
        conservativeHarm[ENT] = alpha_over_sqrt_abs_g * conservativeCArray[ENT];

        primitiveHarm[RHO] = primitiveCArray[RHO];
        primitiveHarm[ENY] = primitiveCArray[ENY];
        primitiveHarm[UX1] = primitiveCArray[UX1];
        primitiveHarm[UX2] = primitiveCArray[UX2];
        primitiveHarm[UX3] = primitiveCArray[UX3];
        if (enable_B) {
          primitiveHarm[BX1] = alpha * conservativeCArray[BX1] * inv_sqrt_abs_g;
          primitiveHarm[BX2] = alpha * conservativeCArray[BX2] * inv_sqrt_abs_g;
          primitiveHarm[BX3] = alpha * conservativeCArray[BX3] * inv_sqrt_abs_g;
        }
        primitiveHarm[ENT] = primitiveCArray[ENT];

        flag(b, 0, k, j, i) =
            Scheme2D::invert(conservativeHarm, primitiveHarm, kAdiabaticIndex,
                             gcov, gcon, gdet) == 0;
        if (!flag(b, 0, k, j, i)) {
          flag(b, 0, k, j, i) =
              Scheme1Dvsq::invert(conservativeHarm, primitiveHarm,
                                  kAdiabaticIndex, gcov, gcon, gdet) == 0;
          if (!flag(b, 0, k, j, i)) {
            flag(b, 0, k, j, i) =
                Scheme1D::invert(conservativeHarm, primitiveHarm,
                                 kAdiabaticIndex, gcov, gcon, gdet) == 0;
          }
        }

        // Write back fluid variables (indices up to UX3) using smart indices
        primitive(b, iRHO, k, j, i) = primitiveHarm[RHO];
        primitive(b, iENY, k, j, i) = primitiveHarm[ENY];
        primitive(b, iUX,   k, j, i) = primitiveHarm[UX1];
        primitive(b, iUX+1, k, j, i) = primitiveHarm[UX2];
        primitive(b, iUX+2, k, j, i) = primitiveHarm[UX3];
        // ENT is written by the scheme itself
      });

  return parthenon::TaskStatus::complete;
}
