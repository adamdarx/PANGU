// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "fixer/recovery_fixer.h"

#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"

parthenon::TaskStatus FixRecovery(parthenon::MeshData<parthenon::Real> *md) {
  using namespace parthenon::package::prelude;
  PARTHENON_INSTRUMENT
  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  const auto pkg_met = pmb0->packages.Get("metric");
  const auto met_type = pkg_met->Param<std::string>("metric_type");
  const bool is_cks = (met_type == "cks");
  const Real r_exc = pkg_met->Param<Real>("r_excise");
  const Real ka = pkg_met->Param<Real>("a");
  const Real ka2 = ka * ka;
  auto block = IndexRange{0, md->NumBlocks() - 1};

  const auto bound_x1_interior = md->GetBoundsI(IndexDomain::interior);
  const auto bound_x2_interior = md->GetBoundsJ(IndexDomain::interior);
  const auto bound_x3_interior = md->GetBoundsK(IndexDomain::interior);

  const auto package_core = pmb0->packages.Get("core");
  const auto enable_B = package_core->Param<bool>("enable_B");
  const auto enable_heating = package_core->Param<bool>("enable_heating");
  const auto& fnames = package_core->Param<std::vector<std::string>>("primitive_field_names");

  PackIndexMap idxMap;
  auto primitive = md->PackVariables(fnames, idxMap);

  const int iRHO = idxMap["density"].first;
  const int iENY = idxMap["energy"].first;
  const int iUX  = idxMap["weighted_velocity"].first;
  const int iENT = idxMap["entropy"].first;
  const int iBX  = enable_B ? idxMap["magnetic_field"].first : -1;
  const int iKEL = enable_heating ? idxMap["electron_entropy"].first : -1;

  PackIndexMap flagIndexMap;
  auto flag = md->PackVariables(std::vector<std::string>{"flag"}, flagIndexMap);

  const int offset_x2 = (bound_x2_interior.s != bound_x2_interior.e) ? 1 : 0;
  pmb0->par_for(
      PARTHENON_AUTO_LABEL, block.s, block.e,
      bound_x3_interior.s, bound_x3_interior.e, bound_x2_interior.s, bound_x2_interior.e,
      bound_x1_interior.s, bound_x1_interior.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        if (flag(b, 0, k, j, i) == 0) {
          if (is_cks) {
            const auto &cr = primitive.GetCoords(b);
            const Real xr=cr.Xc<X1DIR>(i),yr=cr.Xc<X2DIR>(j),zr=cr.Xc<X3DIR>(k);
            const Real rad2=xr*xr+yr*yr+zr*zr;
            const Real sa=SQR(rad2-ka2)+4.0*ka2*zr*zr;
            if (Kokkos::sqrt(0.5*(rad2-ka2+Kokkos::sqrt(sa))) < r_exc) return;
          }
          int pf1 = flag(b, 0, k, j + offset_x2, i - 1);
          int pf2 = flag(b, 0, k, j + offset_x2, i);
          int pf3 = flag(b, 0, k, j + offset_x2, i + 1);
          int pf4 = flag(b, 0, k, j, i + 1);
          int pf5 = flag(b, 0, k, j - offset_x2, i + 1);
          int pf6 = flag(b, 0, k, j - offset_x2, i);
          int pf7 = flag(b, 0, k, j - offset_x2, i - 1);
          int pf8 = flag(b, 0, k, j, i - 1);

          // Helper lambda to interpolate a single pack component
          auto fix = [&](int pi) {
            if (pf2 && pf4 && pf6 && pf8) {
              primitive(b, pi, k, j, i) =
                  0.25 * (primitive(b, pi, k, j + offset_x2, i) +
                          primitive(b, pi, k, j - offset_x2, i) +
                          primitive(b, pi, k, j, i - 1) +
                          primitive(b, pi, k, j, i + 1));
            } else if (pf1 && pf3 && pf5 && pf7) {
              primitive(b, pi, k, j, i) =
                  0.25 * (primitive(b, pi, k, j + offset_x2, i + 1) +
                          primitive(b, pi, k, j + offset_x2, i - 1) +
                          primitive(b, pi, k, j - offset_x2, i + 1) +
                          primitive(b, pi, k, j - offset_x2, i - 1));
            } else {
              primitive(b, pi, k, j, i) =
                  0.125 * (primitive(b, pi, k, j + offset_x2, i - 1) +
                           primitive(b, pi, k, j + offset_x2, i) +
                           primitive(b, pi, k, j + offset_x2, i + 1) +
                           primitive(b, pi, k, j, i + 1) +
                           primitive(b, pi, k, j - offset_x2, i + 1) +
                           primitive(b, pi, k, j - offset_x2, i) +
                           primitive(b, pi, k, j - offset_x2, i - 1) +
                           primitive(b, pi, k, j, i - 1));
            }
          };

          fix(iRHO);
          fix(iENY);
          fix(iUX);
          fix(iUX+1);
          fix(iUX+2);
          if (enable_B) {
            fix(iBX);
            fix(iBX+1);
            fix(iBX+2);
          }
          fix(iENT);
          if (enable_heating) {
            fix(iKEL);
          }

          primitive(b, iUX,   k, j, i) = 0.;
          primitive(b, iUX+1, k, j, i) = 0.;
          primitive(b, iUX+2, k, j, i) = 0.;
          flag(b, 0, k, j, i) = 1;
        }
      });

  return parthenon::TaskStatus::complete;
}
