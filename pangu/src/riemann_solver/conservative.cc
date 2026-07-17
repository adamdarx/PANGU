// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "riemann_solver/conservative.h"

#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include <basic_types.hpp>
#include "initialization/variable_mnemonics.h"
#include "metric/christoffel.h"
#include "physics/contravariant_flux.h"

parthenon::TaskStatus CalculateConservative(
    parthenon::MeshData<parthenon::Real> *md) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  const auto package_core = pmb0->packages.Get("core");
  const auto kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");
  const auto enable_B = package_core->Param<bool>("enable_B");
  const auto enable_heating = package_core->Param<bool>("enable_heating");
  const auto& fnames = package_core->Param<std::vector<std::string>>("primitive_field_names");

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
  auto conservative =
      md->PackVariablesAndFluxes(conservative_tags, conservativeIndexMap);

  // Metric type and parameters for on-the-fly metric computation
  const auto package_metric = pmb0->packages.Get("metric");
  const auto metric_type_str = package_metric->Param<std::string>("metric_type");
  int mtype_int = MetricType::Minkowski;
  if (metric_type_str == "bl") { mtype_int = MetricType::BL; }
  else if (metric_type_str == "cks") { mtype_int = MetricType::CKS; }
  else if (metric_type_str == "mks") { mtype_int = MetricType::MKS; }
  const Real kerr_a = package_metric->Param<Real>("a");
  const Real mks_h = package_metric->Param<Real>("h");

  pmb0->par_for(
      PARTHENON_AUTO_LABEL, block.s, block.e,
      bound_x3_interior.s, bound_x3_interior.e,
      bound_x2_interior.s, bound_x2_interior.e,
      bound_x1_interior.s, bound_x1_interior.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = primitive.GetCoords(b);
        Real gcov[4][4];
        Real gcon[4][4];
        Real gdet;
        const Real x_code[4] = {0.0, coords.Xc<X1DIR>(i), coords.Xc<X2DIR>(j),
                                coords.Xc<X3DIR>(k)};
        ComputeMetricAtLocation(mtype_int, x_code, kerr_a, mks_h, gcov, gcon, gdet);

        Real pcarr[NPRIM] = {0};
        pcarr[RHO] = primitive(b, iRHO, k, j, i);
        pcarr[ENY] = primitive(b, iENY, k, j, i);
        pcarr[UX1] = primitive(b, iUX,   k, j, i);
        pcarr[UX2] = primitive(b, iUX+1, k, j, i);
        pcarr[UX3] = primitive(b, iUX+2, k, j, i);
        if (enable_B) {
          pcarr[BX1] = primitive(b, iBX,   k, j, i);
          pcarr[BX2] = primitive(b, iBX+1, k, j, i);
          pcarr[BX3] = primitive(b, iBX+2, k, j, i);
        }
        pcarr[ENT] = primitive(b, iENT, k, j, i);
        if (enable_heating) {
          pcarr[KEL] = primitive(b, iKEL, k, j, i);
        }

        Real ccarr[NPRIM];
        CalculateContravariantFlux(
            kAdiabaticIndex, pcarr, gcov, gcon,
            gdet, X0DIR, ccarr);

        int c = 0;
        conservative(b, c++, k, j, i) = ccarr[RHO];
        conservative(b, c++, k, j, i) = ccarr[ENY];
        conservative(b, c++, k, j, i) = ccarr[UX1];
        conservative(b, c++, k, j, i) = ccarr[UX2];
        conservative(b, c++, k, j, i) = ccarr[UX3];
        if (enable_B) {
          conservative(b, c++, k, j, i) = ccarr[BX1];
          conservative(b, c++, k, j, i) = ccarr[BX2];
          conservative(b, c++, k, j, i) = ccarr[BX3];
        }
        conservative(b, c++, k, j, i) = ccarr[ENT];
        if (enable_heating) {
          conservative(b, c++, k, j, i) = ccarr[KEL];
        }
      });

  return TaskStatus::complete;
}
