// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "fixer/primitive_fixer.h"

#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"
#include "metric/christoffel.h"
#include "physics/state_calculation.h"
#include "physics/heating_model.h"

parthenon::TaskStatus FixPrimitive(parthenon::MeshData<parthenon::Real> *md) {
  using namespace parthenon::package::prelude;
  PARTHENON_INSTRUMENT
  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  const auto package_core = pmb0->packages.Get("core");
  const auto package_metric = pmb0->packages.Get("metric");
  const auto kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");
  const auto density_floor = package_core->Param<Real>("density_floor");
  const auto density_floor_pow = package_core->Param<Real>("density_floor_pow");
  const auto energy_floor = package_core->Param<Real>("energy_floor");
  const auto energy_floor_pow = package_core->Param<Real>("energy_floor_pow");
  const auto sigma_max = package_core->Param<Real>("sigma_max");
  const auto lorentz_max = package_core->Param<Real>("lorentz_max");
  const RatioLimits ratio = {
      package_core->Param<Real>("ratio_min"), package_core->Param<Real>("ratio_max")};
  const auto enable_B = package_core->Param<bool>("enable_B");
  const auto enable_heating = package_core->Param<bool>("enable_heating");
  const auto& fnames = package_core->Param<std::vector<std::string>>("primitive_field_names");

  const auto metric_type_str = package_metric->Param<std::string>("metric_type");
  int mtype_int = MetricType::Minkowski;
  if (metric_type_str == "bl") { mtype_int = MetricType::BL; }
  else if (metric_type_str == "cks") { mtype_int = MetricType::CKS; }
  else if (metric_type_str == "mks") { mtype_int = MetricType::MKS; }
  const auto kerr_a = package_metric->Param<Real>("a");
  const auto mks_h = package_metric->Param<Real>("h");
  const Real kerr_a2 = kerr_a * kerr_a;
  const Real r_excise = package_metric->Param<Real>("r_excise");
  const Real dexcise = package_metric->Param<Real>("dexcise");
  const Real pexcise = package_metric->Param<Real>("pexcise");
  const Real e_excise = pexcise / (kAdiabaticIndex - 1.0);

  const auto bound_x1_entire = md->GetBoundsI(IndexDomain::entire);
  const auto bound_x2_entire = md->GetBoundsJ(IndexDomain::entire);
  const auto bound_x3_entire = md->GetBoundsK(IndexDomain::entire);
  auto block = IndexRange{0, md->NumBlocks() - 1};

  PackIndexMap idxMap;
  auto primitive = md->PackVariables(fnames, idxMap);

  const int iRHO = idxMap["density"].first;
  const int iENY = idxMap["energy"].first;
  const int iUX  = idxMap["weighted_velocity"].first;
  const int iENT = idxMap["entropy"].first;
  const int iBX  = enable_B ? idxMap["magnetic_field"].first : -1;
  const int iKEL = enable_heating ? idxMap["electron_entropy"].first : -1;

  const Real umax2 = lorentz_max * lorentz_max - 1.0;

  pmb0->par_for(
      PARTHENON_AUTO_LABEL, block.s, block.e,
      bound_x3_entire.s, bound_x3_entire.e, bound_x2_entire.s, bound_x2_entire.e,
      bound_x1_entire.s, bound_x1_entire.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        constexpr Real small = 1.0e-20;
        const Real sigma_max_safe = Kokkos::max(sigma_max, small);
        const Real rho_before_floor = Kokkos::max(primitive(b, iRHO, k, j, i), small);

        const auto &coords = primitive.GetCoords(b);
        const auto x = coords.Xc<X1DIR>(i);
        const auto y = coords.Xc<X2DIR>(j);
        const auto z = coords.Xc<X3DIR>(k);
        const Real x_code[4] = {0.0, x, y, z};

        Real r = 1.0;
        if (mtype_int == MetricType::MKS) {
          r = Kokkos::exp(x);
        } else if (mtype_int == MetricType::CKS) {
          const Real rad2 = x * x + y * y + z * z;
          const Real sqrtarg = SQR(rad2 - kerr_a2) + 4.0 * kerr_a2 * z * z;
          r = Kokkos::sqrt(0.5 * (rad2 - kerr_a2 + Kokkos::sqrt(sqrtarg)));
        }

        Real rho_floor = density_floor * Kokkos::pow(Kokkos::fmax(r, 1.0), density_floor_pow);
        const Real eng_floor = energy_floor * Kokkos::pow(Kokkos::fmax(r, 1.0), energy_floor_pow);

        Real gcov[4][4], gcon[4][4];
        Real gdet;
        ComputeMetricAtLocation(mtype_int, x_code, kerr_a, mks_h, gcov, gcon, gdet);

        const Real rho = primitive(b, iRHO, k, j, i);
        const Real ug = primitive(b, iENY, k, j, i);
        Real pcarr[NPRIM] = {0};
        pcarr[RHO] = rho;
        pcarr[ENY] = ug;
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
        State state;
        CalculateState(pcarr, gcov, gcon, state);

        if (rho_floor < ug / sigma_max_safe) {
          rho_floor = ug / sigma_max_safe;
        }
        const Real sigma = state.bsq / rho;
        if (sigma > sigma_max) {
          rho_floor = state.bsq / sigma_max_safe;
        }

        if (primitive(b, iRHO, k, j, i) < rho_floor)
          primitive(b, iRHO, k, j, i) = rho_floor;
        if (primitive(b, iENY, k, j, i) < eng_floor)
          primitive(b, iENY, k, j, i) = eng_floor;

        const Real u1 = primitive(b, iUX,   k, j, i);
        const Real u2 = primitive(b, iUX+1, k, j, i);
        const Real u3 = primitive(b, iUX+2, k, j, i);
        const Real uvsq = gcov[1][1] * u1 * u1 + gcov[2][2] * u2 * u2 +
                          gcov[3][3] * u3 * u3 +
                          2.0 * (gcov[1][2] * u1 * u2 + gcov[1][3] * u1 * u3 +
                                 gcov[2][3] * u2 * u3);
        if (umax2 > 0.0 && uvsq > umax2) {
          const Real factor = Kokkos::sqrt(umax2 / Kokkos::max(uvsq, small));
          primitive(b, iUX,   k, j, i) *= factor;
          primitive(b, iUX+1, k, j, i) *= factor;
          primitive(b, iUX+2, k, j, i) *= factor;
        }

        const Real rho_after_floor = Kokkos::max(primitive(b, iRHO, k, j, i), small);
        if (enable_heating && rho_after_floor != rho_before_floor) {
          primitive(b, iKEL, k, j, i) *=
              Kokkos::pow(rho_after_floor / rho_before_floor, -kAdiabaticIndex);
        }

        primitive(b, iENT, k, j, i) =
            (kAdiabaticIndex - 1.0) * primitive(b, iENY, k, j, i) *
            Kokkos::pow(primitive(b, iRHO, k, j, i), -kAdiabaticIndex);
        if (enable_heating) {
          primitive(b, iKEL, k, j, i) =
              clampByRatio(ratio, primitive(b, iENT, k, j, i), primitive(b, iKEL, k, j, i));
        }

        if (mtype_int == MetricType::CKS && r < r_excise) {
          primitive(b, iRHO, k, j, i) = dexcise;
          primitive(b, iENY, k, j, i) = e_excise;
          primitive(b, iUX,   k, j, i) = 0.0;
          primitive(b, iUX+1, k, j, i) = 0.0;
          primitive(b, iUX+2, k, j, i) = 0.0;
          primitive(b, iENT, k, j, i) = (kAdiabaticIndex - 1.0) * e_excise *
              Kokkos::pow(dexcise, -kAdiabaticIndex);
          if (enable_heating) primitive(b, iKEL, k, j, i) = e_excise;
        }
      });

  return parthenon::TaskStatus::complete;
}
