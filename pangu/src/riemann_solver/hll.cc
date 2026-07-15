// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "riemann_solver/hll.h"

#include <limits>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include <basic_types.hpp>
#include "initialization/variable_mnemonics.h"
#include "interpolation/interpolater_ppm4.h"
#include "interpolation/interpolater_mc.h"
#include "physics/fast_magnetosonic_speed.h"
#include "physics/contravariant_flux.h"

parthenon::TaskStatus CalculateHLL(parthenon::MeshData<parthenon::Real> *md) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  const auto package_core = pmb0->packages.Get("core");
  const auto &kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");
  const auto enable_B = package_core->Param<bool>("enable_B");
  const auto enable_heating = package_core->Param<bool>("enable_heating");
  const auto &limiter = package_core->Param<std::string>("limiter");
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
  PackIndexMap fastMsIndexMap;
  const std::vector<std::string> fast_ms_tags = {"alfven"};
  auto FastMagnetosonicSpeed =
      md->PackVariables(fast_ms_tags, fastMsIndexMap);

  auto covariant_metric =
      md->PackVariables(std::vector<std::string>{"covariant_metric"});
  auto contravariant_metric =
      md->PackVariables(std::vector<std::string>{"contravariant_metric"});
  auto metric_determinant =
      md->PackVariables(std::vector<std::string>{"metric_determinant"});

  const auto meshgrid_size_x1 =
      pmb0->cellbounds.ncellsi(IndexDomain::entire);
  const auto meshgrid_size_x2 =
      pmb0->cellbounds.ncellsj(IndexDomain::entire);
  const auto meshgrid_size_x3 =
      pmb0->cellbounds.ncellsk(IndexDomain::entire);

  const int offset_x1 = (meshgrid_size_x1 > 1) ? 1 : 0;
  const int offset_x2 = (meshgrid_size_x2 > 1) ? 1 : 0;
  const int offset_x3 = (meshgrid_size_x3 > 1) ? 1 : 0;

  const bool use_mc = (limiter == "mc");

  pmb0->par_for(
      PARTHENON_AUTO_LABEL, block.s, block.e,
      bound_x3_interior.s - offset_x3, bound_x3_interior.e + offset_x3,
      bound_x2_interior.s - offset_x2, bound_x2_interior.e + offset_x2,
      bound_x1_interior.s, bound_x1_interior.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real primitiveLeft[NPRIM] = {0}, primitiveRight[NPRIM] = {0};

        auto recon = [&](int pidx, int cidx) {
          Real ql_i, qr_im1, ql_ip1, qr_i;
          if (use_mc) {
            MC(primitive(b, pidx, k, j, i - 3), primitive(b, pidx, k, j, i - 2),
               primitive(b, pidx, k, j, i - 1), primitive(b, pidx, k, j, i),
               primitive(b, pidx, k, j, i + 1), ql_i, qr_im1);
            MC(primitive(b, pidx, k, j, i - 2), primitive(b, pidx, k, j, i - 1),
               primitive(b, pidx, k, j, i), primitive(b, pidx, k, j, i + 1),
               primitive(b, pidx, k, j, i + 2), ql_ip1, qr_i);
          } else {
            PPM4(primitive(b, pidx, k, j, i - 3), primitive(b, pidx, k, j, i - 2),
                 primitive(b, pidx, k, j, i - 1), primitive(b, pidx, k, j, i),
                 primitive(b, pidx, k, j, i + 1), ql_i, qr_im1);
            PPM4(primitive(b, pidx, k, j, i - 2), primitive(b, pidx, k, j, i - 1),
                 primitive(b, pidx, k, j, i), primitive(b, pidx, k, j, i + 1),
                 primitive(b, pidx, k, j, i + 2), ql_ip1, qr_i);
          }
          primitiveLeft[cidx] = ql_i;
          primitiveRight[cidx] = qr_i;
        };

        int p = 0;
        recon(p++, RHO);
        recon(p++, ENY);
        for (int d = 0; d < 3; ++d) recon(p++, UX1 + d);
        if (enable_B) for (int d = 0; d < 3; ++d) recon(p++, BX1 + d);
        recon(p++, ENT);
        if (enable_heating) recon(p++, KEL);

        Real conservativeLeft[NPRIM];
        Real conservativeRight[NPRIM];
        Real fluxLeft[NPRIM];
        Real fluxRight[NPRIM];

        Real gcovFace[4][4], gconFace[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcovFace[row][col] =
                covariant_metric(b, FACEX1 * 16 + col * 4 + row, k, j, i);
            gconFace[row][col] =
                contravariant_metric(b, FACEX1 * 16 + col * 4 + row, k, j, i);
          }
        }

        Real fastMaxL, fastMaxR;
        Real fastMinL, fastMinR;
        CalculateFastMagnetosonicSpeed(
            kAdiabaticIndex, primitiveLeft, gcovFace, gconFace, X1DIR,
            fastMaxL, fastMinL);
        CalculateFastMagnetosonicSpeed(
            kAdiabaticIndex, primitiveRight, gcovFace, gconFace, X1DIR,
            fastMaxR, fastMinR);
        const auto maxFastCenter = Kokkos::fabs(
            Kokkos::max(Kokkos::max(0., fastMaxL), fastMaxR));
        const auto minFastCenter = Kokkos::fabs(
            Kokkos::max(Kokkos::max(0., -fastMinL), -fastMinR));
        const auto fastMsCenter = Kokkos::max(maxFastCenter, minFastCenter);
        FastMagnetosonicSpeed(b, Vector3D::X1, k, j, i) = fastMsCenter;

        const auto MetricDeterminantFace =
            metric_determinant(b, FACEX1, k, j, i);
        CalculateContravariantFlux(
            kAdiabaticIndex, primitiveLeft, gcovFace, gconFace,
            MetricDeterminantFace, X0DIR, conservativeLeft);
        CalculateContravariantFlux(
            kAdiabaticIndex, primitiveRight, gcovFace, gconFace,
            MetricDeterminantFace, X0DIR, conservativeRight);
        CalculateContravariantFlux(
            kAdiabaticIndex, primitiveLeft, gcovFace, gconFace,
            MetricDeterminantFace, X1DIR, fluxLeft);
        CalculateContravariantFlux(
            kAdiabaticIndex, primitiveRight, gcovFace, gconFace,
            MetricDeterminantFace, X1DIR, fluxRight);

        Real fluxHLL[NPRIM];
        for (int ci = 0; ci < NPRIM; ++ci) {
          fluxHLL[ci] = (maxFastCenter * fluxLeft[ci] +
                         minFastCenter * fluxRight[ci] -
                         maxFastCenter * minFastCenter *
                             (conservativeRight[ci] - conservativeLeft[ci])) /
                        (maxFastCenter + minFastCenter + 1e-20);
        }

        int f = 0;
        conservative(b).flux(X1DIR, f++, k, j, i) = fluxHLL[RHO];
        conservative(b).flux(X1DIR, f++, k, j, i) = fluxHLL[ENY];
        conservative(b).flux(X1DIR, f++, k, j, i) = fluxHLL[UX1];
        conservative(b).flux(X1DIR, f++, k, j, i) = fluxHLL[UX2];
        conservative(b).flux(X1DIR, f++, k, j, i) = fluxHLL[UX3];
        if (enable_B) {
          conservative(b).flux(X1DIR, f++, k, j, i) = fluxHLL[BX1];
          conservative(b).flux(X1DIR, f++, k, j, i) = fluxHLL[BX2];
          conservative(b).flux(X1DIR, f++, k, j, i) = fluxHLL[BX3];
        }
        conservative(b).flux(X1DIR, f++, k, j, i) = fluxHLL[ENT];
        if (enable_heating) {
          conservative(b).flux(X1DIR, f++, k, j, i) = fluxHLL[KEL];
        }
      });

  if (pmb0->pmy_mesh->ndim >= 2)
    pmb0->par_for(
        PARTHENON_AUTO_LABEL, block.s, block.e,
        bound_x3_interior.s - offset_x3, bound_x3_interior.e + offset_x3,
        bound_x1_interior.s - offset_x1, bound_x1_interior.e + offset_x1,
        bound_x2_interior.s, bound_x2_interior.e + 1,
        KOKKOS_LAMBDA(const int b, const int k, const int i, const int j) {
          Real primitiveLeft[NPRIM] = {0}, primitiveRight[NPRIM] = {0};

          auto recon = [&](int pidx, int cidx) {
            Real ql_i, qr_im1, ql_ip1, qr_i;
            if (use_mc) {
              MC(primitive(b, pidx, k, j - 3, i), primitive(b, pidx, k, j - 2, i),
                 primitive(b, pidx, k, j - 1, i), primitive(b, pidx, k, j, i),
                 primitive(b, pidx, k, j + 1, i), ql_i, qr_im1);
              MC(primitive(b, pidx, k, j - 2, i), primitive(b, pidx, k, j - 1, i),
                 primitive(b, pidx, k, j, i), primitive(b, pidx, k, j + 1, i),
                 primitive(b, pidx, k, j + 2, i), ql_ip1, qr_i);
            } else {
              PPM4(primitive(b, pidx, k, j - 3, i), primitive(b, pidx, k, j - 2, i),
                   primitive(b, pidx, k, j - 1, i), primitive(b, pidx, k, j, i),
                   primitive(b, pidx, k, j + 1, i), ql_i, qr_im1);
              PPM4(primitive(b, pidx, k, j - 2, i), primitive(b, pidx, k, j - 1, i),
                   primitive(b, pidx, k, j, i), primitive(b, pidx, k, j + 1, i),
                   primitive(b, pidx, k, j + 2, i), ql_ip1, qr_i);
            }
            primitiveLeft[cidx] = ql_i;
            primitiveRight[cidx] = qr_i;
          };

          int p = 0;
          recon(p++, RHO);
          recon(p++, ENY);
          for (int d = 0; d < 3; ++d) recon(p++, UX1 + d);
          if (enable_B) for (int d = 0; d < 3; ++d) recon(p++, BX1 + d);
          recon(p++, ENT);
          if (enable_heating) recon(p++, KEL);

          Real conservativeLeft[NPRIM];
          Real conservativeRight[NPRIM];
          Real fluxLeft[NPRIM];
          Real fluxRight[NPRIM];

          Real gcovFace[4][4], gconFace[4][4];
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
              gcovFace[row][col] =
                  covariant_metric(b, FACEX2 * 16 + col * 4 + row, k, j, i);
              gconFace[row][col] =
                  contravariant_metric(b, FACEX2 * 16 + col * 4 + row, k, j, i);
            }
          }

          Real fastMaxL, fastMaxR;
          Real fastMinL, fastMinR;
          CalculateFastMagnetosonicSpeed(
              kAdiabaticIndex, primitiveLeft, gcovFace, gconFace, X2DIR,
              fastMaxL, fastMinL);
          CalculateFastMagnetosonicSpeed(
              kAdiabaticIndex, primitiveRight, gcovFace, gconFace,
              X2DIR, fastMaxR, fastMinR);
          const auto maxFastCenter = Kokkos::fabs(
              Kokkos::max(Kokkos::max(0., fastMaxL), fastMaxR));
          const auto minFastCenter = Kokkos::fabs(
              Kokkos::max(Kokkos::max(0., -fastMinL), -fastMinR));
          const auto fastMsCenter = Kokkos::max(
              maxFastCenter, minFastCenter);
          FastMagnetosonicSpeed(b, Vector3D::X2, k, j, i) = fastMsCenter;

          const auto MetricDeterminantFace =
              metric_determinant(b, FACEX2, k, j, i);
          CalculateContravariantFlux(
              kAdiabaticIndex, primitiveLeft, gcovFace, gconFace,
              MetricDeterminantFace, X0DIR, conservativeLeft);
          CalculateContravariantFlux(
              kAdiabaticIndex, primitiveRight, gcovFace, gconFace,
              MetricDeterminantFace, X0DIR, conservativeRight);
          CalculateContravariantFlux(
              kAdiabaticIndex, primitiveLeft, gcovFace, gconFace,
              MetricDeterminantFace, X2DIR, fluxLeft);
          CalculateContravariantFlux(
              kAdiabaticIndex, primitiveRight, gcovFace, gconFace,
              MetricDeterminantFace, X2DIR, fluxRight);

          Real fluxHLL[NPRIM];
          for (int ci = 0; ci < NPRIM; ++ci) {
            fluxHLL[ci] = (maxFastCenter * fluxLeft[ci] +
                           minFastCenter * fluxRight[ci] -
                           maxFastCenter * minFastCenter *
                               (conservativeRight[ci] - conservativeLeft[ci])) /
                          (maxFastCenter + minFastCenter + 1e-20);
          }

          int f = 0;
          conservative(b).flux(X2DIR, f++, k, j, i) = fluxHLL[RHO];
          conservative(b).flux(X2DIR, f++, k, j, i) = fluxHLL[ENY];
          conservative(b).flux(X2DIR, f++, k, j, i) = fluxHLL[UX1];
          conservative(b).flux(X2DIR, f++, k, j, i) = fluxHLL[UX2];
          conservative(b).flux(X2DIR, f++, k, j, i) = fluxHLL[UX3];
          if (enable_B) {
            conservative(b).flux(X2DIR, f++, k, j, i) = fluxHLL[BX1];
            conservative(b).flux(X2DIR, f++, k, j, i) = fluxHLL[BX2];
            conservative(b).flux(X2DIR, f++, k, j, i) = fluxHLL[BX3];
          }
          conservative(b).flux(X2DIR, f++, k, j, i) = fluxHLL[ENT];
          if (enable_heating) {
            conservative(b).flux(X2DIR, f++, k, j, i) = fluxHLL[KEL];
          }
        });

  if (pmb0->pmy_mesh->ndim == 3)
    pmb0->par_for(
        PARTHENON_AUTO_LABEL, block.s, block.e,
        bound_x2_interior.s, bound_x2_interior.e,
        bound_x1_interior.s, bound_x1_interior.e,
        bound_x3_interior.s, bound_x3_interior.e + 1,
        KOKKOS_LAMBDA(const int b, const int j, const int i, const int k) {
          Real primitiveLeft[NPRIM] = {0}, primitiveRight[NPRIM] = {0};

          auto recon = [&](int pidx, int cidx) {
            Real ql_i, qr_im1, ql_ip1, qr_i;
            if (use_mc) {
              MC(primitive(b, pidx, k - 3, j, i), primitive(b, pidx, k - 2, j, i),
                 primitive(b, pidx, k - 1, j, i), primitive(b, pidx, k, j, i),
                 primitive(b, pidx, k + 1, j, i), ql_i, qr_im1);
              MC(primitive(b, pidx, k - 2, j, i), primitive(b, pidx, k - 1, j, i),
                 primitive(b, pidx, k, j, i), primitive(b, pidx, k + 1, j, i),
                 primitive(b, pidx, k + 2, j, i), ql_ip1, qr_i);
            } else {
              PPM4(primitive(b, pidx, k - 3, j, i), primitive(b, pidx, k - 2, j, i),
                   primitive(b, pidx, k - 1, j, i), primitive(b, pidx, k, j, i),
                   primitive(b, pidx, k + 1, j, i), ql_i, qr_im1);
              PPM4(primitive(b, pidx, k - 2, j, i), primitive(b, pidx, k - 1, j, i),
                   primitive(b, pidx, k, j, i), primitive(b, pidx, k + 1, j, i),
                   primitive(b, pidx, k + 2, j, i), ql_ip1, qr_i);
            }
            primitiveLeft[cidx] = ql_i;
            primitiveRight[cidx] = qr_i;
          };

          int p = 0;
          recon(p++, RHO);
          recon(p++, ENY);
          for (int d = 0; d < 3; ++d) recon(p++, UX1 + d);
          if (enable_B) for (int d = 0; d < 3; ++d) recon(p++, BX1 + d);
          recon(p++, ENT);
          if (enable_heating) recon(p++, KEL);

          Real conservativeLeft[NPRIM];
          Real conservativeRight[NPRIM];
          Real fluxLeft[NPRIM];
          Real fluxRight[NPRIM];

          Real gcovFace[4][4], gconFace[4][4];
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
              gcovFace[row][col] =
                  covariant_metric(b, FACEX3 * 16 + col * 4 + row, k, j, i);
              gconFace[row][col] =
                  contravariant_metric(b, FACEX3 * 16 + col * 4 + row, k, j, i);
            }
          }

          Real fastMaxL, fastMaxR;
          Real fastMinL, fastMinR;
          CalculateFastMagnetosonicSpeed(
              kAdiabaticIndex, primitiveLeft, gcovFace, gconFace, X3DIR,
              fastMaxL, fastMinL);
          CalculateFastMagnetosonicSpeed(
              kAdiabaticIndex, primitiveRight, gcovFace, gconFace,
              X3DIR, fastMaxR, fastMinR);
          const auto maxFastCenter = Kokkos::fabs(
              Kokkos::max(Kokkos::max(0., fastMaxL), fastMaxR));
          const auto minFastCenter = Kokkos::fabs(
              Kokkos::max(Kokkos::max(0., -fastMinL), -fastMinR));
          const auto fastMsCenter = Kokkos::max(
              maxFastCenter, minFastCenter);
          FastMagnetosonicSpeed(b, Vector3D::X3, k, j, i) = fastMsCenter;

          const auto MetricDeterminantFace =
              metric_determinant(b, FACEX3, k, j, i);
          CalculateContravariantFlux(
              kAdiabaticIndex, primitiveLeft, gcovFace, gconFace,
              MetricDeterminantFace, X0DIR, conservativeLeft);
          CalculateContravariantFlux(
              kAdiabaticIndex, primitiveRight, gcovFace, gconFace,
              MetricDeterminantFace, X0DIR, conservativeRight);
          CalculateContravariantFlux(
              kAdiabaticIndex, primitiveLeft, gcovFace, gconFace,
              MetricDeterminantFace, X3DIR, fluxLeft);
          CalculateContravariantFlux(
              kAdiabaticIndex, primitiveRight, gcovFace, gconFace,
              MetricDeterminantFace, X3DIR, fluxRight);

          Real fluxHLL[NPRIM];
          for (int ci = 0; ci < NPRIM; ++ci) {
            fluxHLL[ci] = (maxFastCenter * fluxLeft[ci] +
                           minFastCenter * fluxRight[ci] -
                           maxFastCenter * minFastCenter *
                               (conservativeRight[ci] - conservativeLeft[ci])) /
                          (maxFastCenter + minFastCenter + 1e-20);
          }

          int f = 0;
          conservative(b).flux(X3DIR, f++, k, j, i) = fluxHLL[RHO];
          conservative(b).flux(X3DIR, f++, k, j, i) = fluxHLL[ENY];
          conservative(b).flux(X3DIR, f++, k, j, i) = fluxHLL[UX1];
          conservative(b).flux(X3DIR, f++, k, j, i) = fluxHLL[UX2];
          conservative(b).flux(X3DIR, f++, k, j, i) = fluxHLL[UX3];
          if (enable_B) {
            conservative(b).flux(X3DIR, f++, k, j, i) = fluxHLL[BX1];
            conservative(b).flux(X3DIR, f++, k, j, i) = fluxHLL[BX2];
            conservative(b).flux(X3DIR, f++, k, j, i) = fluxHLL[BX3];
          }
          conservative(b).flux(X3DIR, f++, k, j, i) = fluxHLL[ENT];
          if (enable_heating) {
            conservative(b).flux(X3DIR, f++, k, j, i) = fluxHLL[KEL];
          }
        });

  return TaskStatus::complete;
}
