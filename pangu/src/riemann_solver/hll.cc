// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "riemann_solver/hll.h"

#include <limits>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include <basic_types.hpp>
#include "initialization/variable_mnemonics.h"
#include "interpolation/interpolater_mc.h"
#include "physics/fast_magnetosonic_speed.h"
#include "physics/contravariant_flux.h"

parthenon::TaskStatus CalculateHLL(parthenon::MeshData<parthenon::Real> *md) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  const auto package_core = pmb0->packages.Get("core");
  const auto &kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");

  const auto bound_x1_interior = md->GetBoundsI(IndexDomain::interior);
  const auto bound_x2_interior = md->GetBoundsJ(IndexDomain::interior);
  const auto bound_x3_interior = md->GetBoundsK(IndexDomain::interior);
  auto block = IndexRange{0, md->NumBlocks() - 1};

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
      "electron_entropy"
    };
  const auto primitive =
      md->PackVariables(primitive_tags, primitiveIndexMap);
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

  pmb0->par_for(
      PARTHENON_AUTO_LABEL, block.s, block.e,
      bound_x3_interior.s - offset_x3, bound_x3_interior.e + offset_x3,
      bound_x2_interior.s - offset_x2, bound_x2_interior.e + offset_x2,
      bound_x1_interior.s, bound_x1_interior.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real primitiveLeft[NPRIM];
        Real primitiveRight[NPRIM];
        for (int index = 0; index < NPRIM; ++index) {
          primitiveLeft[index] = primitive(b, index, k, j, i - 1) +
                                  0.5 * InterpolateMC(primitive(b, index, k, j, i - 2),
                                                      primitive(b, index, k, j, i - 1),
                                                      primitive(b, index, k, j, i));
          primitiveRight[index] = primitive(b, index, k, j, i) -
                                   0.5 * InterpolateMC(primitive(b, index, k, j, i - 1),
                                                      primitive(b, index, k, j, i),
                                                      primitive(b, index, k, j, i + 1));
        }
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

        for (int index = 0; index < NPRIM; ++index) {
          conservative(b).flux(X1DIR, index, k, j, i) =
              (maxFastCenter * fluxLeft[index] +
               minFastCenter * fluxRight[index] -
               maxFastCenter * minFastCenter *
                   (conservativeRight[index] - conservativeLeft[index])) /
              (maxFastCenter + minFastCenter + 1e-20);
        }
      });

  if (pmb0->pmy_mesh->ndim >= 2)
    pmb0->par_for(
        PARTHENON_AUTO_LABEL, block.s, block.e,
        bound_x3_interior.s - offset_x3, bound_x3_interior.e + offset_x3,
        bound_x1_interior.s - offset_x1, bound_x1_interior.e + offset_x1,
        bound_x2_interior.s, bound_x2_interior.e + 1,
        KOKKOS_LAMBDA(const int b, const int k, const int i, const int j) {
          Real primitiveLeft[NPRIM];
          Real primitiveRight[NPRIM];
          for (int index = 0; index < NPRIM; ++index) {
            primitiveLeft[index] = primitive(b, index, k, j - 1, i) +
                                    0.5 * InterpolateMC(primitive(b, index, k, j - 2, i),
                                                        primitive(b, index, k, j - 1, i),
                                                        primitive(b, index, k, j, i));
            primitiveRight[index] = primitive(b, index, k, j, i) -
                                     0.5 * InterpolateMC(primitive(b, index, k, j - 1, i),
                                                        primitive(b, index, k, j, i),
                                                        primitive(b, index, k, j + 1, i));
          }
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

          for (int index = 0; index < NPRIM; ++index) {
            conservative(b).flux(X2DIR, index, k, j, i) =
                (maxFastCenter * fluxLeft[index] +
                 minFastCenter * fluxRight[index] -
                 maxFastCenter * minFastCenter *
                     (conservativeRight[index] - conservativeLeft[index])) /
                (maxFastCenter + minFastCenter + 1e-20);
          }
        });

  if (pmb0->pmy_mesh->ndim == 3)
    pmb0->par_for(
        PARTHENON_AUTO_LABEL, block.s, block.e,
        bound_x2_interior.s, bound_x2_interior.e,
        bound_x1_interior.s, bound_x1_interior.e,
        bound_x3_interior.s, bound_x3_interior.e + 1,
        KOKKOS_LAMBDA(const int b, const int j, const int i, const int k) {
          Real primitiveLeft[NPRIM];
          Real primitiveRight[NPRIM];
          for (int index = 0; index < NPRIM; ++index) {
            primitiveLeft[index] = primitive(b, index, k - 1, j, i) +
                                    0.5 * InterpolateMC(primitive(b, index, k - 2, j, i),
                                                        primitive(b, index, k - 1, j, i),
                                                        primitive(b, index, k, j, i));
            primitiveRight[index] = primitive(b, index, k, j, i) -
                                     0.5 * InterpolateMC(primitive(b, index, k - 1, j, i),
                                                        primitive(b, index, k, j, i),
                                                        primitive(b, index, k + 1, j, i));
          }

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

          for (int index = 0; index < NPRIM; ++index) {
            conservative(b).flux(X3DIR, index, k, j, i) =
                (maxFastCenter * fluxLeft[index] +
                 minFastCenter * fluxRight[index] -
                 maxFastCenter * minFastCenter *
                     (conservativeRight[index] - conservativeLeft[index])) /
                (maxFastCenter + minFastCenter + 1e-20);
          }
        });

  return TaskStatus::complete;
}
