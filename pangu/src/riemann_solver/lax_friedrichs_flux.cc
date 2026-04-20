// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/riemann_solver module defines lax_friedrichs_flux.cc
// responsibilities for the Pangu runtime. It centers on riemann_solver to express core data
// flow, keep interfaces readable, and preserve predictable behavior across task
// coordination, recovery paths, and performance-sensitive execution.

#include "riemann_solver/lax_friedrichs_flux.h"

#include <limits>
#include <memory>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "initialization/variable_mnemonics.h"
#include "interpolation/interpolater_mc.h"
#include "physics_transformation/alfven_velocity.h"
#include "physics_transformation/contravariant_flux.h"

parthenon::TaskStatus CalculateFluxesSRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
  const auto &kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");

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
  PackIndexMap alfvenVelocityIndexMap;
  const std::vector<std::string> alfven_tags = {"alfven"};
  auto AlfvenVelocity =
      resource->PackVariables(alfven_tags, alfvenVelocityIndexMap);

  const int ScratchLevel = 1;
  const auto meshgrid_size_x1 =
      pmb->cellbounds.ncellsi(IndexDomain::entire);
  const auto meshgrid_size_x2 =
      pmb->cellbounds.ncellsj(IndexDomain::entire);
  const auto meshgrid_size_x3 =
      pmb->cellbounds.ncellsk(IndexDomain::entire);

  const size_t ScratchSizeInBytesX1 =
      ScratchPad2D<Real>::shmem_size(NPRIM, meshgrid_size_x1);
  const size_t ScratchSizeInBytesX2 =
      ScratchPad2D<Real>::shmem_size(NPRIM, meshgrid_size_x2);
  const size_t ScratchSizeInBytesX3 =
      ScratchPad2D<Real>::shmem_size(NPRIM, meshgrid_size_x3);

  const int offset_x1 = (meshgrid_size_x1 > 1) ? 1 : 0;
  const int offset_x2 = (meshgrid_size_x2 > 1) ? 1 : 0;
  const int offset_x3 = (meshgrid_size_x3 > 1) ? 1 : 0;

  pmb->par_for_outer(
      PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX1, ScratchLevel,
      bound_x3_interior.s - offset_x3, bound_x3_interior.e + offset_x3, bound_x2_interior.s - offset_x2,
      bound_x2_interior.e + offset_x2,
      KOKKOS_LAMBDA(team_mbr_t member, const int k, const int j) {
        ScratchPad2D<Real> primitiveLeft(member.team_scratch(ScratchLevel),
                                         NPRIM,
                                         meshgrid_size_x1);
        ScratchPad2D<Real> primitiveRight(member.team_scratch(ScratchLevel),
                                          NPRIM,
                                          meshgrid_size_x1);

        par_for_inner(member, 0, NPRIM - 1, bound_x1_interior.s,
                      bound_x1_interior.e + 1, [&](const int n, const int i) {
                        primitiveLeft(n, i) =
                            primitive(n, k, j, i - 1) +
                            0.5 * InterpolateMc(primitive(n, k, j, i - 2),
                                                primitive(n, k, j, i - 1),
                                                primitive(n, k, j, i));
                        primitiveRight(n, i) =
                            primitive(n, k, j, i) -
                            0.5 * InterpolateMc(primitive(n, k, j, i - 1),
                                                primitive(n, k, j, i),
                                                primitive(n, k, j, i + 1));
                      });

        member.team_barrier();

        par_for_inner(member, bound_x1_interior.s, bound_x1_interior.e + 1, [&](const int i) {
          Real conservativeLeft[NPRIM];
          Real conservativeRight[NPRIM];
          Real fluxLeft[NPRIM];
          Real fluxRight[NPRIM];
          const Real PrimitiveLeftCArray[NPRIM] = {
              primitiveLeft(RHO, i), primitiveLeft(ENY, i),
              primitiveLeft(UX1, i), primitiveLeft(UX2, i),
              primitiveLeft(UX3, i), primitiveLeft(BX1, i),
              primitiveLeft(BX2, i), primitiveLeft(BX3, i),
          };
          const Real PrimitiveRightCArray[NPRIM] = {
              primitiveRight(RHO, i), primitiveRight(ENY, i),
              primitiveRight(UX1, i), primitiveRight(UX2, i),
              primitiveRight(UX3, i), primitiveRight(BX1, i),
              primitiveRight(BX2, i), primitiveRight(BX3, i),
          };
          const auto *primitive_right = PrimitiveRightCArray;

          Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
          Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
          CalculateAlfvenVelocitySRMHD(kAdiabaticIndex, PrimitiveLeftCArray,
                                       X1DIR, maximumAlfvenVelocityLeft,
                                       minimumAlfvenVelocityLeft);
          CalculateAlfvenVelocitySRMHD(kAdiabaticIndex, PrimitiveRightCArray,
                                       X1DIR, maximumAlfvenVelocityRight,
                                       minimumAlfvenVelocityRight);
          const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(
              Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft),
                          maximumAlfvenVelocityRight));
          const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(
              Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft),
                          -minimumAlfvenVelocityRight));
          const auto AlfvenVelocityCenter = Kokkos::max(
              MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);
          AlfvenVelocity(Vector3D::X1, k, j, i) = AlfvenVelocityCenter;

          CalculateContravariantFluxSRMHD(kAdiabaticIndex, PrimitiveLeftCArray,
                                          X0DIR, conservativeLeft);
          CalculateContravariantFluxSRMHD(kAdiabaticIndex, primitive_right,
                                          X0DIR, conservativeRight);
          CalculateContravariantFluxSRMHD(kAdiabaticIndex, PrimitiveLeftCArray,
                                          X1DIR, fluxLeft);
          CalculateContravariantFluxSRMHD(kAdiabaticIndex, primitive_right,
                                          X1DIR, fluxRight);

          for (int index = 0; index < NPRIM; index++)
            conservative.flux(X1DIR, index, k, j, i) =
                0.5 * (fluxLeft[index] + fluxRight[index] -
                       AlfvenVelocityCenter * (conservativeRight[index] -
                                               conservativeLeft[index]));
        });
      });

  if (pmb->pmy_mesh->ndim >= 2)
    pmb->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX2, ScratchLevel,
        bound_x3_interior.s - offset_x3, bound_x3_interior.e + offset_x3, bound_x1_interior.s - offset_x1,
        bound_x1_interior.e + offset_x1,
        KOKKOS_LAMBDA(team_mbr_t member, const int k, const int i) {
          ScratchPad2D<Real> primitiveLeft(member.team_scratch(ScratchLevel),
                                           NPRIM,
                                           meshgrid_size_x2);
          ScratchPad2D<Real> primitiveRight(member.team_scratch(ScratchLevel),
                                            NPRIM,
                                            meshgrid_size_x2);

          par_for_inner(member, 0, NPRIM - 1, bound_x2_interior.s,
                        bound_x2_interior.e + 1, [&](const int n, const int j) {
                          primitiveLeft(n, j) =
                              primitive(n, k, j - 1, i) +
                              0.5 * InterpolateMc(primitive(n, k, j - 2, i),
                                                  primitive(n, k, j - 1, i),
                                                  primitive(n, k, j, i));
                          primitiveRight(n, j) =
                              primitive(n, k, j, i) -
                              0.5 * InterpolateMc(primitive(n, k, j - 1, i),
                                                  primitive(n, k, j, i),
                                                  primitive(n, k, j + 1, i));
                        });

          member.team_barrier();

          par_for_inner(member, bound_x2_interior.s, bound_x2_interior.e + 1, [&](const int j) {
            Real conservativeLeft[NPRIM];
            Real conservativeRight[NPRIM];
            Real fluxLeft[NPRIM];
            Real fluxRight[NPRIM];
            const Real PrimitiveLeftCArray[NPRIM] = {
                primitiveLeft(RHO, j), primitiveLeft(ENY, j),
                primitiveLeft(UX1, j), primitiveLeft(UX2, j),
                primitiveLeft(UX3, j), primitiveLeft(BX1, j),
                primitiveLeft(BX2, j), primitiveLeft(BX3, j),
            };
            const Real PrimitiveRightCArray[NPRIM] = {
                primitiveRight(RHO, j), primitiveRight(ENY, j),
                primitiveRight(UX1, j), primitiveRight(UX2, j),
                primitiveRight(UX3, j), primitiveRight(BX1, j),
                primitiveRight(BX2, j), primitiveRight(BX3, j),
            };

            Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
            Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
            CalculateAlfvenVelocitySRMHD(kAdiabaticIndex, PrimitiveLeftCArray,
                                         X2DIR, maximumAlfvenVelocityLeft,
                                         minimumAlfvenVelocityLeft);
            CalculateAlfvenVelocitySRMHD(kAdiabaticIndex, PrimitiveRightCArray,
                                         X2DIR, maximumAlfvenVelocityRight,
                                         minimumAlfvenVelocityRight);
            const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(
                Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft),
                            maximumAlfvenVelocityRight));
            const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(
                Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft),
                            -minimumAlfvenVelocityRight));
            const auto AlfvenVelocityCenter = Kokkos::max(
                MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);
            AlfvenVelocity(Vector3D::X2, k, j, i) = AlfvenVelocityCenter;

            CalculateContravariantFluxSRMHD(
                kAdiabaticIndex, PrimitiveLeftCArray, X0DIR, conservativeLeft);
            CalculateContravariantFluxSRMHD(kAdiabaticIndex,
                                            PrimitiveRightCArray, X0DIR,
                                            conservativeRight);
            CalculateContravariantFluxSRMHD(
                kAdiabaticIndex, PrimitiveLeftCArray, X2DIR, fluxLeft);
            CalculateContravariantFluxSRMHD(
                kAdiabaticIndex, PrimitiveRightCArray, X1DIR, fluxRight);

            for (int index = 0; index < NPRIM; index++)
              conservative.flux(X2DIR, index, k, j, i) =
                  0.5 * (fluxLeft[index] + fluxRight[index] -
                         AlfvenVelocityCenter * (conservativeRight[index] -
                                                 conservativeLeft[index]));
          });
        });

  if (pmb->pmy_mesh->ndim == 3)
    pmb->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX3, ScratchLevel, bound_x2_interior.s,
        bound_x2_interior.e, bound_x1_interior.s, bound_x1_interior.e,
        KOKKOS_LAMBDA(team_mbr_t member, const int j, const int i) {
          ScratchPad2D<Real> primitiveLeft(member.team_scratch(ScratchLevel),
                                           NPRIM,
                                           meshgrid_size_x3);
          ScratchPad2D<Real> primitiveRight(member.team_scratch(ScratchLevel),
                                            NPRIM,
                                            meshgrid_size_x3);

          par_for_inner(member, 0, NPRIM - 1, bound_x3_interior.s,
                        bound_x3_interior.e + 1, [&](const int n, const int k) {
                          primitiveLeft(n, k) =
                              primitive(n, k - 1, j, i) +
                              0.5 * InterpolateMc(primitive(n, k - 2, j, i),
                                                  primitive(n, k - 1, j, i),
                                                  primitive(n, k, j, i));
                          primitiveRight(n, k) =
                              primitive(n, k, j, i) -
                              0.5 * InterpolateMc(primitive(n, k - 1, j, i),
                                                  primitive(n, k, j, i),
                                                  primitive(n, k + 1, j, i));
                        });

          member.team_barrier();

          par_for_inner(member, bound_x3_interior.s, bound_x3_interior.e + 1, [&](const int k) {
            Real conservativeLeft[NPRIM];
            Real conservativeRight[NPRIM];
            Real fluxLeft[NPRIM];
            Real fluxRight[NPRIM];
            const Real PrimitiveLeftCArray[NPRIM] = {
                primitiveLeft(RHO, k), primitiveLeft(ENY, k),
                primitiveLeft(UX1, k), primitiveLeft(UX2, k),
                primitiveLeft(UX3, k), primitiveLeft(BX1, k),
                primitiveLeft(BX2, k), primitiveLeft(BX3, k),
            };
            const Real PrimitiveRightCArray[NPRIM] = {
                primitiveRight(RHO, k), primitiveRight(ENY, k),
                primitiveRight(UX1, k), primitiveRight(UX2, k),
                primitiveRight(UX3, k), primitiveRight(BX1, k),
                primitiveRight(BX2, k), primitiveRight(BX3, k),
            };

            Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
            Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
            CalculateAlfvenVelocitySRMHD(kAdiabaticIndex, PrimitiveLeftCArray,
                                         X3DIR, maximumAlfvenVelocityLeft,
                                         minimumAlfvenVelocityLeft);
            CalculateAlfvenVelocitySRMHD(kAdiabaticIndex, PrimitiveRightCArray,
                                         X3DIR, maximumAlfvenVelocityRight,
                                         minimumAlfvenVelocityRight);
            const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(
                Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft),
                            maximumAlfvenVelocityRight));
            const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(
                Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft),
                            -minimumAlfvenVelocityRight));
            const auto AlfvenVelocityCenter = Kokkos::max(
                MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);
            AlfvenVelocity(Vector3D::X3, k, j, i) = AlfvenVelocityCenter;

            CalculateContravariantFluxSRMHD(
                kAdiabaticIndex, PrimitiveLeftCArray, X0DIR, conservativeLeft);
            CalculateContravariantFluxSRMHD(kAdiabaticIndex,
                                            PrimitiveRightCArray, X0DIR,
                                            conservativeRight);
            CalculateContravariantFluxSRMHD(
                kAdiabaticIndex, PrimitiveLeftCArray, X3DIR, fluxLeft);
            CalculateContravariantFluxSRMHD(
                kAdiabaticIndex, PrimitiveRightCArray, X3DIR, fluxRight);

            for (int index = 0; index < NPRIM; index++)
              conservative.flux(X3DIR, index, k, j, i) =
                  0.5 * (fluxLeft[index] + fluxRight[index] -
                         AlfvenVelocityCenter * (conservativeRight[index] -
                                                 conservativeLeft[index]));
          });
        });

  return TaskStatus::complete;
}

parthenon::TaskStatus CalculateFluxesGRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
  const auto package_metric = pmb->packages.Get("metric");
  const auto &kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");

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
  PackIndexMap alfvenVelocityIndexMap;
  const std::vector<std::string> alfven_tags = {"alfven"};
  auto AlfvenVelocity =
      resource->PackVariables(alfven_tags, alfvenVelocityIndexMap);

  auto covariant_metric = geom_resource->Get("covariant_metric").data;
  auto contravariant_metric = geom_resource->Get("contravariant_metric").data;
  auto metric_determinant = geom_resource->Get("metric_determinant").data;

  const int ScratchLevel = 1;
  const auto meshgrid_size_x1 =
      pmb->cellbounds.ncellsi(IndexDomain::entire);
  const auto meshgrid_size_x2 =
      pmb->cellbounds.ncellsj(IndexDomain::entire);
  const auto meshgrid_size_x3 =
      pmb->cellbounds.ncellsk(IndexDomain::entire);

  const size_t ScratchSizeInBytesX1 =
      ScratchPad2D<Real>::shmem_size(NPRIM, meshgrid_size_x1);
  const size_t ScratchSizeInBytesX2 =
      ScratchPad2D<Real>::shmem_size(NPRIM, meshgrid_size_x2);
  const size_t ScratchSizeInBytesX3 =
      ScratchPad2D<Real>::shmem_size(NPRIM, meshgrid_size_x3);

  const int offset_x1 = (meshgrid_size_x1 > 1) ? 1 : 0;
  const int offset_x2 = (meshgrid_size_x2 > 1) ? 1 : 0;
  const int offset_x3 = (meshgrid_size_x3 > 1) ? 1 : 0;

  pmb->par_for_outer(
      PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX1, ScratchLevel,
      bound_x3_interior.s - offset_x3, bound_x3_interior.e + offset_x3, bound_x2_interior.s - offset_x2,
      bound_x2_interior.e + offset_x2,
      KOKKOS_LAMBDA(team_mbr_t member, const int k, const int j) {
        ScratchPad2D<Real> primitiveLeft(member.team_scratch(ScratchLevel),
                                         NPRIM,
                                         meshgrid_size_x1);
        ScratchPad2D<Real> primitiveRight(member.team_scratch(ScratchLevel),
                                          NPRIM,
                                          meshgrid_size_x1);

        par_for_inner(member, 0, NPRIM - 1, bound_x1_interior.s,
                      bound_x1_interior.e + 1, [&](const int n, const int i) {
                        primitiveLeft(n, i) =
                            primitive(n, k, j, i - 1) +
                            0.5 * InterpolateMc(primitive(n, k, j, i - 2),
                                                primitive(n, k, j, i - 1),
                                                primitive(n, k, j, i));
                        primitiveRight(n, i) =
                            primitive(n, k, j, i) -
                            0.5 * InterpolateMc(primitive(n, k, j, i - 1),
                                                primitive(n, k, j, i),
                                                primitive(n, k, j, i + 1));
                      });

        member.team_barrier();

        par_for_inner(member, bound_x1_interior.s, bound_x1_interior.e + 1, [&](const int i) {
          Real conservativeLeft[NPRIM];
          Real conservativeRight[NPRIM];
          Real fluxLeft[NPRIM];
          Real fluxRight[NPRIM];

          const Real PrimitiveLeftCArray[NPRIM] = {
              primitiveLeft(RHO, i), primitiveLeft(ENY, i),
              primitiveLeft(UX1, i), primitiveLeft(UX2, i),
              primitiveLeft(UX3, i), primitiveLeft(BX1, i),
              primitiveLeft(BX2, i), primitiveLeft(BX3, i),
          };
          const Real PrimitiveRightCArray[NPRIM] = {
              primitiveRight(RHO, i), primitiveRight(ENY, i),
              primitiveRight(UX1, i), primitiveRight(UX2, i),
              primitiveRight(UX3, i), primitiveRight(BX1, i),
              primitiveRight(BX2, i), primitiveRight(BX3, i),
          };

          Real gcovFace[4][4], gconFace[4][4];
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
              gcovFace[row][col] = covariant_metric(FACEX1, col, row, k, j, i);
              gconFace[row][col] =
                  contravariant_metric(FACEX1, col, row, k, j, i);
            }
          }

          Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
          Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
          CalculateAlfvenVelocityGRMHD(
              kAdiabaticIndex, PrimitiveLeftCArray, gcovFace, gconFace, X1DIR,
              maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
          CalculateAlfvenVelocityGRMHD(
              kAdiabaticIndex, PrimitiveRightCArray, gcovFace, gconFace, X1DIR,
              maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
          const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(
              Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft),
                          maximumAlfvenVelocityRight));
          const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(
              Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft),
                          -minimumAlfvenVelocityRight));
          const auto AlfvenVelocityCenter = Kokkos::max(
              MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);
          AlfvenVelocity(Vector3D::X1, k, j, i) = AlfvenVelocityCenter;

          const auto MetricDeterminantFace = metric_determinant(FACEX1, k, j, i);
          CalculateContravariantFluxGRMHD(
              kAdiabaticIndex, PrimitiveLeftCArray, gcovFace, gconFace,
              MetricDeterminantFace, X0DIR, conservativeLeft);
          CalculateContravariantFluxGRMHD(
              kAdiabaticIndex, PrimitiveRightCArray, gcovFace, gconFace,
              MetricDeterminantFace, X0DIR, conservativeRight);
          CalculateContravariantFluxGRMHD(
              kAdiabaticIndex, PrimitiveLeftCArray, gcovFace, gconFace,
              MetricDeterminantFace, X1DIR, fluxLeft);
          CalculateContravariantFluxGRMHD(
              kAdiabaticIndex, PrimitiveRightCArray, gcovFace, gconFace,
              MetricDeterminantFace, X1DIR, fluxRight);

          for (int index = 0; index < NPRIM; ++index) {
            conservative.flux(X1DIR, index, k, j, i) =
                0.5 * (fluxLeft[index] + fluxRight[index] -
                       AlfvenVelocityCenter * (conservativeRight[index] -
                                               conservativeLeft[index]));
          }
        });
      });

  if (pmb->pmy_mesh->ndim >= 2)
    pmb->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX2, ScratchLevel,
        bound_x3_interior.s - offset_x3, bound_x3_interior.e + offset_x3, bound_x1_interior.s - offset_x1,
        bound_x1_interior.e + offset_x1,
        KOKKOS_LAMBDA(team_mbr_t member, const int k, const int i) {
          ScratchPad2D<Real> primitiveLeft(member.team_scratch(ScratchLevel),
                                           NPRIM,
                                           meshgrid_size_x2);
          ScratchPad2D<Real> primitiveRight(member.team_scratch(ScratchLevel),
                                            NPRIM,
                                            meshgrid_size_x2);

          par_for_inner(member, 0, NPRIM - 1, bound_x2_interior.s,
                        bound_x2_interior.e + 1, [&](const int n, const int j) {
                          primitiveLeft(n, j) =
                              primitive(n, k, j - 1, i) +
                              0.5 * InterpolateMc(primitive(n, k, j - 2, i),
                                                  primitive(n, k, j - 1, i),
                                                  primitive(n, k, j, i));
                          primitiveRight(n, j) =
                              primitive(n, k, j, i) -
                              0.5 * InterpolateMc(primitive(n, k, j - 1, i),
                                                  primitive(n, k, j, i),
                                                  primitive(n, k, j + 1, i));
                        });

          member.team_barrier();

          par_for_inner(member, bound_x2_interior.s, bound_x2_interior.e + 1, [&](const int j) {
            Real conservativeLeft[NPRIM];
            Real conservativeRight[NPRIM];
            Real fluxLeft[NPRIM];
            Real fluxRight[NPRIM];

            const Real PrimitiveLeftCArray[NPRIM] = {
                primitiveLeft(RHO, j), primitiveLeft(ENY, j),
                primitiveLeft(UX1, j), primitiveLeft(UX2, j),
                primitiveLeft(UX3, j), primitiveLeft(BX1, j),
                primitiveLeft(BX2, j), primitiveLeft(BX3, j),
            };
            const Real PrimitiveRightCArray[NPRIM] = {
                primitiveRight(RHO, j), primitiveRight(ENY, j),
                primitiveRight(UX1, j), primitiveRight(UX2, j),
                primitiveRight(UX3, j), primitiveRight(BX1, j),
                primitiveRight(BX2, j), primitiveRight(BX3, j),
            };

            Real gcovFace[4][4], gconFace[4][4];
            for (int row = 0; row < 4; ++row) {
              for (int col = 0; col < 4; ++col) {
                gcovFace[row][col] = covariant_metric(FACEX2, col, row, k, j, i);
                gconFace[row][col] =
                    contravariant_metric(FACEX2, col, row, k, j, i);
              }
            }

            Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
            Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
            CalculateAlfvenVelocityGRMHD(
                kAdiabaticIndex, PrimitiveLeftCArray, gcovFace, gconFace, X2DIR,
                maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
            CalculateAlfvenVelocityGRMHD(
                kAdiabaticIndex, PrimitiveRightCArray, gcovFace, gconFace,
                X2DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
            const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(
                Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft),
                            maximumAlfvenVelocityRight));
            const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(
                Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft),
                            -minimumAlfvenVelocityRight));
            const auto AlfvenVelocityCenter = Kokkos::max(
                MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);
            AlfvenVelocity(Vector3D::X2, k, j, i) = AlfvenVelocityCenter;

            const auto MetricDeterminantFace =
                metric_determinant(FACEX2, k, j, i);
            CalculateContravariantFluxGRMHD(
                kAdiabaticIndex, PrimitiveLeftCArray, gcovFace, gconFace,
                MetricDeterminantFace, X0DIR, conservativeLeft);
            CalculateContravariantFluxGRMHD(
                kAdiabaticIndex, PrimitiveRightCArray, gcovFace, gconFace,
                MetricDeterminantFace, X0DIR, conservativeRight);
            CalculateContravariantFluxGRMHD(
                kAdiabaticIndex, PrimitiveLeftCArray, gcovFace, gconFace,
                MetricDeterminantFace, X2DIR, fluxLeft);
            CalculateContravariantFluxGRMHD(
                kAdiabaticIndex, PrimitiveRightCArray, gcovFace, gconFace,
                MetricDeterminantFace, X2DIR, fluxRight);

            for (int index = 0; index < NPRIM; ++index) {
              conservative.flux(X2DIR, index, k, j, i) =
                  0.5 * (fluxLeft[index] + fluxRight[index] -
                         AlfvenVelocityCenter * (conservativeRight[index] -
                                                 conservativeLeft[index]));
            }
          });
        });

  if (pmb->pmy_mesh->ndim == 3)
    pmb->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * ScratchSizeInBytesX3, ScratchLevel, bound_x2_interior.s,
        bound_x2_interior.e, bound_x1_interior.s, bound_x1_interior.e,
        KOKKOS_LAMBDA(team_mbr_t member, const int j, const int i) {
          ScratchPad2D<Real> primitiveLeft(member.team_scratch(ScratchLevel),
                                           NPRIM,
                                           meshgrid_size_x3);
          ScratchPad2D<Real> primitiveRight(member.team_scratch(ScratchLevel),
                                            NPRIM,
                                            meshgrid_size_x3);

          par_for_inner(member, 0, NPRIM - 1, bound_x3_interior.s,
                        bound_x3_interior.e + 1, [&](const int n, const int k) {
                          primitiveLeft(n, k) =
                              primitive(n, k - 1, j, i) +
                              0.5 * InterpolateMc(primitive(n, k - 2, j, i),
                                                  primitive(n, k - 1, j, i),
                                                  primitive(n, k, j, i));
                          primitiveRight(n, k) =
                              primitive(n, k, j, i) -
                              0.5 * InterpolateMc(primitive(n, k - 1, j, i),
                                                  primitive(n, k, j, i),
                                                  primitive(n, k + 1, j, i));
                        });

          member.team_barrier();

          par_for_inner(member, bound_x3_interior.s, bound_x3_interior.e + 1, [&](const int k) {
            Real conservativeLeft[NPRIM];
            Real conservativeRight[NPRIM];
            Real fluxLeft[NPRIM];
            Real fluxRight[NPRIM];

            const Real PrimitiveLeftCArray[NPRIM] = {
                primitiveLeft(RHO, k), primitiveLeft(ENY, k),
                primitiveLeft(UX1, k), primitiveLeft(UX2, k),
                primitiveLeft(UX3, k), primitiveLeft(BX1, k),
                primitiveLeft(BX2, k), primitiveLeft(BX3, k),
            };
            const Real PrimitiveRightCArray[NPRIM] = {
                primitiveRight(RHO, k), primitiveRight(ENY, k),
                primitiveRight(UX1, k), primitiveRight(UX2, k),
                primitiveRight(UX3, k), primitiveRight(BX1, k),
                primitiveRight(BX2, k), primitiveRight(BX3, k),
            };

            Real gcovFace[4][4], gconFace[4][4];
            for (int row = 0; row < 4; ++row) {
              for (int col = 0; col < 4; ++col) {
                gcovFace[row][col] = covariant_metric(FACEX3, col, row, k, j, i);
                gconFace[row][col] =
                    contravariant_metric(FACEX3, col, row, k, j, i);
              }
            }

            Real maximumAlfvenVelocityLeft, maximumAlfvenVelocityRight;
            Real minimumAlfvenVelocityLeft, minimumAlfvenVelocityRight;
            CalculateAlfvenVelocityGRMHD(
                kAdiabaticIndex, PrimitiveLeftCArray, gcovFace, gconFace, X3DIR,
                maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
            CalculateAlfvenVelocityGRMHD(
                kAdiabaticIndex, PrimitiveRightCArray, gcovFace, gconFace,
                X3DIR, maximumAlfvenVelocityRight, minimumAlfvenVelocityRight);
            const auto MaximumAlfvenVelocityCenter = Kokkos::fabs(
                Kokkos::max(Kokkos::max(0., maximumAlfvenVelocityLeft),
                            maximumAlfvenVelocityRight));
            const auto MinimumAlfvenVelocityCenter = Kokkos::fabs(
                Kokkos::max(Kokkos::max(0., -minimumAlfvenVelocityLeft),
                            -minimumAlfvenVelocityRight));
            const auto AlfvenVelocityCenter = Kokkos::max(
                MaximumAlfvenVelocityCenter, MinimumAlfvenVelocityCenter);
            AlfvenVelocity(Vector3D::X3, k, j, i) = AlfvenVelocityCenter;

            const auto MetricDeterminantFace =
                metric_determinant(FACEX3, k, j, i);
            CalculateContravariantFluxGRMHD(
                kAdiabaticIndex, PrimitiveLeftCArray, gcovFace, gconFace,
                MetricDeterminantFace, X0DIR, conservativeLeft);
            CalculateContravariantFluxGRMHD(
                kAdiabaticIndex, PrimitiveRightCArray, gcovFace, gconFace,
                MetricDeterminantFace, X0DIR, conservativeRight);
            CalculateContravariantFluxGRMHD(
                kAdiabaticIndex, PrimitiveLeftCArray, gcovFace, gconFace,
                MetricDeterminantFace, X3DIR, fluxLeft);
            CalculateContravariantFluxGRMHD(
                kAdiabaticIndex, PrimitiveRightCArray, gcovFace, gconFace,
                MetricDeterminantFace, X3DIR, fluxRight);

            for (int index = 0; index < NPRIM; ++index) {
              conservative.flux(X3DIR, index, k, j, i) =
                  0.5 * (fluxLeft[index] + fluxRight[index] -
                         AlfvenVelocityCenter * (conservativeRight[index] -
                                                 conservativeLeft[index]));
            }
          });
        });

  return TaskStatus::complete;
}
