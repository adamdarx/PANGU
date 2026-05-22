// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "riemann_solver/hll.h"

#include <limits>
#include <memory>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "initialization/variable_mnemonics.h"
#include "interpolation/interpolater_mc.h"
#include "physics/alfven_velocity.h"
#include "physics/contravariant_flux.h"

parthenon::TaskStatus CalculateHLL(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &init_resource) {
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
	  "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
	  "electron_entropy"
    };
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

  auto covariant_metric = init_resource->Get("covariant_metric").data;
  auto contravariant_metric = init_resource->Get("contravariant_metric").data;
  auto metric_determinant = init_resource->Get("metric_determinant").data;

  const auto meshgrid_size_x1 =
      pmb->cellbounds.ncellsi(IndexDomain::entire);
  const auto meshgrid_size_x2 =
      pmb->cellbounds.ncellsj(IndexDomain::entire);
  const auto meshgrid_size_x3 =
      pmb->cellbounds.ncellsk(IndexDomain::entire);

  const int offset_x1 = (meshgrid_size_x1 > 1) ? 1 : 0;
  const int offset_x2 = (meshgrid_size_x2 > 1) ? 1 : 0;
  const int offset_x3 = (meshgrid_size_x3 > 1) ? 1 : 0;

  pmb->par_for(
      PARTHENON_AUTO_LABEL,
      bound_x3_interior.s - offset_x3, bound_x3_interior.e + offset_x3,
      bound_x2_interior.s - offset_x2, bound_x2_interior.e + offset_x2,
      bound_x1_interior.s, bound_x1_interior.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real primitiveLeft[NPRIM];
        Real primitiveRight[NPRIM];
        for(int index = 0; index < NPRIM; ++index) {
					primitiveLeft[index] = primitive(index, k, j, i - 1) +
																	0.5 * InterpolateMC(primitive(index, k, j, i - 2),
																											primitive(index, k, j, i - 1),
																											primitive(index, k, j, i));
					primitiveRight[index] = primitive(index, k, j, i) -
																	0.5 * InterpolateMC(primitive(index, k, j, i - 1),
																											primitive(index, k, j, i),
																											primitive(index, k, j, i + 1));
        }
        Real conservativeLeft[NPRIM];
        Real conservativeRight[NPRIM];
        Real fluxLeft[NPRIM];
        Real fluxRight[NPRIM];
        
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
        CalculateAlfvenVelocity(
            kAdiabaticIndex, primitiveLeft, gcovFace, gconFace, X1DIR,
            maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
        CalculateAlfvenVelocity(
            kAdiabaticIndex, primitiveRight, gcovFace, gconFace, X1DIR,
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
        conservative.flux(X1DIR, index, k, j, i) =
            (MaximumAlfvenVelocityCenter * fluxLeft[index] + MinimumAlfvenVelocityCenter * fluxRight[index]
            - MaximumAlfvenVelocityCenter * MinimumAlfvenVelocityCenter * (conservativeRight[index] - conservativeLeft[index])) /
            (MaximumAlfvenVelocityCenter + MinimumAlfvenVelocityCenter + 1e-20);
        }
    });

  if (pmb->pmy_mesh->ndim >= 2)
    pmb->par_for(
        PARTHENON_AUTO_LABEL,
        bound_x3_interior.s - offset_x3, bound_x3_interior.e + offset_x3,
				bound_x1_interior.s - offset_x1, bound_x1_interior.e + offset_x1,
				bound_x2_interior.s, bound_x2_interior.e + 1,
        KOKKOS_LAMBDA(const int k, const int i, const int j) {
					Real primitiveLeft[NPRIM];
					Real primitiveRight[NPRIM];
					for(int index = 0; index < NPRIM; ++index) {
						primitiveLeft[index] = primitive(index, k, j - 1, i) +
																		0.5 * InterpolateMC(primitive(index, k, j - 2, i),
																										primitive(index, k, j - 1, i),
																										primitive(index, k, j, i));
						primitiveRight[index] = primitive(index, k, j, i) -
																		0.5 * InterpolateMC(primitive(index, k, j - 1, i),
																										primitive(index, k, j, i),
																										primitive(index, k, j + 1, i));
					}
					Real conservativeLeft[NPRIM];
					Real conservativeRight[NPRIM];
					Real fluxLeft[NPRIM];
					Real fluxRight[NPRIM];

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
					CalculateAlfvenVelocity(
							kAdiabaticIndex, primitiveLeft, gcovFace, gconFace, X2DIR,
							maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
					CalculateAlfvenVelocity(
							kAdiabaticIndex, primitiveRight, gcovFace, gconFace,
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
						conservative.flux(X2DIR, index, k, j, i) =
                (MaximumAlfvenVelocityCenter * fluxLeft[index] + MinimumAlfvenVelocityCenter * fluxRight[index]
                - MaximumAlfvenVelocityCenter * MinimumAlfvenVelocityCenter * (conservativeRight[index] - conservativeLeft[index])) /
                (MaximumAlfvenVelocityCenter + MinimumAlfvenVelocityCenter + 1e-20);
					}
				});

  if (pmb->pmy_mesh->ndim == 3)
    pmb->par_for(
        PARTHENON_AUTO_LABEL,
				bound_x2_interior.s, bound_x2_interior.e,
				bound_x1_interior.s, bound_x1_interior.e,
				bound_x3_interior.s, bound_x3_interior.e + 1,
        KOKKOS_LAMBDA(const int j, const int i, const int k) {
					Real primitiveLeft[NPRIM];
					Real primitiveRight[NPRIM];
					for(int index = 0; index < NPRIM; ++index) {
						primitiveLeft[index] = primitive(index, k - 1, j, i) +
																		0.5 * InterpolateMC(primitive(index, k - 2, j, i),
																										primitive(index, k - 1, j, i),
																										primitive(index, k, j, i));
						primitiveRight[index] = primitive(index, k, j, i) -
																		0.5 * InterpolateMC(primitive(index, k - 1, j, i),
																										primitive(index, k, j, i),
																										primitive(index, k + 1, j, i));
					}

					Real conservativeLeft[NPRIM];
					Real conservativeRight[NPRIM];
					Real fluxLeft[NPRIM];
					Real fluxRight[NPRIM];

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
					CalculateAlfvenVelocity(
							kAdiabaticIndex, primitiveLeft, gcovFace, gconFace, X3DIR,
							maximumAlfvenVelocityLeft, minimumAlfvenVelocityLeft);
					CalculateAlfvenVelocity(
							kAdiabaticIndex, primitiveRight, gcovFace, gconFace,
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
						conservative.flux(X3DIR, index, k, j, i) =
								(MaximumAlfvenVelocityCenter * fluxLeft[index] + MinimumAlfvenVelocityCenter * fluxRight[index]
                - MaximumAlfvenVelocityCenter * MinimumAlfvenVelocityCenter * (conservativeRight[index] - conservativeLeft[index])) /
                (MaximumAlfvenVelocityCenter + MinimumAlfvenVelocityCenter + 1e-20);
					}
				});

  return TaskStatus::complete;
}
