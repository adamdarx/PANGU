// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "riemann_solver/source_term.h"

#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include <basic_types.hpp>
#include "initialization/variable_mnemonics.h"
#include "physics/energy_momentum_tensor.h"

parthenon::TaskStatus AddGeometricSource(parthenon::MeshData<parthenon::Real> *md,
                                         parthenon::Real dt) {
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

  auto covariant_metric =
      md->PackVariables(std::vector<std::string>{"covariant_metric"});
  auto contravariant_metric =
      md->PackVariables(std::vector<std::string>{"contravariant_metric"});
  auto metric_determinant =
      md->PackVariables(std::vector<std::string>{"metric_determinant"});
  auto connection =
      md->PackVariables(std::vector<std::string>{"connection"});

  pmb0->par_for(
      PARTHENON_AUTO_LABEL, block.s, block.e,
      bound_x3_interior.s, bound_x3_interior.e, bound_x2_interior.s, bound_x2_interior.e,
      bound_x1_interior.s, bound_x1_interior.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real gcov[4][4];
        Real gcon[4][4];
        const auto SqrtAbsMetricDeterminant =
            Kokkos::sqrt(Kokkos::abs(metric_determinant(b, CENTER, k, j, i)));

        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcov[row][col] =
                covariant_metric(b, CENTER * 16 + col * 4 + row, k, j, i);
            gcon[row][col] =
                contravariant_metric(b, CENTER * 16 + col * 4 + row, k, j, i);
          }
        }

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

        Real MixedEnergyMomentumTensor[4][4];
        for (int row = 0; row < 4; ++row) {
          CalculateEnergyMomentumTensor(kAdiabaticIndex, pcarr,
                                             gcov, gcon, row,
                                             MixedEnergyMomentumTensor[row]);
        }

        for (int dir = 0; dir < 4; ++dir) {
          Real contraction = 0.0;
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
              contraction += MixedEnergyMomentumTensor[row][col] *
                             connection(b, col * 16 + dir * 4 + row, k, j, i);
            }
          }

          if (dir == 0) {
            conservative(b, iENY, k, j, i) +=
                contraction * SqrtAbsMetricDeterminant * dt;
          } else {
            conservative(b, iUX + (dir - 1), k, j, i) +=
                contraction * SqrtAbsMetricDeterminant * dt;
          }
        }
      });

  return TaskStatus::complete;
}
