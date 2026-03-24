#pragma once

#include <basic_types.hpp>
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"

#include "Invertor.hpp"

namespace Transform {

KOKKOS_INLINE_FUNCTION
parthenon::Real LapseFromMetric(const parthenon::Real gcon[4][4]) {
  return Kokkos::sqrt(-1.0 / gcon[0][0]);
}

KOKKOS_INLINE_FUNCTION
parthenon::Real SpatialContravariantMetric(const parthenon::Real gcon[4][4], const int i,
                                           const int j) {
  return gcon[i][j] - gcon[0][i] * gcon[0][j] / gcon[0][0];
}

KOKKOS_INLINE_FUNCTION
void NormalizeConservedBySqrtAbsG(const parthenon::Real conservedGR[PrimitiveVariableNumber],
                                  const parthenon::Real gdet,
                                  parthenon::Real normalized[PrimitiveVariableNumber]) {
  const auto inv_sqrt_abs_g = 1.0 / Kokkos::sqrt(Kokkos::abs(gdet));
  for (int n = 0; n < PrimitiveVariableNumber; ++n) {
    normalized[n] = conservedGR[n] * inv_sqrt_abs_g;
  }
}

// Convert GR conserved variables to the SR form required by the SRMHD inversion kernel.
KOKKOS_INLINE_FUNCTION
void TransformConservedGRMHDToSRMHD(const parthenon::Real conservedGR[PrimitiveVariableNumber],
                                    const parthenon::Real gcov[4][4],
                                    const parthenon::Real gcon[4][4],
                                    const parthenon::Real gdet,
                                    parthenon::Real conservedSR[PrimitiveVariableNumber],
                                    parthenon::Real &s2, parthenon::Real &b2,
                                    parthenon::Real &rpar) {
  parthenon::Real u[PrimitiveVariableNumber];
  NormalizeConservedBySqrtAbsG(conservedGR, gdet, u);

  const auto alpha = LapseFromMetric(gcon);
  conservedSR[RHO] = alpha * u[RHO];

  const auto t0_0 = u[UU] - u[RHO];
  conservedSR[UU] = gcon[0][0] * t0_0 + gcon[0][1] * u[U1] + gcon[0][2] * u[U2] +
                    gcon[0][3] * u[U3];
  conservedSR[UU] *= (-1.0 / gcon[0][0]);
  conservedSR[UU] -= conservedSR[RHO];

  const auto m1l = alpha * u[U1];
  const auto m2l = alpha * u[U2];
  const auto m3l = alpha * u[U3];

  conservedSR[U1] = SpatialContravariantMetric(gcon, 1, 1) * m1l +
                    SpatialContravariantMetric(gcon, 1, 2) * m2l +
                    SpatialContravariantMetric(gcon, 1, 3) * m3l;
  conservedSR[U2] = SpatialContravariantMetric(gcon, 2, 1) * m1l +
                    SpatialContravariantMetric(gcon, 2, 2) * m2l +
                    SpatialContravariantMetric(gcon, 2, 3) * m3l;
  conservedSR[U3] = SpatialContravariantMetric(gcon, 3, 1) * m1l +
                    SpatialContravariantMetric(gcon, 3, 2) * m2l +
                    SpatialContravariantMetric(gcon, 3, 3) * m3l;

  s2 = m1l * conservedSR[U1] + m2l * conservedSR[U2] + m3l * conservedSR[U3];

  conservedSR[B1] = alpha * u[B1];
  conservedSR[B2] = alpha * u[B2];
  conservedSR[B3] = alpha * u[B3];

  b2 = gcov[1][1] * conservedSR[B1] * conservedSR[B1] +
       gcov[2][2] * conservedSR[B2] * conservedSR[B2] +
       gcov[3][3] * conservedSR[B3] * conservedSR[B3] +
       2.0 * (gcov[1][2] * conservedSR[B1] * conservedSR[B2] +
              gcov[1][3] * conservedSR[B1] * conservedSR[B3] +
              gcov[2][3] * conservedSR[B2] * conservedSR[B3]);

  rpar = (conservedSR[B1] * m1l + conservedSR[B2] * m2l + conservedSR[B3] * m3l) /
         conservedSR[RHO];
}

KOKKOS_INLINE_FUNCTION
parthenon::Real CalculateLorentzFactorGR(const parthenon::Real primitive[PrimitiveVariableNumber],
                                         const parthenon::Real gcov[4][4]) {
  const auto q = gcov[1][1] * primitive[U1] * primitive[U1] +
                 gcov[2][2] * primitive[U2] * primitive[U2] +
                 gcov[3][3] * primitive[U3] * primitive[U3] +
                 2.0 * (gcov[1][2] * primitive[U1] * primitive[U2] +
                        gcov[1][3] * primitive[U1] * primitive[U3] +
                        gcov[2][3] * primitive[U2] * primitive[U3]);
  return Kokkos::sqrt(1.0 + q);
}

KOKKOS_INLINE_FUNCTION
void ApplyVelocityCeilingGR(parthenon::Real primitive[PrimitiveVariableNumber],
                            const parthenon::Real gcov[4][4],
                            const parthenon::Real gamma_max) {
  if (gamma_max <= 1.0) {
    return;
  }

  const auto lorentz = CalculateLorentzFactorGR(primitive, gcov);
  if (lorentz <= gamma_max) {
    return;
  }

  const auto factor = Kokkos::sqrt((gamma_max * gamma_max - 1.0) /
                                   (lorentz * lorentz - 1.0));
  primitive[U1] *= factor;
  primitive[U2] *= factor;
  primitive[U3] *= factor;
}

parthenon::TaskStatus TransformConservativeGRMHDToSRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  const auto MeshblockPointer = resource->GetBlockPointer();
  const auto BoundX1 = MeshblockPointer->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto BoundX2 = MeshblockPointer->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto BoundX3 = MeshblockPointer->cellbounds.GetBoundsK(IndexDomain::interior);

  PackIndexMap conservativeIndexMap;
  const std::vector<std::string> ConservativeTags = {"Conservative"};
  auto conservative = resource->PackVariables(ConservativeTags, conservativeIndexMap);

  PackIndexMap covariantMetricIndexMap;
  const std::vector<std::string> CovariantMetricTags = {"CovariantMetric"};
  const auto CovariantMetric = resource->PackVariables(CovariantMetricTags, covariantMetricIndexMap);

  PackIndexMap contravariantMetricIndexMap;
  const std::vector<std::string> ContravariantMetricTags = {"ContravariantMetric"};
  const auto ContravariantMetric = resource->PackVariables(ContravariantMetricTags, contravariantMetricIndexMap);

  PackIndexMap metricDeterminantIndexMap;
  const std::vector<std::string> MetricDeterminantTags = {"MetricDeterminant"};
  const auto MetricDeterminant = resource->PackVariables(MetricDeterminantTags, metricDeterminantIndexMap);

  MeshblockPointer->par_for(
      PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real gcov[4][4];
        Real gcon[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcov[row][col] = CovariantMetric(row * 4 + col, k, j, i);
            gcon[row][col] = ContravariantMetric(row * 4 + col, k, j, i);
          }
        }

        Real conservedGR[PrimitiveVariableNumber];
        Real conservedSR[PrimitiveVariableNumber];
        for (int n = 0; n < PrimitiveVariableNumber; ++n) {
          conservedGR[n] = conservative(n, k, j, i);
        }

        Real s2 = 0.0, b2 = 0.0, rpar = 0.0;
        TransformConservedGRMHDToSRMHD(conservedGR, gcov, gcon, MetricDeterminant(0, k, j, i),
                                       conservedSR, s2, b2, rpar);

        (void)s2;
        (void)b2;
        (void)rpar;

        conservative(DensityIndex, k, j, i) = conservedSR[RHO];
        conservative(EnergyIndex, k, j, i) = conservedSR[UU] + conservedSR[RHO];
        conservative(WeightedVelocityX1, k, j, i) = conservedSR[U1];
        conservative(WeightedVelocityX2, k, j, i) = conservedSR[U2];
        conservative(WeightedVelocityX3, k, j, i) = conservedSR[U3];
        conservative(MagneticFieldX1, k, j, i) = conservedSR[B1];
        conservative(MagneticFieldX2, k, j, i) = conservedSR[B2];
        conservative(MagneticFieldX3, k, j, i) = conservedSR[B3];
      });

  return TaskStatus::complete;
}

parthenon::TaskStatus TransformPrimitiveSRMHDToGRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  const auto MeshblockPointer = resource->GetBlockPointer();
  const auto BoundX1 = MeshblockPointer->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto BoundX2 = MeshblockPointer->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto BoundX3 = MeshblockPointer->cellbounds.GetBoundsK(IndexDomain::interior);

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> PrimitiveTags = {"Density", "Energy", "WeightedVelocity", "MagneticField"};
  auto primitive = resource->PackVariables(PrimitiveTags, primitiveIndexMap);

  PackIndexMap contravariantMetricIndexMap;
  const std::vector<std::string> ContravariantMetricTags = {"ContravariantMetric"};
  const auto ContravariantMetric = resource->PackVariables(ContravariantMetricTags, contravariantMetricIndexMap);

  MeshblockPointer->par_for(
      PARTHENON_AUTO_LABEL, BoundX3.s, BoundX3.e, BoundX2.s, BoundX2.e, BoundX1.s, BoundX1.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real gcon[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcon[row][col] = ContravariantMetric(row * 4 + col, k, j, i);
          }
        }

        const Real alpha = LapseFromMetric(gcon);
        primitive(MagneticFieldX1, k, j, i) /= alpha;
        primitive(MagneticFieldX2, k, j, i) /= alpha;
        primitive(MagneticFieldX3, k, j, i) /= alpha;
      });

  return TaskStatus::complete;
}

} // namespace Transform
