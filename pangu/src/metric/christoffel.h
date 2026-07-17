// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_METRIC_CHRISTOFFEL_H
#define PANGU_SRC_METRIC_CHRISTOFFEL_H

#include <Kokkos_Core.hpp>

#include <basic_types.hpp>

#include "metric/BL.h"
#include "metric/CKS.h"
#include "metric/MKS.h"
#include "metric/minkowski.h"
#include "metric/tensor_algebra.h"

// Metric type enumeration matching the "metric_type" parameter string values.
// Used to dispatch to the correct analytical metric function at runtime on device.
namespace MetricType {
enum : int { Minkowski = 0, BL = 1, CKS = 2, MKS = 3 };
}

// ---------------------------------------------------------------------------
// Compute the code-basis covariant metric at coordinate position x_code[4]
// by dispatching to the appropriate analytical metric function.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION void
ComputeCodeMetricAt(const int mtype, const parthenon::Real x_code[4],
                    const parthenon::Real a, const parthenon::Real h,
                    parthenon::Real gcov_code[4][4]) {
  switch (mtype) {
    case MetricType::BL:
      BL::CalculateCodeMetric(x_code, gcov_code, h, a);
      break;
    case MetricType::CKS:
      CKS::CalculateCodeMetric(x_code, gcov_code, a);
      break;
    case MetricType::MKS:
      MKS::CalculateCodeMetric(x_code, gcov_code, h, a);
      break;
    case MetricType::Minkowski:
    default:
      Minkowski::CalculateCodeMetric(x_code, gcov_code);
      break;
  }
}

// ---------------------------------------------------------------------------
// Compute the full metric at a code-coordinate position: gcov, gcon, and
// the raw determinant gdet.  Users needing the volume element compute
// sqrt(|gdet|) on their own.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION void
ComputeMetricAtLocation(const int mtype, const parthenon::Real x_code[4],
                        const parthenon::Real a, const parthenon::Real h,
                        parthenon::Real gcov[4][4],
                        parthenon::Real gcon[4][4],
                        parthenon::Real &gdet) {
  ComputeCodeMetricAt(mtype, x_code, a, h, gcov);
  invert(gcov, gcon);
  gdet = determinant(gcov);
}
// given computational coordinate position x_code[4], using central finite
// differences (MetricDiffDelta = 1e-5) of the analytical code-basis metric.
//
// This replicates the algorithm previously used in every problem generator
// to populate the stored "connection" field.  For Minkowski spacetime the
// connection is trivially zero and the function returns immediately.
//
// Result: conn[μ][α][β] = Γ^μ_{αβ}
//   This matches the stored connection field indexing:
//     connection(b, col*16 + dir*4 + row, k, j, i) = Γ^{col}_{dir,row}
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION void
ComputeChristoffelConnection(const int mtype,
                             const parthenon::Real x_code[4],
                             const parthenon::Real a, const parthenon::Real h,
                             parthenon::Real conn[4][4][4]) {
  using parthenon::Real;

  // Zero initialize
  for (int ii = 0; ii < 4; ++ii) {
    for (int jj = 0; jj < 4; ++jj) {
      for (int kk = 0; kk < 4; ++kk) {
        conn[ii][jj][kk] = 0.0;
      }
    }
  }

  if (mtype == MetricType::Minkowski) return;

  // Compute metric at cell center
  Real gcov[4][4], gcon[4][4];
  Real gdet;
  ComputeMetricAtLocation(mtype, x_code, a, h, gcov, gcon, gdet);

  // Compute metric derivatives via central finite differences
  constexpr Real delta = 1.0e-5;
  Real dgcov[4][4][4] = {};  // dgcov[dir][row][col] = ∂_{dir} g_{row,col}

  for (int dir = 0; dir < 4; ++dir) {
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        dgcov[dir][row][col] = 0.0;
      }
    }
    if (dir > 0) {  // spatial directions only; time derivative = 0
      Real xp[4] = {x_code[0], x_code[1], x_code[2], x_code[3]};
      Real xm[4] = {x_code[0], x_code[1], x_code[2], x_code[3]};
      xp[dir] += delta;
      xm[dir] -= delta;
      Real gp[4][4], gm[4][4];
      ComputeCodeMetricAt(mtype, xp, a, h, gp);
      ComputeCodeMetricAt(mtype, xm, a, h, gm);
      for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
          dgcov[dir][row][col] = (gp[row][col] - gm[row][col]) / (2.0 * delta);
        }
      }
    }
  }

  // Christoffel symbols of the first kind (fully covariant)
  // conn_cov[ii][jj][kk] = Γ_{ii, jj, kk}
  // = 0.5 * (∂_{jj} g_{ii,kk} + ∂_{kk} g_{ii,jj} - ∂_{ii} g_{jj,kk})
  Real conn_cov[4][4][4];
  for (int ii = 0; ii < 4; ++ii) {
    for (int jj = 0; jj < 4; ++jj) {
      for (int kk = 0; kk < 4; ++kk) {
        conn_cov[ii][jj][kk] =
            0.5 * (dgcov[jj][ii][kk] + dgcov[kk][ii][jj] - dgcov[ii][jj][kk]);
      }
    }
  }

  // Raise first index: Γ^{ii}_{jj,kk} = g^{ii,ll} * Γ_{ll,jj,kk}
  for (int ii = 0; ii < 4; ++ii) {
    for (int jj = 0; jj < 4; ++jj) {
      for (int kk = 0; kk < 4; ++kk) {
        Real conn_val = 0.0;
        for (int ll = 0; ll < 4; ++ll) {
          conn_val += gcon[ii][ll] * conn_cov[ll][jj][kk];
        }
        conn[ii][jj][kk] = conn_val;
      }
    }
  }
}

#endif  // PANGU_SRC_METRIC_CHRISTOFFEL_H
