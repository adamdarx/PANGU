// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_METRIC_CKS_H
#define PANGU_SRC_METRIC_CKS_H

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"

namespace CKS {

KOKKOS_INLINE_FUNCTION
void CalculatePhysicalCoordinates(const parthenon::Real x[4],
                                  parthenon::Real y[4]) {
  y[0] = x[0];
  y[1] = x[1];
  y[2] = x[2];
  y[3] = x[3];
}

KOKKOS_INLINE_FUNCTION
void CalculatePhysicalMetric(const parthenon::Real x[4],
                             parthenon::Real gcov[4][4],
                             const parthenon::Real a) {

  const parthenon::Real xx = x[1];
  const parthenon::Real yy = x[2];
  const parthenon::Real zz = x[3];

  const parthenon::Real r2_cart = xx * xx + yy * yy + zz * zz;
  const parthenon::Real a2 = a * a;
  const parthenon::Real tmp = r2_cart - a2;
  const parthenon::Real sqrtarg = tmp * tmp + 4.0 * a2 * zz * zz;
  const parthenon::Real r = Kokkos::sqrt(0.5 * (tmp + Kokkos::sqrt(sqrtarg)));
  const parthenon::Real r_safe = (r > 1.0e-15) ? r : 1.0e-15;

  const parthenon::Real r2 = r_safe * r_safe;
  const parthenon::Real r3 = r2 * r_safe;
  const parthenon::Real r4 = r2 * r2;

  const parthenon::Real xi_denom = r4 + a2 * zz * zz;
  const parthenon::Real xi = (xi_denom > 0.0) ? (2.0 * r3 / xi_denom) : 0.0;

  const parthenon::Real denom_l = r2 + a2;
  const parthenon::Real l[4] = {
      1.0,
      (r_safe * xx + a * yy) / denom_l,
      (r_safe * yy - a * xx) / denom_l,
      zz / r_safe,
  };

  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      gcov[mu][nu] = 0.0;
    }
  }

  gcov[0][0] = -1.0;
  gcov[1][1] = 1.0;
  gcov[2][2] = 1.0;
  gcov[3][3] = 1.0;

  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      gcov[mu][nu] += xi * l[mu] * l[nu];
    }
  }
}

KOKKOS_INLINE_FUNCTION
void CalculateCodeMetric(const parthenon::Real x_code[4],
                         parthenon::Real gcov_code[4][4],
                         const parthenon::Real a) {
  constexpr parthenon::Real delta = 1.0e-5;

  parthenon::Real gcov_physical[4][4];
  CalculatePhysicalMetric(x_code, gcov_physical, a);

  parthenon::Real jac[4][4];
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      jac[row][col] = 0.0;
    }
  }

  for (int col = 0; col < 4; ++col) {
    parthenon::Real xh[4];
    parthenon::Real xl[4];
    for (int idx = 0; idx < 4; ++idx) {
      xh[idx] = x_code[idx];
      xl[idx] = x_code[idx];
    }
    xh[col] += delta;
    xl[col] -= delta;

    parthenon::Real yh[4];
    parthenon::Real yl[4];
    CalculatePhysicalCoordinates(xh, yh);
    CalculatePhysicalCoordinates(xl, yl);

    for (int row = 0; row < 4; ++row) {
      jac[row][col] = (yh[row] - yl[row]) / (xh[col] - xl[col]);
    }
  }

  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      gcov_code[row][col] = 0.0;
      for (int p = 0; p < 4; ++p) {
        for (int q = 0; q < 4; ++q) {
          gcov_code[row][col] +=
              gcov_physical[p][q] * jac[row][p] * jac[q][col];
        }
      }
    }
  }
}

}  // namespace CKS

#endif  // PANGU_SRC_METRIC_CKS_H
