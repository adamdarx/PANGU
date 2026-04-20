// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/metric module defines boyer_lindquist.h responsibilities for the
// Pangu runtime. It centers on Kokkos_Core to express core data flow, keep interfaces
// readable, and preserve predictable behavior across task coordination, recovery paths, and
// performance-sensitive execution.

#ifndef PANGU_SRC_METRIC_BOYERLINDQUIST_H
#define PANGU_SRC_METRIC_BOYERLINDQUIST_H

#include <Kokkos_Core.hpp>

#include <cmath>

#include "basic_types.hpp"

namespace BoyerLindquist {

KOKKOS_INLINE_FUNCTION
void CalculatePhysicalCoordinates(const parthenon::Real x[4],
                                  parthenon::Real y[4], const parthenon::Real h,
                                  const parthenon::Real a) {
  y[0] = x[0];
  y[1] = Kokkos::exp(x[1]);
  y[2] = M_PI_2 * (x[2] + 1.0) + 0.5 * h * Kokkos::sin(M_PI * (x[2] + 1.0));
  y[3] = x[3];
}

KOKKOS_INLINE_FUNCTION
void CalculatePhysicalMetric(const parthenon::Real x[4],
                             parthenon::Real gcov[4][4],
                             const parthenon::Real h, const parthenon::Real a) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      gcov[i][j] = 0.0;
    }
  }

  parthenon::Real y[4];
  CalculatePhysicalCoordinates(x, y, h, a);

  const parthenon::Real r = y[1];
  const parthenon::Real theta = y[2];
  const parthenon::Real cth = Kokkos::cos(theta);
  const parthenon::Real sth = Kokkos::sin(theta);
  const parthenon::Real sth2 = sth * sth;
  const parthenon::Real r2 = r * r;
  const parthenon::Real a2 = a * a;

  const parthenon::Real mu = 1.0 + a2 * cth * cth / r2;
  const parthenon::Real delta = 1.0 - 2.0 / r + a2 / r2;
  const parthenon::Real inv_r_mu = 1.0 / (r * mu);

  gcov[0][0] = -1.0 + 2.0 * inv_r_mu;
  gcov[0][3] = -2.0 * a * sth2 * inv_r_mu;

  gcov[1][1] = mu / delta;
  gcov[2][2] = r2 / mu;

  gcov[3][0] = gcov[0][3];
  gcov[3][3] = r2 * sth2 * (1.0 + a2 / r2 + 2.0 * a2 * sth2 / (r * r2 * mu));
}

KOKKOS_INLINE_FUNCTION
void CalculateCodeMetric(const parthenon::Real x_code[4],
                         parthenon::Real gcov_code[4][4],
                         const parthenon::Real h, const parthenon::Real a) {
  constexpr parthenon::Real delta = 1.0e-5;

  parthenon::Real gcov_physical[4][4];
  CalculatePhysicalMetric(x_code, gcov_physical, h, a);

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
    CalculatePhysicalCoordinates(xh, yh, h, a);
    CalculatePhysicalCoordinates(xl, yl, h, a);

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

}  

#endif  // PANGU_SRC_METRIC_BOYERLINDQUIST_H
