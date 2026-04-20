// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/metric module defines minkowski.h responsibilities for the Pangu
// runtime. It centers on Kokkos_Core to express core data flow, keep interfaces readable,
// and preserve predictable behavior across task coordination, recovery paths, and
// performance-sensitive execution.

#ifndef PANGU_SRC_METRIC_MINKOWSKI_H
#define PANGU_SRC_METRIC_MINKOWSKI_H

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"

namespace Minkowski {

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
                             parthenon::Real gcov[4][4]) {
  (void)x;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      gcov[i][j] = (i == j) * (1 - 2 * (i == 0));
    }
  }
}

KOKKOS_INLINE_FUNCTION
void CalculateCodeMetric(const parthenon::Real x_code[4],
                         parthenon::Real gcov_code[4][4]) {
  constexpr parthenon::Real delta = 1.0e-5;

  parthenon::Real gcov_physical[4][4];
  CalculatePhysicalMetric(x_code, gcov_physical);

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

}  

#endif  // PANGU_SRC_METRIC_MINKOWSKI_H
