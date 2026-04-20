// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/metric module defines tensor_algebra.h responsibilities for the
// Pangu runtime. It centers on Kokkos_Core to express core data flow, keep interfaces
// readable, and preserve predictable behavior across task coordination, recovery paths, and
// performance-sensitive execution.

#ifndef PANGU_SRC_METRIC_TENSORALGEBRA_H
#define PANGU_SRC_METRIC_TENSORALGEBRA_H

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"

KOKKOS_INLINE_FUNCTION
parthenon::Real dot4(const parthenon::Real row_vec[4],
                     const parthenon::Real col_vec[4]) {
  return row_vec[0] * col_vec[0] + row_vec[1] * col_vec[1] +
         row_vec[2] * col_vec[2] + row_vec[3] * col_vec[3];
}

KOKKOS_INLINE_FUNCTION
parthenon::Real dot3(const parthenon::Real row_vec[3],
                     const parthenon::Real col_vec[3],
                     const parthenon::Real metric_tensor[3][3]) {
  return row_vec[0] * metric_tensor[0][0] * col_vec[0] +
         row_vec[0] * metric_tensor[0][1] * col_vec[1] +
         row_vec[0] * metric_tensor[0][2] * col_vec[2] +
         row_vec[1] * metric_tensor[1][0] * col_vec[0] +
         row_vec[1] * metric_tensor[1][1] * col_vec[1] +
         row_vec[1] * metric_tensor[1][2] * col_vec[2] +
         row_vec[2] * metric_tensor[2][0] * col_vec[0] +
         row_vec[2] * metric_tensor[2][1] * col_vec[1] +
         row_vec[2] * metric_tensor[2][2] * col_vec[2];
}

KOKKOS_INLINE_FUNCTION
parthenon::Real square3(const parthenon::Real vec[3],
                        const parthenon::Real metric_tensor[3][3]) {
  return dot3(vec, vec, metric_tensor);
}

KOKKOS_INLINE_FUNCTION
parthenon::Real determinant(const parthenon::Real matrix[4][4]) {
  return matrix[0][0] *
             (matrix[1][1] *
                  (matrix[2][2] * matrix[3][3] - matrix[2][3] * matrix[3][2]) -
              matrix[1][2] *
                  (matrix[2][1] * matrix[3][3] - matrix[2][3] * matrix[3][1]) +
              matrix[1][3] *
                  (matrix[2][1] * matrix[3][2] - matrix[2][2] * matrix[3][1])) -
         matrix[0][1] *
             (matrix[1][0] *
                  (matrix[2][2] * matrix[3][3] - matrix[2][3] * matrix[3][2]) -
              matrix[1][2] *
                  (matrix[2][0] * matrix[3][3] - matrix[2][3] * matrix[3][0]) +
              matrix[1][3] *
                  (matrix[2][0] * matrix[3][2] - matrix[2][2] * matrix[3][0])) +
         matrix[0][2] *
             (matrix[1][0] *
                  (matrix[2][1] * matrix[3][3] - matrix[2][3] * matrix[3][1]) -
              matrix[1][1] *
                  (matrix[2][0] * matrix[3][3] - matrix[2][3] * matrix[3][0]) +
              matrix[1][3] *
                  (matrix[2][0] * matrix[3][1] - matrix[2][1] * matrix[3][0])) -
         matrix[0][3] *
             (matrix[1][0] *
                  (matrix[2][1] * matrix[3][2] - matrix[2][2] * matrix[3][1]) -
              matrix[1][1] *
                  (matrix[2][0] * matrix[3][2] - matrix[2][2] * matrix[3][0]) +
              matrix[1][2] *
                  (matrix[2][0] * matrix[3][1] - matrix[2][1] * matrix[3][0]));
}

KOKKOS_INLINE_FUNCTION
int invert(const parthenon::Real matrix[4][4], parthenon::Real inverse[4][4]) {
  parthenon::Real augmented[4][8];

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      augmented[i][j] = matrix[i][j];
      augmented[i][j + 4] = (i == j) ? 1.0 : 0.0;
    }
  }

  for (int i = 0; i < 4; ++i) {
    if (augmented[i][i] == 0.0) {
      int swap_row = -1;
      for (int j = i + 1; j < 4; ++j) {
        if (augmented[j][i] != 0.0) {
          swap_row = j;
          break;
        }
      }
      if (swap_row == -1) {
        return 0;
      }
      for (int j = 0; j < 8; ++j) {
        const parthenon::Real tmp = augmented[i][j];
        augmented[i][j] = augmented[swap_row][j];
        augmented[swap_row][j] = tmp;
      }
    }

    const parthenon::Real pivot = augmented[i][i];
    for (int j = 0; j < 8; ++j) {
      augmented[i][j] /= pivot;
    }

    for (int j = 0; j < 4; ++j) {
      if (j == i) {
        continue;
      }
      const parthenon::Real factor = augmented[j][i];
      for (int k = 0; k < 8; ++k) {
        augmented[j][k] -= factor * augmented[i][k];
      }
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      inverse[i][j] = augmented[i][j + 4];
    }
  }

  return 1;
}

#endif  // PANGU_SRC_METRIC_TENSORALGEBRA_H
