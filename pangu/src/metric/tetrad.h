// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_METRIC_TETRAD_H
#define PANGU_SRC_METRIC_TETRAD_H

#include "basic_types.hpp"
#include "initialization/variable_mnemonics.h"

struct Tetrad {
  parthenon::Real basis[3][3];
  parthenon::Real dual[3][3];
  parthenon::Real scale;
};

KOKKOS_INLINE_FUNCTION
parthenon::Real SpatialDot(const parthenon::Real gamma[3][3],
                           const parthenon::Real a[3],
                           const parthenon::Real b[3]) {
  parthenon::Real dot = 0.0;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      dot += gamma[row][col] * a[row] * b[col];
    }
  }
  return dot;
}

KOKKOS_INLINE_FUNCTION
void NormalizeSpatialVector(const parthenon::Real gamma[3][3],
                            parthenon::Real vector[3]) {
  const parthenon::Real norm =
      Kokkos::sqrt(Kokkos::max(SpatialDot(gamma, vector, vector), 1.0e-300));
  for (int index = 0; index < 3; ++index) {
    vector[index] /= norm;
  }
}

KOKKOS_INLINE_FUNCTION
void BuildTetrad(const parthenon::Real gcov[4][4],
                 const parthenon::Real metric_determinant, const int dir,
                 Tetrad &tetrad) {
  parthenon::Real gamma[3][3];
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      gamma[row][col] = gcov[row + 1][col + 1];
    }
  }

  const int normal = dir - 1;
  const int tangent1 = (normal + 1) % 3;
  const int tangent2 = (normal + 2) % 3;

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      tetrad.basis[row][col] = 0.0;
      tetrad.dual[row][col] = 0.0;
    }
  }

  tetrad.basis[0][normal] = 1.0;
  NormalizeSpatialVector(gamma, tetrad.basis[0]);

  tetrad.basis[1][tangent1] = 1.0;
  parthenon::Real projection =
      SpatialDot(gamma, tetrad.basis[1], tetrad.basis[0]);
  for (int index = 0; index < 3; ++index) {
    tetrad.basis[1][index] -= projection * tetrad.basis[0][index];
  }
  NormalizeSpatialVector(gamma, tetrad.basis[1]);

  tetrad.basis[2][tangent2] = 1.0;
  projection = SpatialDot(gamma, tetrad.basis[2], tetrad.basis[0]);
  for (int index = 0; index < 3; ++index) {
    tetrad.basis[2][index] -= projection * tetrad.basis[0][index];
  }
  projection = SpatialDot(gamma, tetrad.basis[2], tetrad.basis[1]);
  for (int index = 0; index < 3; ++index) {
    tetrad.basis[2][index] -= projection * tetrad.basis[1][index];
  }
  NormalizeSpatialVector(gamma, tetrad.basis[2]);

  for (int local = 0; local < 3; ++local) {
    for (int col = 0; col < 3; ++col) {
      for (int row = 0; row < 3; ++row) {
        tetrad.dual[local][col] += gamma[col][row] * tetrad.basis[local][row];
      }
    }
  }

  tetrad.scale = Kokkos::sqrt(Kokkos::abs(metric_determinant)) *
                 tetrad.basis[0][normal];
}

KOKKOS_INLINE_FUNCTION
void ProjectPrimitiveToTetrad(const Tetrad &tetrad,
                              const parthenon::Real primitive[NPRIM],
                              parthenon::Real local_primitive[NPRIM]) {
  local_primitive[RHO] = primitive[RHO];
  local_primitive[ENY] = primitive[ENY];
  local_primitive[ENT] = primitive[ENT];
  local_primitive[KEL] = primitive[KEL];

  const parthenon::Real velocity[3] = {primitive[UX1], primitive[UX2],
                                      primitive[UX3]};
  const parthenon::Real magnetic_field[3] = {primitive[BX1], primitive[BX2],
                                            primitive[BX3]};
  for (int local = 0; local < 3; ++local) {
    local_primitive[UX1 + local] = 0.0;
    local_primitive[BX1 + local] = 0.0;
    for (int global = 0; global < 3; ++global) {
      local_primitive[UX1 + local] +=
          tetrad.dual[local][global] * velocity[global];
      local_primitive[BX1 + local] +=
          tetrad.dual[local][global] * magnetic_field[global];
    }
  }
}

KOKKOS_INLINE_FUNCTION
void ProjectTetradFluxToGlobal(const Tetrad &tetrad,
                               const parthenon::Real local_flux[NPRIM],
                               parthenon::Real global_flux[NPRIM]) {
  global_flux[RHO] = tetrad.scale * local_flux[RHO];
  global_flux[ENY] = tetrad.scale * (local_flux[RHO] - local_flux[ENY]);

  for (int global = 0; global < 3; ++global) {
    global_flux[UX1 + global] = 0.0;
    global_flux[BX1 + global] = 0.0;
    for (int local = 0; local < 3; ++local) {
      global_flux[UX1 + global] +=
          tetrad.dual[local][global] * local_flux[UX1 + local];
      global_flux[BX1 + global] +=
          tetrad.basis[local][global] * local_flux[BX1 + local];
    }
    global_flux[UX1 + global] *= tetrad.scale;
    global_flux[BX1 + global] *= tetrad.scale;
  }

  global_flux[ENT] = tetrad.scale * local_flux[ENT];
  global_flux[KEL] = tetrad.scale * local_flux[KEL];
}

#endif  // PANGU_SRC_METRIC_TETRAD_H
