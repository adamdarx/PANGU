// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_PHYSICS_STATECALCULATION_H
#define PANGU_SRC_PHYSICS_STATECALCULATION_H

#include "basic_types.hpp"
#include "initialization/variable_mnemonics.h"
#include "metric/tensor_algebra.h"

struct State {
  parthenon::Real ucon[4];
  parthenon::Real ucov[4];
  parthenon::Real bcon[4];
  parthenon::Real bcov[4];
  parthenon::Real bsq;
};

KOKKOS_INLINE_FUNCTION
void CalculateState(
    parthenon::Real primitive[NPRIM],
    const parthenon::Real gcov[4][4], const parthenon::Real gcon[4][4],
    State &state) {
  const auto alpha = 1.0 / Kokkos::sqrt(-gcon[0][0]);

  parthenon::Real gamma_ij[3][3] = {{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};
  for (int row = 1; row < 4; ++row) {
    for (int col = 1; col < 4; ++col) {
      gamma_ij[row - 1][col - 1] = gcov[row][col];
    }
  }

  const parthenon::Real u_tilde[3] = {primitive[UX1], primitive[UX2],
                                      primitive[UX3]};
  const auto lorentz_sq = 1.0 + square3(u_tilde, gamma_ij);
  const auto lorentz = Kokkos::sqrt(lorentz_sq);

  state.ucon[0] = lorentz / alpha;
  state.ucon[1] = primitive[UX1] - state.ucon[0] * gcon[0][1] * alpha * alpha;
  state.ucon[2] = primitive[UX2] - state.ucon[0] * gcon[0][2] * alpha * alpha;
  state.ucon[3] = primitive[UX3] - state.ucon[0] * gcon[0][3] * alpha * alpha;

  for (int row = 0; row < 4; ++row) {
    state.ucov[row] = 0.0;
    for (int col = 0; col < 4; ++col) {
      state.ucov[row] += gcov[row][col] * state.ucon[col];
    }
  }

  state.bcon[0] = primitive[BX1] * state.ucov[1] +
                  primitive[BX2] * state.ucov[2] +
                  primitive[BX3] * state.ucov[3];

  state.bcon[1] =
      (primitive[BX1] + state.bcon[0] * state.ucon[1]) / state.ucon[0];
  state.bcon[2] =
      (primitive[BX2] + state.bcon[0] * state.ucon[2]) / state.ucon[0];
  state.bcon[3] =
      (primitive[BX3] + state.bcon[0] * state.ucon[3]) / state.ucon[0];

  for (int row = 0; row < 4; ++row) {
    state.bcov[row] = 0.0;
    for (int col = 0; col < 4; ++col) {
      state.bcov[row] += gcov[row][col] * state.bcon[col];
    }
  }

  state.bsq = state.bcon[0] * state.bcov[0] + state.bcon[1] * state.bcov[1] + state.bcon[2] * state.bcov[2] + state.bcon[3] * state.bcov[3];
}

#endif  // PANGU_SRC_PHYSICS_STATECALCULATION_H
