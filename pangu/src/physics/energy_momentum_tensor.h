// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_PHYSICS_ENERGYMOMENTUMTENSOR_H
#define PANGU_SRC_PHYSICS_ENERGYMOMENTUMTENSOR_H
#include "basic_types.hpp"
#include "initialization/variable_mnemonics.h"
#include "metric/tensor_algebra.h"
#include "physics/state_calculation.h"

KOKKOS_INLINE_FUNCTION
void CalculateEnergyMomentumTensor(
    const parthenon::Real gamma,
    parthenon::Real prim[NPRIM],
    const parthenon::Real gcov[4][4], const parthenon::Real gcon[4][4],
    const int dir, parthenon::Real t_dir[4]) {
  State state;
  CalculateState(prim, gcov, gcon, state);

  const auto p_gas = (gamma - 1.) * prim[ENY];
  const auto h = prim[RHO] + gamma * prim[ENY];
  const auto e_tot = h + state.bsq;
  const auto p_tot = p_gas + 0.5 * state.bsq;

  for (int m = 0; m < 4; ++m) {
    t_dir[m] = e_tot * state.ucon[dir] * state.ucov[m] + p_tot * (dir == m) -
               state.bcon[dir] * state.bcov[m];
  }
}

#endif  // PANGU_SRC_PHYSICS_ENERGYMOMENTUMTENSOR_H
