// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/physics_transformation module defines energy_momentum_tensor.h
// responsibilities for the Pangu runtime. It centers on basic_types to express core data
// flow, keep interfaces readable, and preserve predictable behavior across task
// coordination, recovery paths, and performance-sensitive execution.

#ifndef PANGU_SRC_PHYSICSTRANSFORMATION_ENERGYMOMENTUMTENSOR_H
#define PANGU_SRC_PHYSICSTRANSFORMATION_ENERGYMOMENTUMTENSOR_H
#include "basic_types.hpp"
#include "initialization/variable_mnemonics.h"
#include "metric/tensor_algebra.h"
#include "physics_transformation/state_calculation.h"

KOKKOS_INLINE_FUNCTION
void CalculateEnergyMomentumTensorSRMHD(
    const parthenon::Real gamma,
    const parthenon::Real prim[NPRIM], const int dir,
    parthenon::Real t_dir[4]) {
  State state;
  CalculateSRMHDState(prim, state);

  const auto b_sq = dot4(state.bcon, state.bcov);
  const auto h = prim[RHO] + gamma * prim[ENY];
  const auto e_tot = b_sq + h;
  const auto p_gas = (gamma - 1.) * prim[ENY];
  const auto p_tot = p_gas + 0.5 * b_sq;

  for (int m = 0; m < 4; ++m) {
    t_dir[m] = e_tot * state.ucon[dir] * state.ucov[m] + p_tot * (dir == m) -
               state.bcon[dir] * state.bcov[m];
  }
}

KOKKOS_INLINE_FUNCTION
void CalculateEnergyMomentumTensorGRMHD(
    const parthenon::Real gamma,
    const parthenon::Real prim[NPRIM],
    const parthenon::Real gcov[4][4], const parthenon::Real gcon[4][4],
    const int dir, parthenon::Real t_dir[4]) {
  State state;
  CalculateGRMHDState(prim, gcov, gcon, state);

  const auto b_sq = dot4(state.bcon, state.bcov);
  const auto p_gas = (gamma - 1.) * prim[ENY];
  const auto h = prim[RHO] + gamma * prim[ENY];
  const auto e_tot = h + b_sq;
  const auto p_tot = p_gas + 0.5 * b_sq;

  for (int m = 0; m < 4; ++m) {
    t_dir[m] = e_tot * state.ucon[dir] * state.ucov[m] + p_tot * (dir == m) -
               state.bcon[dir] * state.bcov[m];
  }
}

#endif  // PANGU_SRC_PHYSICSTRANSFORMATION_ENERGYMOMENTUMTENSOR_H
