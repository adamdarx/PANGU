// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_PHYSICS_CONTRAVARIANTFLUX_H
#define PANGU_SRC_PHYSICS_CONTRAVARIANTFLUX_H

#include "basic_types.hpp"
#include "initialization/variable_mnemonics.h"
#include "physics/energy_momentum_tensor.h"
#include "physics/state_calculation.h"

KOKKOS_INLINE_FUNCTION
void CalculateContravariantFlux(
    const parthenon::Real gamma,
    parthenon::Real prim[NPRIM],
    const parthenon::Real gcov[4][4], const parthenon::Real gcon[4][4],
    const parthenon::Real gdet, const int dir,
    parthenon::Real flux[NPRIM]) {
  State state;
  CalculateState(prim, gcov, gcon, state);
  parthenon::Real t_dir[4];
  CalculateEnergyMomentumTensor(gamma, prim, gcov, gcon, dir, t_dir);

  flux[RHO] = prim[RHO] * state.ucon[dir];
  flux[ENY] = t_dir[0] + flux[RHO];
  flux[UX1] = t_dir[1];
  flux[UX2] = t_dir[2];
  flux[UX3] = t_dir[3];
  flux[BX1] = state.bcon[1] * state.ucon[dir] - state.bcon[dir] * state.ucon[1];
  flux[BX2] = state.bcon[2] * state.ucon[dir] - state.bcon[dir] * state.ucon[2];
  flux[BX3] = state.bcon[3] * state.ucon[dir] - state.bcon[dir] * state.ucon[3];
  flux[ENT] = prim[ENT] * flux[RHO];
  flux[KEL] = prim[KEL] * flux[RHO];

  const auto sqrt_abs_gdet = Kokkos::sqrt(Kokkos::abs(gdet));
  for (int n = 0; n < NPRIM; ++n) {
    flux[n] *= sqrt_abs_gdet;
  }
}

#endif  // PANGU_SRC_PHYSICS_CONTRAVARIANTFLUX_H
