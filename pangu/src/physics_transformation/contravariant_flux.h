// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/physics_transformation module defines contravariant_flux.h
// responsibilities for the Pangu runtime. It centers on basic_types to express core data
// flow, keep interfaces readable, and preserve predictable behavior across task
// coordination, recovery paths, and performance-sensitive execution.

#ifndef PANGU_SRC_PHYSICSTRANSFORMATION_CONTRAVARIANTFLUX_H
#define PANGU_SRC_PHYSICSTRANSFORMATION_CONTRAVARIANTFLUX_H

#include "basic_types.hpp"
#include "initialization/variable_mnemonics.h"
#include "physics_transformation/energy_momentum_tensor.h"
#include "physics_transformation/state_calculation.h"

KOKKOS_INLINE_FUNCTION
void CalculateContravariantFluxSRMHD(
    const parthenon::Real gamma,
    const parthenon::Real prim[NPRIM], const int dir,
    parthenon::Real flux[NPRIM]) {
  State state;
  CalculateSRMHDState(prim, state);

  parthenon::Real t_dir[4];
  CalculateEnergyMomentumTensorSRMHD(gamma, prim, dir, t_dir);

  flux[RHO] = prim[RHO] * state.ucon[dir];
  flux[ENY] = t_dir[0] + flux[RHO];
  flux[UX1] = t_dir[1];
  flux[UX2] = t_dir[2];
  flux[UX3] = t_dir[3];
  flux[BX1] = state.bcon[1] * state.ucon[dir] - state.bcon[dir] * state.ucon[1];
  flux[BX2] = state.bcon[2] * state.ucon[dir] - state.bcon[dir] * state.ucon[2];
  flux[BX3] = state.bcon[3] * state.ucon[dir] - state.bcon[dir] * state.ucon[3];
}

KOKKOS_INLINE_FUNCTION
void CalculateContravariantFluxGRMHD(
    const parthenon::Real gamma,
    const parthenon::Real prim[NPRIM],
    const parthenon::Real gcov[4][4], const parthenon::Real gcon[4][4],
    const parthenon::Real gdet, const int dir,
    parthenon::Real flux[NPRIM]) {
  State state;
  CalculateGRMHDState(prim, gcov, gcon, state);
  parthenon::Real t_dir[4];
  CalculateEnergyMomentumTensorGRMHD(gamma, prim, gcov, gcon, dir, t_dir);

  flux[RHO] = prim[RHO] * state.ucon[dir];
  flux[ENY] = t_dir[0] + flux[RHO];
  flux[UX1] = t_dir[1];
  flux[UX2] = t_dir[2];
  flux[UX3] = t_dir[3];
  flux[BX1] = state.bcon[1] * state.ucon[dir] - state.bcon[dir] * state.ucon[1];
  flux[BX2] = state.bcon[2] * state.ucon[dir] - state.bcon[dir] * state.ucon[2];
  flux[BX3] = state.bcon[3] * state.ucon[dir] - state.bcon[dir] * state.ucon[3];

  const auto sqrt_abs_gdet = Kokkos::sqrt(Kokkos::abs(gdet));
  for (int n = 0; n < NPRIM; ++n) {
    flux[n] *= sqrt_abs_gdet;
  }
}

#endif  // PANGU_SRC_PHYSICSTRANSFORMATION_CONTRAVARIANTFLUX_H
