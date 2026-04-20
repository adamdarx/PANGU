// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/physics_transformation module defines alfven_velocity.h
// responsibilities for the Pangu runtime. It centers on basic_types to express core data
// flow, keep interfaces readable, and preserve predictable behavior across task
// coordination, recovery paths, and performance-sensitive execution.

#ifndef PANGU_SRC_PHYSICSTRANSFORMATION_ALFVENVELOCITY_H
#define PANGU_SRC_PHYSICSTRANSFORMATION_ALFVENVELOCITY_H
#include "basic_types.hpp"
#include "initialization/variable_mnemonics.h"
#include "metric/tensor_algebra.h"
#include "physics_transformation/state_calculation.h"

KOKKOS_INLINE_FUNCTION
void CalculateAlfvenVelocitySRMHD(
    const parthenon::Real gamma,
    const parthenon::Real prim[NPRIM], const int dir,
    parthenon::Real &v_max, parthenon::Real &v_min) {
  State state;
  CalculateSRMHDState(prim, state);

  const auto b_sq = dot4(state.bcon, state.bcov);
  const auto h = prim[RHO] + gamma * prim[ENY];
  const auto h_tot = b_sq + h;
  const auto va_sq = b_sq / h_tot;
  const auto cs_sq = gamma * (gamma - 1.) * prim[ENY] / h;
  auto cf_sq = cs_sq + va_sq - cs_sq * va_sq;

  if (cf_sq < 0.) {
    cf_sq = 1e-10;
  } else if (cf_sq > 1.) {
    cf_sq = 1.;
  }

  parthenon::Real ncon[4] = {0., 0., 0., 0.};
  parthenon::Real ncov[4] = {0., 0., 0., 0.};
  parthenon::Real tcon[4] = {-1., 0., 0., 0.};
  parthenon::Real tcov[4] = {1., 0., 0., 0.};

  ncov[dir] = 1.0;
  ncon[dir] = (dir == 0) ? -1.0 : 1.0;

  const auto n_sq = dot4(ncon, ncov);
  const auto t_sq = dot4(tcon, tcov);
  const auto n_dot_u = dot4(ncov, state.ucon);
  const auto t_dot_u = dot4(tcov, state.ucon);
  const auto n_dot_t = dot4(ncon, tcov);

  const auto n_dot_u_sq = n_dot_u * n_dot_u;
  const auto t_dot_u_sq = t_dot_u * t_dot_u;
  const auto nu_tu = n_dot_u * t_dot_u;

  const auto a = t_dot_u_sq - (t_sq + t_dot_u_sq) * cf_sq;
  const auto b = 2. * (nu_tu - (n_dot_t + nu_tu) * cf_sq);
  const auto c = n_dot_u_sq - (n_sq + n_dot_u_sq) * cf_sq;

  auto disc = b * b - 4. * a * c;

  if ((disc < 0.0) && (disc > -1.e-10)) {
    disc = 0.0;
  } else if (disc < -1.e-10) {
    disc = 0.;
  }

  disc = Kokkos::sqrt(disc);
  const auto v_plus = -(-b + disc) / (2. * a);
  const auto v_minus = -(-b - disc) / (2. * a);

  if (v_plus > v_minus) {
    v_max = v_plus;
    v_min = v_minus;
  } else {
    v_max = v_minus;
    v_min = v_plus;
  }
}

KOKKOS_INLINE_FUNCTION
void CalculateAlfvenVelocityGRMHD(
    const parthenon::Real gamma,
    const parthenon::Real prim[NPRIM],
    const parthenon::Real gcov[4][4], const parthenon::Real gcon[4][4],
    const int dir, parthenon::Real &v_max, parthenon::Real &v_min) {
  parthenon::Real ncon[4] = {0., 0., 0., 0.};
  parthenon::Real ncov[4] = {0., 0., 0., 0.};
  parthenon::Real tcon[4] = {0., 0., 0., 0.};
  parthenon::Real tcov[4] = {0., 0., 0., 0.};

  ncov[dir] = 1.0;
  tcov[0] = 1.0;

  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      ncon[row] += gcon[row][col] * ncov[col];
      tcon[row] += gcon[row][col] * tcov[col];
    }
  }

  State state;
  CalculateGRMHDState(prim, gcov, gcon, state);

  const auto b_sq = dot4(state.bcon, state.bcov);
  const auto h = prim[RHO] + gamma * prim[ENY];
  const auto h_tot = b_sq + h;
  const auto va_sq = b_sq / h_tot;
  const auto cs_sq = gamma * (gamma - 1.) * prim[ENY] / h;
  auto cf_sq = cs_sq + va_sq - cs_sq * va_sq;

  if (cf_sq < 0.) {
    cf_sq = 1e-10;
  }
  if (cf_sq > 1.) {
    cf_sq = 1.;
  }

  const auto n_sq = dot4(ncon, ncov);
  const auto t_sq = dot4(tcon, tcov);
  const auto n_dot_u = dot4(ncov, state.ucon);
  const auto t_dot_u = dot4(tcov, state.ucon);
  const auto n_dot_t = dot4(ncon, tcov);

  const auto n_dot_u_sq = n_dot_u * n_dot_u;
  const auto t_dot_u_sq = t_dot_u * t_dot_u;
  const auto nu_tu = n_dot_u * t_dot_u;

  const auto a = t_dot_u_sq - (t_sq + t_dot_u_sq) * cf_sq;
  const auto b = 2. * (nu_tu - (n_dot_t + nu_tu) * cf_sq);
  const auto c = n_dot_u_sq - (n_sq + n_dot_u_sq) * cf_sq;

  auto disc = b * b - 4. * a * c;
  if ((disc < 0.0) && (disc > -1.e-10)) {
    disc = 0.0;
  } else if (disc < -1.e-10) {
    disc = 0.;
  }

  disc = Kokkos::sqrt(disc);
  const auto v_plus = -(-b + disc) / (2. * a);
  const auto v_minus = -(-b - disc) / (2. * a);

  if (v_plus > v_minus) {
    v_max = v_plus;
    v_min = v_minus;
  } else {
    v_max = v_minus;
    v_min = v_plus;
  }
}

#endif  // PANGU_SRC_PHYSICSTRANSFORMATION_ALFVENVELOCITY_H
