// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_RECOVERY_SCHEMEKASTAUN_H
#define PANGU_SRC_RECOVERY_SCHEMEKASTAUN_H

#include <cmath>

#include "recovery/constants.h"

namespace SchemeKastaun {

// Ideal-gas specialization of the one-dimensional mu root solve described in
// Kastaun et al., Phys. Rev. D 103, 023018 (2021), arXiv:2005.01821.
KOKKOS_INLINE_FUNCTION
Real square(const Real x) { return x * x; }

KOKKOS_INLINE_FUNCTION
void lower_spatial(const Real vcon[3], const Real gamma[6], Real vcov[3]) {
  vcov[0] = gamma[0] * vcon[0] + gamma[1] * vcon[1] + gamma[2] * vcon[2];
  vcov[1] = gamma[1] * vcon[0] + gamma[3] * vcon[1] + gamma[4] * vcon[2];
  vcov[2] = gamma[2] * vcon[0] + gamma[4] * vcon[1] + gamma[5] * vcon[2];
}

KOKKOS_INLINE_FUNCTION
void raise_spatial(const Real vcov[3], const Real gamma_inv[6], Real vcon[3]) {
  vcon[0] = gamma_inv[0] * vcov[0] + gamma_inv[1] * vcov[1] +
            gamma_inv[2] * vcov[2];
  vcon[1] = gamma_inv[1] * vcov[0] + gamma_inv[3] * vcov[1] +
            gamma_inv[4] * vcov[2];
  vcon[2] = gamma_inv[2] * vcov[0] + gamma_inv[4] * vcov[1] +
            gamma_inv[5] * vcov[2];
}

KOKKOS_INLINE_FUNCTION
Real contract(const Real vcon[3], const Real wcov[3]) {
  return vcon[0] * wcov[0] + vcon[1] * wcov[1] + vcon[2] * wcov[2];
}

KOKKOS_INLINE_FUNCTION
Real spatial_determinant(const Real gamma[6]) {
  return gamma[0] * gamma[3] * gamma[5] +
         2.0 * gamma[1] * gamma[2] * gamma[4] -
         gamma[0] * gamma[4] * gamma[4] -
         gamma[1] * gamma[1] * gamma[5] -
         gamma[2] * gamma[2] * gamma[3];
}

KOKKOS_INLINE_FUNCTION
int invert_spatial_metric(const Real gamma[6], Real gamma_inv[6]) {
  const Real det = spatial_determinant(gamma);
  if (!(det > 0.0) || !std::isfinite(det)) {
    return 1;
  }
  const Real inv_det = 1.0 / det;
  gamma_inv[0] = (gamma[3] * gamma[5] - gamma[4] * gamma[4]) * inv_det;
  gamma_inv[1] = (gamma[2] * gamma[4] - gamma[1] * gamma[5]) * inv_det;
  gamma_inv[2] = (gamma[1] * gamma[4] - gamma[2] * gamma[3]) * inv_det;
  gamma_inv[3] = (gamma[0] * gamma[5] - gamma[2] * gamma[2]) * inv_det;
  gamma_inv[4] = (gamma[1] * gamma[2] - gamma[0] * gamma[4]) * inv_det;
  gamma_inv[5] = (gamma[0] * gamma[3] - gamma[1] * gamma[1]) * inv_det;
  return 0;
}

KOKKOS_INLINE_FUNCTION
void ncov_calc(const Real gcon[4][4], Real ncov[4]) {
  const Real lapse = Kokkos::sqrt(-1.0 / gcon[0][0]);
  for (int index = 0; index < 4; ++index) {
    ncov[index] = -lapse * (index == 0);
  }
}

KOKKOS_INLINE_FUNCTION
void raise_g(const Real vcov[4], const Real gcon[4][4], Real vcon[4]) {
  for (int row = 0; row < 4; ++row) {
    vcon[row] = 0.0;
    for (int col = 0; col < 4; ++col) {
      vcon[row] += gcon[row][col] * vcov[col];
    }
  }
}

KOKKOS_INLINE_FUNCTION
Real root_function(const Real mu, const Real D, const Real q, const Real bsq,
                   const Real rsq, const Real rbsq, const Real gamma) {
  const Real x = 1.0 / (1.0 + mu * bsq);
  const Real xsq = x * x;
  const Real rbarsq = x * (rsq * x + mu * (x + 1.0) * rbsq);
  const Real musq = mu * mu;
  Real vsq = musq * rbarsq;
  vsq = Kokkos::min(Kokkos::max(vsq, 0.0), 1.0 - 1.0e-14);
  const Real inv_lorentz = Kokkos::sqrt(1.0 - vsq);

  const Real qbar =
      q - 0.5 * bsq - 0.5 * musq * xsq * (bsq * rsq - rbsq);
  const Real eoverD = qbar - mu * rbarsq + 1.0;
  const Real rho = D * inv_lorentz;
  Real internal_energy = D * eoverD - rho;
  internal_energy = Kokkos::max(internal_energy, 1.0e-300);
  const Real pressure = (gamma - 1.0) * internal_energy;
  const Real h = 1.0 + (internal_energy + pressure) / rho;

  const Real nu_a = h * inv_lorentz;
  const Real nu_b = eoverD + pressure / D;
  const Real nu = Kokkos::max(nu_a, nu_b);
  const Real mu_hat = 1.0 / (nu + mu * rbarsq);
  return mu - mu_hat;
}

KOKKOS_INLINE_FUNCTION
int false_position(Real &mu, const Real D, const Real q, const Real bsq,
                   const Real rsq, const Real rbsq, const Real gamma) {
  Real lower = 0.0;
  Real upper = 1.0;
  Real f_lower = root_function(lower, D, q, bsq, rsq, rbsq, gamma);
  Real f_upper = root_function(upper, D, q, bsq, rsq, rbsq, gamma);

  if (!std::isfinite(f_lower) || !std::isfinite(f_upper)) {
    return 1;
  }
  if (Kokkos::fabs(f_lower) <= NEWT_TOL) {
    mu = lower;
    return 0;
  }
  if (Kokkos::fabs(f_upper) <= NEWT_TOL) {
    mu = upper;
    return 0;
  }
  if (f_lower * f_upper > 0.0) {
    return 2;
  }

  int side = 0;
  mu = lower;
  for (int iter = 0; iter < MAX_NEWT_ITER; ++iter) {
    const Real old_mu = mu;
    mu = (f_upper * lower - f_lower * upper) / (f_upper - f_lower);
    const Real f_mu = root_function(mu, D, q, bsq, rsq, rbsq, gamma);
    if (!std::isfinite(mu) || !std::isfinite(f_mu)) {
      return 3;
    }
    if (Kokkos::fabs((mu - old_mu) / (mu + 1.0e-300)) <= NEWT_TOL) {
      return 0;
    }
    if (f_mu * f_lower >= 0.0) {
      if (side == 1) {
        const Real weight = 1.0 - f_mu / f_lower;
        f_upper = (weight > 0.0) ? f_upper * weight : 0.5 * f_upper;
      }
      lower = mu;
      f_lower = f_mu;
      side = 1;
    } else {
      if (side == -1) {
        const Real weight = 1.0 - f_mu / f_upper;
        f_lower = (weight > 0.0) ? f_lower * weight : 0.5 * f_lower;
      }
      upper = mu;
      f_upper = f_mu;
      side = -1;
    }
  }

  return 4;
}

KOKKOS_INLINE_FUNCTION
int invert(Real U[8], Real prim[8], Real gamma, Real gcov[4][4],
           Real gcon[4][4], Real gdet) {
  (void)gdet;

  Real spatial_metric[6] = {gcov[1][1], gcov[1][2], gcov[1][3],
                            gcov[2][2], gcov[2][3], gcov[3][3]};
  Real spatial_metric_inv[6];
  if (invert_spatial_metric(spatial_metric, spatial_metric_inv) != 0) {
    return 1;
  }

  const Real D = U[RHO];
  if (!(D > 0.0) || !std::isfinite(D)) {
    return 2;
  }

  Real ncov[4], Qcov[4], Qcon[4];
  ncov_calc(gcon, ncov);
  Qcov[0] = U[ENY];
  Qcov[1] = U[UX1];
  Qcov[2] = U[UX2];
  Qcov[3] = U[UX3];
  raise_g(Qcov, gcon, Qcon);

  const Real Qdotn = Qcon[0] * ncov[0];
  const Real tau = -Qdotn - D;
  if (!std::isfinite(tau)) {
    return 3;
  }

  Real S_d[3] = {U[UX1], U[UX2], U[UX3]};
  Real S_u[3];
  raise_spatial(S_d, spatial_metric_inv, S_u);

  Real B_u[3] = {U[BX1], U[BX2], U[BX3]};
  Real B_d[3];
  lower_spatial(B_u, spatial_metric, B_d);

  const Real Bsq = contract(B_u, B_d);
  const Real Ssq = contract(S_u, S_d);
  const Real SdotB = contract(B_u, S_d);
  if (!std::isfinite(Bsq) || !std::isfinite(Ssq) || !std::isfinite(SdotB)) {
    return 4;
  }

  const Real inv_D = 1.0 / D;
  const Real sqrt_D = Kokkos::sqrt(D);
  const Real bsq = Bsq * inv_D;
  const Real rsq = Ssq * inv_D * inv_D;
  const Real rbsq = square(SdotB / (D * sqrt_D));
  const Real q = tau * inv_D;

  Real mu;
  const int root_status = false_position(mu, D, q, bsq, rsq, rbsq, gamma);
  if (root_status != 0) {
    return 100 + root_status;
  }

  const Real x = 1.0 / (1.0 + mu * bsq);
  const Real rb = SdotB / (D * sqrt_D);
  const Real rbarsq = x * (rsq * x + mu * (x + 1.0) * rbsq);
  Real vsq = mu * mu * rbarsq;
  if (!(vsq >= 0.0) || vsq >= 1.0 || !std::isfinite(vsq)) {
    return 5;
  }

  const Real lorentz = 1.0 / Kokkos::sqrt(1.0 - vsq);
  const Real rho = D / lorentz;
  const Real qbar =
      q - 0.5 * bsq - 0.5 * mu * mu * x * x * (bsq * rsq - rbsq);
  const Real eoverD = qbar - mu * rbarsq + 1.0;
  const Real internal_energy = D * eoverD - rho;
  if (!(rho > 0.0) || !(internal_energy > 0.0) ||
      !std::isfinite(internal_energy)) {
    return 6;
  }

  Real r_u[3] = {S_u[0] * inv_D, S_u[1] * inv_D, S_u[2] * inv_D};
  Real b_u[3] = {B_u[0] / sqrt_D, B_u[1] / sqrt_D, B_u[2] / sqrt_D};
  const Real Wmu_x = lorentz * mu * x;
  prim[RHO] = rho;
  prim[ENY] = internal_energy;
  for (int index = 0; index < 3; ++index) {
    prim[UX1 + index] = Wmu_x * (r_u[index] + rb * mu * b_u[index]);
  }
  prim[BX1] = U[BX1];
  prim[BX2] = U[BX2];
  prim[BX3] = U[BX3];

  return 0;
}

}  // namespace SchemeKastaun

#endif  // PANGU_SRC_RECOVERY_SCHEMEKASTAUN_H
