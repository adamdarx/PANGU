// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_RECOVERY_SCHEME1D_H
#define PANGU_SRC_RECOVERY_SCHEME1D_H

#include "recovery/constants.h"

namespace Scheme1D {
KOKKOS_INLINE_FUNCTION
Real pressure_rho0_u(Real rho0, Real u, Real gamma) {
  return ((gamma - 1.) * u); 
}

KOKKOS_INLINE_FUNCTION
Real pressure_rho0_w(Real rho0, Real w, Real gamma) {
  return ((gamma - 1.) * (w - rho0) / gamma); 
}

KOKKOS_INLINE_FUNCTION
void ncov_calc(const Real gcon[4][4], Real ncov[4]) {
  Real lapse = sqrt(-1.0 / gcon[0][0]); 
  for (int i = 0; i < 4; i++)
    ncov[i] = -lapse * (i == 0); 
  return;
}

KOKKOS_INLINE_FUNCTION
void raise_g(const Real vcov[4], const Real gcon[4][4], Real vcon[4]) {
  for (int i = 0; i < 4; i++) {
    vcon[i] = 0;
    for (int j = 0; j < 4; j++) {
      vcon[i] += gcon[i][j] * vcov[j]; 
    }
  }
  return;
}

KOKKOS_INLINE_FUNCTION
void lower_g(const Real vcon[4], const Real gcov[4][4], Real vcov[4]) {
  for (int i = 0; i < 4; i++) {
    vcov[i] = 0;
    for (int j = 0; j < 4; j++) {
      vcov[i] += gcov[i][j] * vcon[j]; 
    }
  }
  return;
}

KOKKOS_INLINE_FUNCTION
Real vsq_calc(Real &Bsq, Real &Qsq, Real &Qtsq, Real &QdotBsq, Real W) {
  Real Wsq, Xsq;

  Wsq = W * W;
  Xsq = (Bsq + W) * (Bsq + W);

  return ((Wsq * Qtsq + QdotBsq * (Bsq + 2. * W)) / (Wsq * Xsq));
}

KOKKOS_INLINE_FUNCTION
Real dvsq_dW(Real &Bsq, Real &Qtsq, Real &QdotBsq, Real W) {
  Real W3, X3, X;

  X = Bsq + W;
  W3 = W * W * W;
  X3 = X * X * X;

  return (-2. * (Qtsq / X3 + QdotBsq * (3 * W * X + Bsq * Bsq) / (W3 * X3)));
}

KOKKOS_INLINE_FUNCTION
Real pressure_of_rho(Real &K_atm, Real rho0, Real gamma) {
  return (K_atm * Kokkos::pow(rho0, gamma));
}

KOKKOS_INLINE_FUNCTION
Real u_of_p(Real p, Real gamma) { return (p / (gamma - 1.)); }

KOKKOS_FUNCTION
void func_gnr2_rho(Real &D, Real &K_atm, Real &W_for_gnr2, Real x[], Real dx[],
                   Real resid[], Real jac[][1], Real *f, Real *df, int n,
                   Real gamma) {
  Real A, B, C, rho, W, B0;

  A = D * D;
  B0 = A * gamma * K_atm;
  B = B0 / (gamma - 1.);
  rho = x[0];
  W = W_for_gnr2;
  C = Kokkos::pow(rho, gamma - 1.);

  resid[0] = rho * W - A - B * C;

  jac[0][0] = W - B0 * C / rho;

  dx[0] = -resid[0] / jac[0][0];
  *f = 0.5 * resid[0] * resid[0];
  *df = -2. * (*f);

  return;
}

KOKKOS_FUNCTION
int gnr2(Real &W_for_gnr2_old, Real &W_for_gnr2, Real &rho_for_gnr2_old,
         Real &rho_for_gnr2, Real &QdotBsq, Real &K_atm, Real &D, Real &Bsq,
         Real &Qtsq, Real x[], int n, Real gamma,
         void (*funcd)(Real &, Real &, Real &, Real[], Real[], Real[],
                       Real[][1], Real *, Real *, int, Real)) {
  Real f, df, dx[1], x_old[1], resid[1], jac[1][1];
  Real errx, x_orig[1];
  int n_iter, id, jd, i_extra, doing_extra;
  Real dW, dvsq, vsq_old, vsq, W, W_old;

  int keep_iterating;

  
  errx = 1.;
  df = f = 1.;
  i_extra = doing_extra = 0;
  for (id = 0; id < n; id++)
    x_old[id] = x_orig[id] = x[id];

  n_iter = 0;

  
  keep_iterating = 1;
  while (keep_iterating) {
    (*funcd)(D, K_atm, W_for_gnr2, x, dx, resid, jac, &f, &df, n,
             gamma); 

    
    errx = 0.;
    for (id = 0; id < n; id++) {
      x_old[id] = x[id];
    }

    
    for (id = 0; id < n; id++) {
      x[id] += dx[id];
    }

    
    for (id = 0; id < n; id++) {
      errx += (x[id] == 0.) ? Kokkos::abs(dx[id]) : Kokkos::abs(dx[id] / x[id]);
    }
    errx /= 1. * n;

    
    
    x[0] = Kokkos::abs(x[0]);

    
    
    
    if ((Kokkos::abs(errx) <= NEWT_TOL2) && (doing_extra == 0) &&
        (EXTRA_NEWT_ITER > 0)) {
      doing_extra = 0;
    }

    if (doing_extra == 1)
      i_extra++;

    
    if (((Kokkos::abs(errx) <= NEWT_TOL2) && (doing_extra == 0)) ||
        (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
      keep_iterating = 0;
    }

    n_iter++;
  }

  
  if ((std::isfinite(f) == 0) || (std::isfinite(df) == 0) ||
      (std::isfinite(x[0]) == 0)) {
    return (2);
  }

  
  if (Kokkos::abs(errx) > MIN_NEWT_TOL) {
    return (1);
  }
  if ((Kokkos::abs(errx) <= MIN_NEWT_TOL) && (Kokkos::abs(errx) > NEWT_TOL)) {
    
    
    return (0);
  }
  if (Kokkos::abs(errx) <= NEWT_TOL) {
    
    
    return (0);
  }

  return (0);
}

KOKKOS_FUNCTION
int general_newton_raphson(Real &W_for_gnr2_old, Real &W_for_gnr2,
                           Real &rho_for_gnr2_old, Real &rho_for_gnr2,
                           Real &QdotBsq, Real &K_atm, Real &D, Real &Bsq,
                           Real &Qtsq, Real x[], int n, Real gamma,
                           void (*funcd)(Real &, Real &, Real &, Real &, Real &,
                                         Real &, Real &, Real &, Real &, Real[],
                                         Real[], Real[], Real[][1], Real *,
                                         Real *, int, Real)) {
  Real f, df, dx[1], x_old[1], resid[1], jac[1][1];
  Real errx, x_orig[1];
  int n_iter, id, jd, i_extra, doing_extra;
  Real dW, dvsq, vsq_old, vsq, W, W_old;

  int keep_iterating, i_increase;

  
  errx = 1.;
  df = f = 1.;
  i_extra = doing_extra = 0;
  for (id = 0; id < n; id++)
    x_old[id] = x_orig[id] = x[id];

  vsq_old = vsq = W = W_old = 0.;

  n_iter = 0;

  
  keep_iterating = 1;
  while (keep_iterating) {
    (*funcd)(W_for_gnr2_old, W_for_gnr2, rho_for_gnr2_old, rho_for_gnr2,
             QdotBsq, K_atm, D, Bsq, Qtsq, x, dx, resid, jac, &f, &df, n,
             gamma); 

    
    errx = 0.;
    for (id = 0; id < n; id++) {
      x_old[id] = x[id];
    }

    
    for (id = 0; id < n; id++) {
      x[id] += dx[id];
    }

    
    i_increase = 0;
    while (
        ((x[0] * x[0] * x[0] * (x[0] + 2. * Bsq) -
          QdotBsq * (2. * x[0] + Bsq)) <= x[0] * x[0] * (Qtsq - Bsq * Bsq)) &&
        (i_increase < 10)) {
      x[0] -= (1. * i_increase) * dx[0] / 10.;
      i_increase++;
    }

    
    
    

    
    
    errx = (x[0] == 0.) ? fabs(dx[0]) : fabs(dx[0] / x[0]);

    
    
    
    x[0] = fabs(x[0]);

    
    
    
    
    

    if ((fabs(errx) <= NEWT_TOL) && (doing_extra == 0) &&
        (EXTRA_NEWT_ITER > 0)) {
      doing_extra = 1;
    }

    if (doing_extra == 1)
      i_extra++;

    if (((fabs(errx) <= NEWT_TOL) && (doing_extra == 0)) ||
        (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
      keep_iterating = 0;
    }

    n_iter++;

  } 

  
  if ((std::isfinite(f) == 0) || (std::isfinite(df) == 0) ||
      (std::isfinite(x[0]) == 0)) {
    return (2);
  }

  if (fabs(errx) > MIN_NEWT_TOL) {
    return (1);
  }
  if ((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > NEWT_TOL)) {
    return (0);
  }
  if (fabs(errx) <= NEWT_TOL) {
    return (0);
  }

  return (0);
}

void func_1d_orig1(Real &W_for_gnr2_old, Real &W_for_gnr2,
                   Real &rho_for_gnr2_old, Real &rho_for_gnr2, Real &QdotBsq,
                   Real &K_atm, Real &D, Real &Bsq, Real &Qtsq, Real x[],
                   Real dx[], Real resid[], Real jac[][1], Real *f, Real *df,
                   int n, Real gamma) {
  int retval, ntries;
  Real Dc, t1, t10, t2, t21, t23, t26, t29, t3, t30;
  Real t32, t33, t34, t38, t5, t51, t67, t8, W, x_rho[1], rho, rho_g;

  W = x[0];
  W_for_gnr2_old = W_for_gnr2;
  W_for_gnr2 = W;

  
  rho_g = x_rho[0] = rho_for_gnr2;

  ntries = 0;
  while ((retval = gnr2(W_for_gnr2_old, W_for_gnr2, rho_for_gnr2_old,
                        rho_for_gnr2, QdotBsq, K_atm, D, Bsq, Qtsq, x_rho, 1,
                        gamma, func_gnr2_rho)) &&
         (ntries++ < 10)) {
    rho_g *= 10.;
    x_rho[0] = rho_g;
  }

  rho_for_gnr2_old = rho_for_gnr2;
  rho = rho_for_gnr2 = x_rho[0];

  Dc = D;
  t1 = Dc * Dc;
  t2 = QdotBsq * t1;
  t3 = t2 * Bsq;
  t5 = Bsq * Bsq;
  t8 = t1 * Bsq;
  t10 = t1 * W;
  t21 = W * W;
  t23 = rho * rho;
  t26 = 1 / t1;
  resid[0] =
      (t3 + (2.0 * t2 + ((Qtsq - t5) * t1 + (-2.0 * t8 - t10) * W) * W) * W +
       (t5 + (2.0 * Bsq + W) * W) * t21 * t23) *
      t26 / t21;
  t29 = t1 * t1;
  t30 = QdotBsq * t29;
  t32 = gamma * K_atm;
  t33 = pow(rho, 1.0 * gamma);
  t34 = t32 * t33;
  t38 = t23 * t33;
  t51 = gamma * t1 * K_atm * t33;
  t67 = t21 * W;
  jac[0][0] =
      -2.0 *
      (t30 * Bsq * t34 +
       (t30 * t34 + ((-t38 * Bsq * t32 + Bsq * gamma * t1 * K_atm * t33) * t1 +
                     (-t38 * gamma * K_atm + t51) * t1 * W) *
                        t21) *
           W +
       ((-t3 + (-t2 + (-t8 - t10) * t21) * W) * W +
        (-t5 - Bsq * W) * t67 * t23) *
           t23) *
      t26 / (t51 - W * t23) / t67;

  dx[0] = -resid[0] / jac[0][0];

  *f = 0.5 * resid[0] * resid[0];
  *df = -2. * (*f);

  return;
}

void func_1d_orig2(Real &W_for_gnr2_old, Real &W_for_gnr2,
                   Real &rho_for_gnr2_old, Real &rho_for_gnr2, Real &QdotBsq,
                   Real &K_atm, Real &D, Real &Bsq, Real &Qtsq, Real x[],
                   Real dx[], Real resid[], Real jac[][1], Real *f, Real *df,
                   int n, Real gamma) {
  Real Dc, t1, t10, t2, t21, t23, t26, t3, t5, t8, W, rho;

  W = x[0];
  Dc = D;
  t1 = Dc * Dc;
  t2 = QdotBsq * t1;
  t3 = t2 * Bsq;
  t5 = Bsq * Bsq;
  t8 = t1 * Bsq;
  t10 = t1 * W;
  t21 = W * W;
  rho = t1 * (1. + gamma * K_atm / (gamma - 1.)) / W;
  t23 = rho * rho;
  t26 = 1 / t1;
  resid[0] =
      (t3 + (2.0 * t2 + ((Qtsq - t5) * t1 + (-2.0 * t8 - t10) * W) * W) * W +
       (t5 + (2.0 * Bsq + W) * W) * t21 * t23) *
      t26 / t21;
  jac[0][0] = -2.0 *
              (t3 + (t2 + (t8 + t10) * t21) * W + (t5 + Bsq * W) * t21 * t23) *
              t26 / t21 / W;

  dx[0] = -resid[0] / jac[0][0];

  *f = 0.5 * resid[0] * resid[0];
  *df = -2. * (*f);

  return;
}

KOKKOS_FUNCTION
int invert(Real U[9], Real prim[9], Real gamma, Real gcov[4][4],
           Real gcon[4][4], Real gdet) {
  Real x_1d[1];
  Real QdotB, Bcon[4], Bcov[4], Qcov[4], Qcon[4], ncov[4], ncon[4], Qsq,
      Qtcon[4];
  Real rho0, uu, p, w, uvsq, gtmp, W_last, W, utsq, vsq;
  int i, j, retval, i_increase;
  Real K_atm = prim[ENT];
  
  retval = 0;

  prim[BX1] = U[BX1];
  prim[BX2] = U[BX2];
  prim[BX3] = U[BX3];

  
  Bcon[0] = 0.;
  Bcon[1] = U[BX1];
  Bcon[2] = U[BX2];
  Bcon[3] = U[BX3];
  lower_g(Bcon, gcov, Bcov);

  Qcov[0] = U[ENY];
  Qcov[1] = U[UX1];
  Qcov[2] = U[UX2];
  Qcov[3] = U[UX3];
  raise_g(Qcov, gcon, Qcon);

  Real Bsq = 0.;
  for (i = 1; i < 4; i++)
    Bsq += Bcon[i] * Bcov[i];

  QdotB = 0.;
  for (i = 0; i < 4; i++)
    QdotB += Qcov[i] * Bcon[i];
  Real QdotBsq = QdotB * QdotB;

  ncov_calc(gcon, ncov);
  raise_g(ncov, gcon, ncon);

  Real Qdotn = Qcon[0] * ncov[0];

  Qsq = 0.;
  for (i = 0; i < 4; i++)
    Qsq += Qcov[i] * Qcon[i];

  Real Qtsq = Qsq + Qdotn * Qdotn;

  Real D = U[RHO];

  
  utsq = 0.;
  for (i = 1; i < 4; i++)
    for (j = 1; j < 4; j++)
      utsq += gcov[i][j] * prim[UX1 + i - 1] * prim[UX1 + j - 1];

  if ((utsq < 0.) && (Kokkos::abs(utsq) < 1.0e-13)) {
    utsq = Kokkos::abs(utsq);
  }
  if (utsq < 0. || utsq > UTSQ_TOO_BIG) {
    retval = 2;
    return (retval);
  }

  uvsq = 1. + utsq;
  Real Gamma = Kokkos::sqrt(uvsq);

  
  
  
  rho0 = D / Gamma;
  p = pressure_of_rho(K_atm, rho0, gamma);
  uu = u_of_p(p, gamma);
  w = rho0 + uu + p;

  W_last = w * uvsq;

  
  i_increase = 0;
  while (((W_last * W_last * W_last * (W_last + 2. * Bsq) -
           QdotBsq * (2. * W_last + Bsq)) <=
          W_last * W_last * (Qtsq - Bsq * Bsq)) &&
         (i_increase < 10)) {
    W_last *= 10.;
    i_increase++;
  }

  Real W_for_gnr2 = W_last;
  Real W_for_gnr2_old = W_last;
  Real rho_for_gnr2 = rho0;
  Real rho_for_gnr2_old = rho0;

  
  x_1d[0] = W_last;

#if (USE_ISENTROPIC)
  retval = general_newton_raphson(W_for_gnr2_old, W_for_gnr2, rho_for_gnr2_old,
                                  rho_for_gnr2, QdotBsq, K_atm, D, Bsq, Qtsq,
                                  x_1d, 1, gamma, func_1d_orig1);
#else
  retval = general_newton_raphson(W_for_gnr2_old, W_for_gnr2, rho_for_gnr2_old,
                                  rho_for_gnr2, QdotBsq, K_atm, D, Bsq, Qtsq,
                                  x_1d, 1, gamma, func_1d_orig2);
#endif

  W = x_1d[0];

  
  if ((retval != 0) || (W == FAIL_VAL)) {
    retval = retval * 100 + 1;
    return (retval);
  } else {
    if (W <= 0. || W > W_TOO_BIG) {
      retval = 3;
      return (retval);
    }
  }

  
  vsq = vsq_calc(Bsq, Qsq, Qtsq, QdotBsq, W);
  if (vsq >= 1.) {
    retval = 4;
    return (retval);
  }

  
  gtmp = Kokkos::sqrt(1. - vsq);
  Gamma = 1. / gtmp;
  rho0 = D * gtmp;

  w = W * (1. - vsq);

  p = pressure_of_rho(K_atm, rho0, gamma);
  uu = u_of_p(p, gamma);

  
  
  
  if ((rho0 <= 0.) || (uu <= 0.)) {
    retval = 5;
    return (retval);
  }

  prim[RHO] = rho0;
  prim[ENY] = uu;
  prim[ENT] = (gamma - 1.) * uu * Kokkos::pow(rho0, -gamma);
  for (i = 1; i < 4; i++)
    Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
  for (i = 1; i < 4; i++)
    prim[UX1 + i - 1] = Gamma / (W + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / W);
  
  prim[BX1] = U[BX1];
  prim[BX2] = U[BX2];
  prim[BX3] = U[BX3];
  
  return (retval);
}
} 

#endif  // PANGU_SRC_RECOVERY_SCHEME1D_H
