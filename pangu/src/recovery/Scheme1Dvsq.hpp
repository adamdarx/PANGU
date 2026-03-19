#pragma once

#include <cmath>

#include "Restorer.hpp"

namespace Scheme1Dvsq {
KOKKOS_INLINE_FUNCTION
Real pressure_rho0_u(Real rho0, Real uu, Real gamma) {
  return ((gamma - 1.) * uu); ///< 理想气体状态方程: p = (γ-1)u
}

KOKKOS_INLINE_FUNCTION
Real pressure_rho0_w(Real rho0, Real w, Real gamma) {
  return ((gamma - 1.) * (w - rho0) / gamma); ///< 从焓推导压强
}

KOKKOS_INLINE_FUNCTION
void ncov_calc(const Real gcon[4][4], Real ncov[4]) {
  Real lapse = Kokkos::sqrt(-1.0 / gcon[0][0]); ///< Lapse函数
  for (int i = 0; i < 4; i++)
    ncov[i] = -lapse * (i == 0); ///< n_μ = (-α, 0, 0, 0)
  return;
}

KOKKOS_INLINE_FUNCTION
void raise_g(const Real vcov[4], const Real gcon[4][4], Real vcon[4]) {
  for (int i = 0; i < 4; i++) {
    vcon[i] = 0;
    for (int j = 0; j < 4; j++) {
      vcon[i] += gcon[i][j] * vcov[j]; ///< v^μ = g^{μν} v_ν
    }
  }
  return;
}

KOKKOS_INLINE_FUNCTION
void lower_g(const Real vcon[4], const Real gcov[4][4], Real vcov[4]) {
  for (int i = 0; i < 4; i++) {
    vcov[i] = 0;
    for (int j = 0; j < 4; j++) {
      vcov[i] += gcov[i][j] * vcon[j]; ///< v_μ = g_{μν} v^ν
    }
  }
  return;
}

KOKKOS_FUNCTION
void validate_x(Real x[1], Real x0[1]) {
  Real small = 1.e-10;

  x[0] = (x[0] >= 1.0) ? (0.5 * (x0[0] + 1.)) : x[0];
  x[0] = (x[0] < -small) ? (0.5 * x0[0]) : x[0];
  x[0] = Kokkos::fabs(x[0]);

  return;
}

KOKKOS_FUNCTION
Real pressure_of_rho(Real &K_atm, Real rho0, Real gamma) {
  return (K_atm * Kokkos::pow(rho0, gamma));
}

KOKKOS_FUNCTION
Real u_of_p(Real p, Real gamma) { return (p / (gamma - 1.)); }

KOKKOS_FUNCTION
Real W_of_vsq(Real &D, Real &K_atm, Real vsq, Real *p, Real *rho, Real *uu,
              Real gamma) {
  Real gtmp;

  gtmp = (1. - vsq);
  *rho = D * Kokkos::sqrt(gtmp);
  *p = pressure_of_rho(K_atm, *rho, gamma);
  *uu = u_of_p(*p, gamma);

  return ((*rho + *uu + *p) / gtmp);
}

KOKKOS_FUNCTION
Real dWdvsq_calc(Real vsq, Real rho, Real p, Real gamma) {
  return ((gamma * (2. - gamma) * p + (gamma - 1.) * rho) /
          (2. * (gamma - 1.) * (1. - vsq) * (1. - vsq)));
}

KOKKOS_FUNCTION
void func_1d_gnr(Real &Bsq, Real &QdotBsq, Real &Qtsq, Real &Qdotn, Real &D,
                 Real &K_atm, Real x[], Real dx[], Real resid[], Real jac[][1],
                 Real *f, Real *df, int n, Real gamma) {
  Real vsq, W, W0, Wsq, W3, dWdvsq, dpdrho, fact_tmp, rho, p, uu;
  int retval, iters;

  vsq = x[0];

  // Calculate best value for W given current guess for vsq:
  W = W_of_vsq(D, K_atm, vsq, &p, &rho, &uu, gamma);
  Wsq = W * W;
  W3 = W * Wsq;

  // Doing this assuming  P = (G-1) u :

  dWdvsq = dWdvsq_calc(vsq, rho, p, gamma);

  fact_tmp = (Bsq + W);

  resid[0] = Qtsq - vsq * fact_tmp * fact_tmp + QdotBsq * (Bsq + 2. * W) / Wsq;
  jac[0][0] = -fact_tmp * (fact_tmp + 2. * dWdvsq * (vsq + QdotBsq / W3));

  dx[0] = -resid[0] / jac[0][0];

  *f = 0.5 * resid[0] * resid[0];
  *df = -2. * (*f);
}

KOKKOS_FUNCTION
int general_newton_raphson(Real &Bsq, Real &QdotBsq, Real &Qtsq, Real &Qdotn,
                           Real &D, Real &K_atm, Real x[], int n,
                           void (*funcd)(Real &, Real &, Real &, Real &, Real &,
                                         Real &, Real[], Real[], Real[],
                                         Real[][1], Real *, Real *, int, Real),
                           Real gamma) {
  Real f, df, dx[1], x_old[1], resid[1], jac[1][1];
  Real errx, x_orig[1];
  int n_iter, id, jd, i_extra, doing_extra;
  Real dW, dvsq, vsq_old, vsq, W, W_old, rho, p, uu;

  int keep_iterating;

  // Initialize various parameters and variables:
  errx = 1.;
  df = f = 1.;
  i_extra = doing_extra = 0;

  for (id = 0; id < n; id++)
    x_old[id] = x_orig[id] = x[id];

  vsq_old = vsq = W = W_old = 0.;

  n_iter = 0;

  /* Start the Newton-Raphson iterations : */
  keep_iterating = 1;
  while (keep_iterating) {

    (*funcd)(Bsq, QdotBsq, Qtsq, Qdotn, D, K_atm, x, dx, resid, jac, &f, &df, n,
             gamma); /* returns with new dx, f, df */

    /* Save old values before calculating the new: */
    errx = 0.;
    for (id = 0; id < n; id++) {
      x_old[id] = x[id];
    }

    for (id = 0; id < n; id++) {
      x[id] += dx[id];
    }

    /****************************************/
    /* Make sure that the new x[] is physical : */
    /****************************************/
    // METHOD specific
    validate_x(x, x_old);

    /****************************************/
    /* Calculate the convergence criterion */
    /****************************************/

    /* For the new criterion, always look at error in "W" : */
    // METHOD specific
    W_old = W;
    W = W_of_vsq(D, K_atm, x[0], &p, &rho, &uu, gamma);
    errx = (W == 0.) ? Kokkos::abs(W - W_old) : Kokkos::abs((W - W_old) / W);
    errx += (x[0] == 0.) ? Kokkos::abs(x[0] - x_old[0])
                         : Kokkos::abs((x[0] - x_old[0]) / x[0]);

    /*****************************************************************************/
    /* If we've reached the tolerance level, then just do a few extra iterations
     */
    /*   before stopping */
    /*****************************************************************************/

    if ((Kokkos::abs(errx) <= NEWT_TOL) && (doing_extra == 0) &&
        (EXTRA_NEWT_ITER > 0)) {
      doing_extra = 1;
    }

    if (doing_extra == 1)
      i_extra++;

    // See if we've done the extra iterations, or have done too many iterations:
    if (((Kokkos::abs(errx) <= NEWT_TOL) && (doing_extra == 0)) ||
        (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
      keep_iterating = 0;
    }

    n_iter++;

  } // END of while(keep_iterating)

  /*  Check for bad untrapped divergences : */
  if ((std::isfinite(f) == 0) || (std::isfinite(df) == 0)) {
    return (2);
  }

  // Return in different ways depending on whether a solution was found:
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
int restore(Real U[8], Real prim[8], Real gamma) {
  Real gcon[4][4] = {{-1., 0., 0., 0.},
                   {0., 1., 0., 0.},
                   {0., 0., 1., 0.},
                   {0., 0., 0., 1.}};
  Real gcov[4][4] = {{-1., 0., 0., 0.},
                   {0., 1., 0., 0.},
                   {0., 0., 1., 0.},
                   {0., 0., 0., 1.}};
  Real gdet = -1.;
  Real x_1d[1];
  Real QdotB, Bcon[4], Bcov[4], Qcov[4], Qcon[4], ncov[4], ncon[4], Qsq,
      Qtcon[4];
  Real rho0, uu, p, w, gammasq, Gamma, gtmp, W_last, W, utsq, vsq;
  int i, j, retval;
  Real K_atm = (gamma - 1.) * prim[UU] * Kokkos::pow(prim[RHO], -gamma);
  // Assume ok initially:
  retval = 0;

  prim[B1] = U[B1];
  prim[B2] = U[B2];
  prim[B3] = U[B3];

  // Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
  Bcon[0] = 0.;
  Bcon[1] = U[B1];
  Bcon[2] = U[B2];
  Bcon[3] = U[B3];
  lower_g(Bcon, gcov, Bcov);

  Qcov[0] = U[UU];
  Qcov[1] = U[U1];
  Qcov[2] = U[U2];
  Qcov[3] = U[U3];
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

  /* calculate W from last timestep and use  for guess */
  utsq = 0.;
  for (i = 1; i < 4; i++)
    for (j = 1; j < 4; j++)
      utsq += gcov[i][j] * prim[U1 + i - 1] * prim[U1 + j - 1];

  if ((utsq < 0.) && (Kokkos::abs(utsq) < 1.0e-13)) {
    utsq = Kokkos::abs(utsq);
  }
  if (utsq < 0. || utsq > UTSQ_TOO_BIG) {
    retval = 2;
    return (retval);
  }

  gammasq = 1. + utsq;
  Gamma = Kokkos::sqrt(gammasq);

  // Always calculate rho from D and Gamma so that using D in EOS remains
  // consistent
  //   i.e. you don't get positive values for dP/d(vsq) .
  rho0 = D / Gamma;
  uu = prim[UU];
  p = pressure_rho0_u(rho0, uu, gamma);
  w = rho0 + uu + p;

  W_last = w * gammasq;

  // Initialize independent variables for Newton-Raphson:
  x_1d[0] = 1. - 1. / gammasq;

  // Find vsq via Newton-Raphson:
  retval = general_newton_raphson(Bsq, QdotBsq, Qtsq, Qdotn, D, K_atm, x_1d, 1,
                                  func_1d_gnr, gamma);

  /* Problem with solver, so return denoting error before doing anything
   * further */
  if (retval != 0) {
    retval = retval * 100 + 1;
    return (retval);
  }

  // Calculate v^2 :
  vsq = x_1d[0];
  if ((vsq >= 1.) || (vsq < 0.)) {
    retval = 4;
    return (retval);
  }

  // Find W from this vsq:
  W = W_of_vsq(D, K_atm, vsq, &p, &rho0, &uu, gamma);

  // Recover the primitive variables from the scalars and conserved variables:
  gtmp = Kokkos::sqrt(1. - vsq);
  Gamma = 1. / gtmp;

  w = W * (1. - vsq);

  // User may want to handle this case differently, e.g. do NOT return upon
  // a negative rho/u, calculate v^i so that rho/u can be floored by other
  // routine:
  if ((rho0 <= 0.) || (uu <= 0.)) {
    retval = 5;
    return (retval);
  }

  prim[RHO] = rho0;
  prim[UU] = uu;
  for (i = 1; i < 4; i++)
    Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
  for (i = 1; i < 4; i++)
    prim[U1 + i - 1] = Gamma / (W + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / W);
  /* set field components */
  prim[B1] = U[B1];
  prim[B2] = U[B2];
  prim[B3] = U[B3];

  /* done! */
  return (retval);
}
} // namespace Scheme1Dvsq
