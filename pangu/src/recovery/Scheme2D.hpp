#pragma once

#include <cmath>

#include "IRecovery.hpp"

namespace Scheme2D {
KOKKOS_INLINE_FUNCTION
Real dpdvsq_calc(Real &D, Real W, Real vsq, Real gamma) {
    return ((gamma - 1.) * (0.5 * D / Kokkos::sqrt(1. - vsq) - W) / gamma);
}

KOKKOS_INLINE_FUNCTION
Real dpdW_calc_vsq(Real W, Real vsq, Real gamma) {
    return ((gamma - 1.) * (1. - vsq) / gamma);
}

KOKKOS_INLINE_FUNCTION
Real pressure_W_vsq(Real &D, Real W, Real vsq, Real gamma) {
    Real gtmp = 1. - vsq;
    return ((gamma - 1.) * (W * gtmp - D * Kokkos::sqrt(gtmp)) / gamma);
}

KOKKOS_INLINE_FUNCTION
Real vsq_calc(Real &Bsq, Real &QdotBsq, Real &Qtsq, Real W) {
    Real Wsq, Xsq;

    Wsq = W * W;
    Xsq = (Bsq + W) * (Bsq + W);

    return ((Wsq * Qtsq + QdotBsq * (Bsq + 2. * W)) / (Wsq * Xsq));
}

KOKKOS_INLINE_FUNCTION
Real x1_of_x0(Real &Bsq, Real &QdotBsq, Real &Qtsq, Real x0) {
    Real vsq;
    Real dv = 1.e-15;
    vsq = Kokkos::abs(
        vsq_calc(Bsq, QdotBsq, Qtsq, x0)); // guaranteed to be positive
    return ((vsq > 1.) ? (1.0 - dv) : vsq);
}

KOKKOS_INLINE_FUNCTION
void validate_x(Real x[2], Real x0[2]) {
    Real dv = 1.e-15;

    /* Always take the absolute value of x[0] and check to see if it's too big: */
    x[0] = Kokkos::abs(x[0]);
    x[0] = (x[0] > kWTooBig) ? x0[0] : x[0];

    x[1] = (x[1] < 0.) ? 0. : x[1];        /* if it's too small */
    x[1] = (x[1] > 1.) ? (1. - dv) : x[1]; /* if it's too big   */

    return;
}

KOKKOS_FUNCTION
void func_vsq(Real &Bsq, Real &QdotBsq, Real &Qtsq, Real &Qdotn, Real &D,
                Real x[], Real dx[], Real resid[], Real jac[][2], Real *f,
                Real *df, int n, Real gamma) {
    Real W, vsq, Wsq, p_tmp, dPdvsq, dPdW, temp, tmp2, tmp3;
    Real t11;
    Real t16;
    Real t18;
    Real t2;
    Real t21;
    Real t23;
    Real t24;
    Real t25;
    Real t3;
    Real t35;
    Real t36;
    Real t4;
    Real t40;
    Real t9;

    W = x[0];
    vsq = x[1];

    Wsq = W * W;

    p_tmp = pressure_W_vsq(D, W, vsq, gamma);
    dPdW = dpdW_calc_vsq(W, vsq, gamma);
    dPdvsq = dpdvsq_calc(D, W, vsq, gamma);
    // These expressions were calculated using Mathematica, but made into
    // efficient code using Maple.  Since we know the analytic form of the
    // equations, we can explicitly calculate the Newton-Raphson step:

    t2 = -0.5 * Bsq + dPdvsq;
    t3 = Bsq + W;
    t4 = t3 * t3;
    t9 = 1.0 / Wsq;
    t11 = Qtsq - vsq * t4 + QdotBsq * (Bsq + 2.0 * W) * t9;
    t16 = QdotBsq * t9;
    t18 = -Qdotn - 0.5 * Bsq * (1.0 + vsq) + 0.5 * t16 - W + p_tmp;
    t21 = 1.0 / t3;
    t23 = 1.0 / W;
    t24 = t16 * t23;
    t25 = -1.0 + dPdW - t24;
    t35 = t25 * t3 + (Bsq - 2.0 * dPdvsq) * (QdotBsq + vsq * Wsq * W) * t9 * t23;
    t36 = 1.0 / t35;
    dx[0] = -(t2 * t11 + t4 * t18) * t21 * t36;
    t40 = (vsq + t24) * t3;
    dx[1] = -(-t25 * t11 - 2.0 * t40 * t18) * t21 * t36;
    jac[0][0] = -2.0 * t40;
    jac[0][1] = -t4;
    jac[1][0] = t25;
    jac[1][1] = t2;
    resid[0] = t11;
    resid[1] = t18;
    *df = -resid[0] * resid[0] - resid[1] * resid[1];

    *f = -0.5 * (*df);
}

KOKKOS_FUNCTION
int general_newton_raphson(Real &Bsq, Real &QdotBsq, Real &Qtsq, Real &Qdotn,
                            Real &D, Real x[], int n,
                            void (*funcd)(Real &, Real &, Real &, Real &, Real &,
                                            Real[], Real[], Real[], Real[][2],
                                            Real *, Real *, int, Real),
                            Real gamma) {
    Real f, df, dx[2], x_old[2];
    Real resid[2], jac[2][2];
    Real errx, x_orig[2];
    int n_iter, id, jd, i_extra, doing_extra;
    Real dW, dvsq, vsq_old, vsq, W, W_old;

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
    (*funcd)(Bsq, QdotBsq, Qtsq, Qdotn, D, x, dx, resid, jac, &f, &df, n,
                gamma); /* returns with new dx, f, df */

    /* Save old values before calculating the new: */
    errx = 0.;
    for (id = 0; id < n; id++) {
        x_old[id] = x[id];
    }

    /* Make the newton step: */
    for (id = 0; id < n; id++) {
        x[id] += dx[id];
    }

    /****************************************/
    /* Calculate the convergence criterion */
    /****************************************/
    errx = (x[0] == 0.) ? Kokkos::abs(dx[0]) : Kokkos::abs(dx[0] / x[0]);
    // if 2D method, make sure both W and vsq converge
    // this is important in non-relativistic flows where
    // vsq could be << 1
    if (n > 1) {
        errx += (x[1] == 0.) ? Kokkos::abs(dx[1]) : Kokkos::abs(dx[1] / x[1]);
    }

    /****************************************/
    /* Make sure that the new x[] is physical : */
    /****************************************/
    validate_x(x, x_old);

    /*****************************************************************************/
    /* If we've reached the tolerance level, then just do a few extra iterations
        */
    /*  before stopping */
    /*****************************************************************************/

    if ((Kokkos::abs(errx) <= kNewtTol) && (doing_extra == 0) &&
        (kExtraNewtIter > 0))
        doing_extra = 1;

    if (doing_extra == 1)
        i_extra++;

    if (((Kokkos::abs(errx) <= kNewtTol) && (doing_extra == 0)) ||
        (i_extra > kExtraNewtIter) || (n_iter >= (kMaxNewtIter - 1))) {
        keep_iterating = 0;
    }

    n_iter++;
    } // END of while(keep_iterating)

    /*  Check for bad untrapped divergences : */
    if ((std::isfinite(f) == 0) || (std::isfinite(df) == 0)) {
    //   fprintf(stderr,"failure, untrapped divergences, %g %g %g %g \n", f, df,
    //   x, dx);
    return (2);
    }

    if (Kokkos::abs(errx) > kMinNewtTol) {
    return (1);
    //  fprintf(stderr,"failure, tolerance not reached \n");
    }
    if ((Kokkos::abs(errx) <= kMinNewtTol) && (Kokkos::abs(errx) > kNewtTol)) {
    return (0);
    }
    if (Kokkos::abs(errx) <= kNewtTol) {
    return (0);
    }

    return (0);
}

KOKKOS_FUNCTION
int invert(Real U[8], Real prim[8], Real gamma) {
    Real gcon[4][4] = {{-1., 0., 0., 0.},
                    {0., 1., 0., 0.},
                    {0., 0., 1., 0.},
                    {0., 0., 0., 1.}};
    Real gcov[4][4] = {{-1., 0., 0., 0.},
                    {0., 1., 0., 0.},
                    {0., 0., 1., 0.},
                    {0., 0., 0., 1.}};
    Real gdet = -1.;
    Real x_2d[2];
    Real QdotB, Bcon[4], Bcov[4], Qcov[4], Qcon[4], ncov[4], ncon[4], Qsq,
        Qtcon[4];
    Real rho0, uu, p, w, gammasq, Gamma, gtmp, W_last, W, utsq, vsq;

    int i, j, n, retval, i_increase;
    n = 2;

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
    IRecovery::lower_g(Bcon, gcov, Bcov);

    Qcov[0] = U[UU];
    Qcov[1] = U[U1];
    Qcov[2] = U[U2];
    Qcov[3] = U[U3];
    IRecovery::raise_g(Qcov, gcon, Qcon);

    Real Bsq = 0.;
    for (i = 1; i < 4; i++)
    Bsq += Bcon[i] * Bcov[i];

    QdotB = 0.;
    for (i = 0; i < 4; i++)
    QdotB += Qcov[i] * Bcon[i];
    Real QdotBsq = QdotB * QdotB;

    IRecovery::ncov_calc(gcon, ncov);
    IRecovery::raise_g(ncov, gcon, ncon);

    Real Qdotn = Qcon[0] * ncov[0];

    Qsq = 0.;
    for (i = 0; i < 4; i++)
    Qsq += Qcov[i] * Qcon[i];

    Real Qtsq = Qsq + Qdotn * Qdotn;

    Real D = U[RHO];

    /* calculate W from last timestep and use for guess */
    utsq = 0.;
    for (i = 1; i < 4; i++)
    for (j = 1; j < 4; j++)
        utsq += gcov[i][j] * prim[U1 + i - 1] * prim[U1 + j - 1];

    if ((utsq < 0.) && (Kokkos::abs(utsq) < 1.0e-13)) {
    utsq = Kokkos::abs(utsq);
    }
    if (utsq < 0. || utsq > kUtsqTooBig) {
    retval = 2;
    return (retval);
    //  fprintf(stderr,"failure, utsq_too_big at %d %d\n", i,j);
    }

    gammasq = 1. + utsq;
    Gamma = Kokkos::sqrt(gammasq);

    // Always calculate rho from D and gamma so that using D in EOS remains
    // consistent
    //   i.e. you don't get positive values for dP/d(vsq) .
    rho0 = D / Gamma;
    uu = prim[UU];
    p = pressure_rho0_u(rho0, uu, gamma);
    w = rho0 + uu + p;

    W_last = w * gammasq;

    // Make sure that W is large enough so that v^2 < 1 :
    i_increase = 0;
    while (((W_last * W_last * W_last * (W_last + 2. * Bsq) -
            QdotBsq * (2. * W_last + Bsq)) <=
            W_last * W_last * (Qtsq - Bsq * Bsq)) &&
            (i_increase < 10)) {
    W_last *= 10.;
    i_increase++;
    }

    // Calculate W and vsq:
    x_2d[0] = Kokkos::abs(W_last);
    x_2d[1] = x1_of_x0(Bsq, QdotBsq, Qtsq, W_last);
    retval = general_newton_raphson(Bsq, QdotBsq, Qtsq, Qdotn, D, x_2d, n,
                                    func_vsq, gamma);

    W = x_2d[0];
    vsq = x_2d[1];
    /* Problem with solver, so return denoting error before doing anything
    * further */ 
    if ((retval != 0) || (W == kFailVal)) {
    //  fprintf(stderr,"failure, in solver at %d %d\n", i,j);
    retval = retval * 100 + 1;
    return (retval);

    } else {
    if (W <= 0. || W > kWTooBig) {
        //  fprintf(stderr,"failure, W_too_big at %d %d\n", i,j);
        retval = 3;
        return (retval);
    }
    }

    // Calculate v^2:
    if (vsq >= 1.) {
    retval = 4;
    return (retval);
    }

    // Recover the primitive variables from the scalars and conserved variables:
    gtmp = Kokkos::sqrt(1. - vsq);
    Gamma = 1. / gtmp;
    rho0 = D * gtmp;

    w = W * (1. - vsq);
    p = pressure_rho0_w(rho0, w, gamma);
    uu = w - (rho0 + p);

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
    return (retval);
}

class Scheme2DRecovery final : public IRecovery {
    public:
    int invert(Real U[8], Real prim[8], Real gamma) const override {
    return Scheme2D::invert(U, prim, gamma);
    }
};

} // namespace Scheme2D
