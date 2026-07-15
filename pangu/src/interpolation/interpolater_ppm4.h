// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.
//
// PPM4 (Piecewise Parabolic Method, 4th-order) reconstruction.
// Based on Colella & Woodward 1984 (CW) limiters, as implemented in AthenaK.
//
// REFERENCE:
//   P. Colella & P. Woodward, "The Piecewise Parabolic Method (PPM) for
//   Gas-Dynamical Simulations", JCP, 54, 174 (1984)

#ifndef PANGU_SRC_INTERPOLATION_INTERPOLATERPPM4_H
#define PANGU_SRC_INTERPOLATION_INTERPOLATERPPM4_H

#include <basic_types.hpp>

//----------------------------------------------------------------------------------------
//! \fn PPM4
//! \brief PPM parabolic reconstruction with Colella-Woodward limiters.
//! Given 5 cell-centered values q_im2..q_ip2 for cell i, returns the
//! reconstructed values at the left interface of cell i+1 (ql_ip1) and
//! right interface of cell i (qr_i).
//!
//! For face f between cells i-1 and i:
//!   PPM4(q_{i-3}, q_{i-2}, q_{i-1}, q_i,     q_{i+1}, ql_i,   qr_{i-1})
//!   left state  = ql_i     (= qrv of cell i-1)
//!   PPM4(q_{i-2}, q_{i-1}, q_i,     q_{i+1}, q_{i+2}, ql_{i+1}, qr_i)
//!   right state = qr_i     (= qlv of cell i)

KOKKOS_INLINE_FUNCTION
void PPM4(const parthenon::Real &q_im2, const parthenon::Real &q_im1,
          const parthenon::Real &q_i,   const parthenon::Real &q_ip1,
          const parthenon::Real &q_ip2, parthenon::Real &ql_ip1,
          parthenon::Real &qr_i) {
  // 4th-order interface values (CW eqn 1.6 / CS eqn 16 / PH 3.26-3.27)
  // qlv = left  interface of cell i = q[i-1/2]
  // qrv = right interface of cell i = q[i+1/2]
  parthenon::Real qlv =
      (7.0 * (q_i + q_im1) - (q_im2 + q_ip1)) / 12.0;
  parthenon::Real qrv =
      (7.0 * (q_i + q_ip1) - (q_im1 + q_ip2)) / 12.0;

  // Limit qrv and qlv to neighboring cell-centered values (CS eqn 13)
  qlv = Kokkos::fmax(qlv, Kokkos::fmin(q_i, q_im1));
  qlv = Kokkos::fmin(qlv, Kokkos::fmax(q_i, q_im1));
  qrv = Kokkos::fmax(qrv, Kokkos::fmin(q_i, q_ip1));
  qrv = Kokkos::fmin(qrv, Kokkos::fmax(q_i, q_ip1));

  // Monotonize interpolated L/R states (CS eqns 14, 15 / CW eqn 1.10)
  parthenon::Real qc = qrv - q_i;
  parthenon::Real qd = qlv - q_i;
  if ((qc * qd) >= 0.0) {
    qlv = q_i;
    qrv = q_i;
  } else {
    if (Kokkos::fabs(qc) >= 2.0 * Kokkos::fabs(qd)) {
      qrv = q_i - 2.0 * qd;
    }
    if (Kokkos::fabs(qd) >= 2.0 * Kokkos::fabs(qc)) {
      qlv = q_i - 2.0 * qc;
    }
  }

  // Return: ql at cell i+1 (right interface of cell i → left of face i+1)
  //         qr at cell i   (left interface of cell i  → right of face i)
  ql_ip1 = qrv;
  qr_i   = qlv;
}

#endif  // PANGU_SRC_INTERPOLATION_INTERPOLATERPPM4_H
