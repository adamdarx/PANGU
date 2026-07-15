// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.
//
// MC (Monotonized Central) reconstruction — 2nd-order TVD, van Leer 1977.
// Uses a 3-point stencil → nghost = 2 is sufficient.
//
// REFERENCE:
//   B. van Leer, "Towards the Ultimate Conservative Difference Scheme. IV.
//   A New Approach to Numerical Convection", JCP, 23, 276 (1977)

#ifndef PANGU_SRC_INTERPOLATION_INTERPOLATERMC_H
#define PANGU_SRC_INTERPOLATION_INTERPOLATERMC_H

#include <basic_types.hpp>

KOKKOS_INLINE_FUNCTION
double minmod3(const double a, const double b, const double c) {
  if (a > 0.0 && b > 0.0 && c > 0.0)
    return Kokkos::fmin(Kokkos::fmin(a, b), c);
  if (a < 0.0 && b < 0.0 && c < 0.0)
    return Kokkos::fmax(Kokkos::fmax(a, b), c);
  return 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn MC
//! \brief Monotonized Central reconstruction (same signature as PPM4 for drop-in swap).
//! Given 5 cell-centered values q_im2..q_ip2 for cell i, returns the reconstructed
//! values at the right interface (ql_ip1) and left interface (qr_i) of cell i.
//!
//! The slope of cell i is computed from the central triplet (q_im1, q_i, q_ip1):
//!   slope = minmod(2*(q_i - q_im1), (q_ip1 - q_im1)/2, 2*(q_ip1 - q_i))
//!
//! Call pattern (identical to PPM4):
//!   Left  state at face i: MC(q_{i-3}, q_{i-2}, q_{i-1}, q_i, q_{i+1}, ql_i,   qr_{i-1})
//!   Right state at face i: MC(q_{i-2}, q_{i-1}, q_i,     q_{i+1}, q_{i+2}, ql_{i+1}, qr_i)

KOKKOS_INLINE_FUNCTION
void MC(const double &q_im2, const double &q_im1,
        const double &q_i,   const double &q_ip1,
        const double &q_ip2, double &ql_ip1,
        double &qr_i) {
  const double slope = minmod3(2.0*(q_i - q_im1),
                             0.5*(q_ip1 - q_im1),
                             2.0*(q_ip1 - q_i));
  ql_ip1 = q_i + 0.5 * slope;
  qr_i   = q_i - 0.5 * slope;
}

#endif  // PANGU_SRC_INTERPOLATION_INTERPOLATERMC_H
