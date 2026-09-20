#ifndef PANGU_RECONSTRUCT_MODEL_PPMC_H_
#define PANGU_RECONSTRUCT_MODEL_PPMC_H_

// Copyright (c) 2020-2021, Athena-Parthenon Collaboration.
// Copyright (c) 2020, James M. Stone and the Athena code team.
// Derived from AthenaPK src/recon/ppm_simple.hpp under BSD-3-Clause.

#include <parthenon/parthenon.hpp>

namespace pangu::reconstruct::model {

struct PPMC {
  // Reads three cells on each side of the face.
  static constexpr int stencil_radius = 3;
  static constexpr bool cell_edge_states = false;

  KOKKOS_INLINE_FUNCTION static parthenon::Real Sign(const parthenon::Real value) {
    return value < 0.0 ? -1.0 : 1.0;
  }

  KOKKOS_INLINE_FUNCTION static void Cell(
      const parthenon::Real q_im2, const parthenon::Real q_im1,
      const parthenon::Real q_i, const parthenon::Real q_ip1,
      const parthenon::Real q_ip2, parthenon::Real& right_edge,
      parthenon::Real& left_edge) {
    constexpr parthenon::Real c2 = 1.25;
    parthenon::Real qa = q_i - q_im1;
    parthenon::Real qb = q_ip1 - q_i;
    const parthenon::Real dd_im1 = 0.5 * qa + 0.5 * (q_im1 - q_im2);
    const parthenon::Real dd_i = 0.5 * qb + 0.5 * qa;
    const parthenon::Real dd_ip1 = 0.5 * (q_ip2 - q_ip1) + 0.5 * qb;
    parthenon::Real face_left = 0.5 * (q_im1 + q_i) + (dd_im1 - dd_i) / 6.0;
    parthenon::Real face_right = 0.5 * (q_i + q_ip1) + (dd_i - dd_ip1) / 6.0;
    const parthenon::Real d2_im1 = q_im2 + q_i - 2.0 * q_im1;
    const parthenon::Real d2_i = q_im1 + q_ip1 - 2.0 * q_i;
    const parthenon::Real d2_ip1 = q_i + q_ip2 - 2.0 * q_ip1;
    parthenon::Real qa_tmp = face_left - q_im1;
    parthenon::Real qb_tmp = q_i - face_left;
    qa = 3.0 * (q_im1 + q_i - 2.0 * face_left);
    qb = d2_im1;
    parthenon::Real qc = d2_i;
    parthenon::Real qd = 0.0;
    if (Sign(qa) == Sign(qb) && Sign(qa) == Sign(qc))
      qd = Sign(qa) * fmin(c2 * fabs(qb), fmin(c2 * fabs(qc), fabs(qa)));
    const parthenon::Real limited_left = 0.5 * (q_im1 + q_i) - qd / 6.0;
    if (qa_tmp * qb_tmp < 0.0)
      face_left = limited_left;
    qa_tmp = face_right - q_i;
    qb_tmp = q_ip1 - face_right;
    qa = 3.0 * (q_i + q_ip1 - 2.0 * face_right);
    qb = d2_i;
    qc = d2_ip1;
    qd = 0.0;
    if (Sign(qa) == Sign(qb) && Sign(qa) == Sign(qc))
      qd = Sign(qa) * fmin(c2 * fabs(qb), fmin(c2 * fabs(qc), fabs(qa)));
    const parthenon::Real limited_right = 0.5 * (q_i + q_ip1) - qd / 6.0;
    if (qa_tmp * qb_tmp < 0.0)
      face_right = limited_right;
    const parthenon::Real d2_face = 6.0 * (face_left + face_right - 2.0 * q_i);
    left_edge = face_left;
    right_edge = face_right;
    const parthenon::Real delta_minus = q_i - left_edge;
    const parthenon::Real delta_plus = right_edge - q_i;
    qa_tmp = delta_minus * delta_plus;
    qb_tmp = (q_ip1 - q_i) * (q_i - q_im1);
    qa = d2_im1;
    qb = d2_i;
    qc = d2_ip1;
    qd = d2_face;
    parthenon::Real qe = 0.0;
    if (Sign(qa) == Sign(qb) && Sign(qa) == Sign(qc) && Sign(qa) == Sign(qd))
      qe = Sign(qd) * fmin(fmin(c2 * fabs(qa), c2 * fabs(qb)),
                            fmin(c2 * fabs(qc), fabs(qd)));
    qa = fmax(fabs(q_im1), fabs(q_im2));
    qb = fmax(fmax(fabs(q_i), fabs(q_ip1)), fabs(q_ip2));
    parthenon::Real ratio = 0.0;
    if (fabs(qd) > 1.0e-12 * fmax(qa, qb))
      ratio = qe / qd;
    const parthenon::Real smooth_left = q_i - ratio * delta_minus;
    const parthenon::Real smooth_right = q_i + ratio * delta_plus;
    const parthenon::Real monotone_left = q_i - 2.0 * delta_plus;
    const parthenon::Real monotone_right = q_i + 2.0 * delta_minus;
    if (qa_tmp <= 0.0 || qb_tmp <= 0.0) {
      if (ratio <= 1.0 - 1.0e-12) {
        left_edge = smooth_left;
        right_edge = smooth_right;
      }
    } else {
      if (fabs(delta_minus) >= 2.0 * fabs(delta_plus))
        left_edge = monotone_left;
      if (fabs(delta_plus) >= 2.0 * fabs(delta_minus))
        right_edge = monotone_right;
    }
  }

  KOKKOS_INLINE_FUNCTION static void Reconstruct(
      const parthenon::Real qmmm, const parthenon::Real qmm, const parthenon::Real qm,
      const parthenon::Real q0, const parthenon::Real qp, const parthenon::Real qpp,
      parthenon::Real& left, parthenon::Real& right) {
    parthenon::Real unused_left_edge;
    Cell(qmmm, qmm, qm, q0, qp, left, unused_left_edge);
    parthenon::Real unused_right_edge;
    Cell(qmm, qm, q0, qp, qpp, unused_right_edge, right);
  }
};

} // namespace pangu::reconstruct::model

#endif
