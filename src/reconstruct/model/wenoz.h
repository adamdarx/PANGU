#ifndef PANGU_RECONSTRUCT_MODEL_WENOZ_H_
#define PANGU_RECONSTRUCT_MODEL_WENOZ_H_

#include <parthenon/parthenon.hpp>

namespace pangu::reconstruct::model {

struct WENOZ {
  // Reads three cells on each side of the face.
  static constexpr int stencil_radius = 3;
  static constexpr bool cell_edge_states = false;

  KOKKOS_INLINE_FUNCTION static parthenon::Real Square(const parthenon::Real value) {
    return value * value;
  }

  KOKKOS_INLINE_FUNCTION static void Cell(
      const parthenon::Real q_im2, const parthenon::Real q_im1,
      const parthenon::Real q_i, const parthenon::Real q_ip1,
      const parthenon::Real q_ip2, parthenon::Real& right_face,
      parthenon::Real& left_face) {
    constexpr parthenon::Real beta0 = 13.0 / 12.0;
    constexpr parthenon::Real beta1 = 0.25;
    constexpr parthenon::Real epsilon = 1.0e-42;
    parthenon::Real beta[3];
    beta[0] = beta0 * Square(q_im2 + q_i - 2.0 * q_im1) +
              beta1 * Square(q_im2 + 3.0 * q_i - 4.0 * q_im1);
    beta[1] = beta0 * Square(q_im1 + q_ip1 - 2.0 * q_i) +
              beta1 * Square(q_im1 - q_ip1);
    beta[2] = beta0 * Square(q_ip2 + q_i - 2.0 * q_ip1) +
              beta1 * Square(q_ip2 + 3.0 * q_i - 4.0 * q_ip1);
    const parthenon::Real tau5 = fabs(beta[0] - beta[2]);
    parthenon::Real indicator[3];
    indicator[0] = Square(tau5 / (beta[0] + epsilon));
    indicator[1] = Square(tau5 / (beta[1] + epsilon));
    indicator[2] = Square(tau5 / (beta[2] + epsilon));
    parthenon::Real candidate[3];
    candidate[0] = 2.0 * q_im2 - 7.0 * q_im1 + 11.0 * q_i;
    candidate[1] = -q_im1 + 5.0 * q_i + 2.0 * q_ip1;
    candidate[2] = 2.0 * q_i + 5.0 * q_ip1 - q_ip2;
    parthenon::Real alpha[3];
    alpha[0] = 0.1 * (1.0 + indicator[0]);
    alpha[1] = 0.6 * (1.0 + indicator[1]);
    alpha[2] = 0.3 * (1.0 + indicator[2]);
    parthenon::Real alpha_sum = 6.0 * (alpha[0] + alpha[1] + alpha[2]);
    right_face = (candidate[0] * alpha[0] + candidate[1] * alpha[1] +
                  candidate[2] * alpha[2]) /
                 alpha_sum;
    candidate[0] = 2.0 * q_ip2 - 7.0 * q_ip1 + 11.0 * q_i;
    candidate[1] = -q_ip1 + 5.0 * q_i + 2.0 * q_im1;
    candidate[2] = 2.0 * q_i + 5.0 * q_im1 - q_im2;
    alpha[0] = 0.1 * (1.0 + indicator[2]);
    alpha[2] = 0.3 * (1.0 + indicator[0]);
    alpha_sum = 6.0 * (alpha[0] + alpha[1] + alpha[2]);
    left_face = (candidate[0] * alpha[0] + candidate[1] * alpha[1] +
                 candidate[2] * alpha[2]) /
                alpha_sum;
  }

  KOKKOS_INLINE_FUNCTION static void Reconstruct(
      const parthenon::Real qmmm, const parthenon::Real qmm, const parthenon::Real qm,
      const parthenon::Real q0, const parthenon::Real qp, const parthenon::Real qpp,
      parthenon::Real& left, parthenon::Real& right) {
    parthenon::Real unused_left;
    Cell(qmmm, qmm, qm, q0, qp, left, unused_left);
    parthenon::Real unused_right;
    Cell(qmm, qm, q0, qp, qpp, unused_right, right);
  }
};

} // namespace pangu::reconstruct::model

#endif
