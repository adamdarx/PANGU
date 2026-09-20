#ifndef PANGU_RECONSTRUCT_MODEL_PPM_H_
#define PANGU_RECONSTRUCT_MODEL_PPM_H_

#include <parthenon/parthenon.hpp>

namespace pangu::reconstruct::model {

struct PPM {
  // Reads three cells on each side of the face.
  static constexpr int stencil_radius = 3;
  static constexpr bool cell_edge_states = false;

  KOKKOS_INLINE_FUNCTION static parthenon::Real FourthOrderInterface(
      const parthenon::Real q0, const parthenon::Real q1, const parthenon::Real q2,
      const parthenon::Real q3) {
    const parthenon::Real candidate =
        (7.0 / 12.0) * (q1 + q2) - (1.0 / 12.0) * (q0 + q3);
    return fmin(fmax(candidate, fmin(q1, q2)), fmax(q1, q2));
  }

  KOKKOS_INLINE_FUNCTION static void LimitParabola(const parthenon::Real center,
                                                    parthenon::Real& left_edge,
                                                    parthenon::Real& right_edge) {
    const parthenon::Real difference = right_edge - left_edge;
    const parthenon::Real curvature = center - 0.5 * (left_edge + right_edge);
    if ((right_edge - center) * (center - left_edge) <= 0.0) {
      left_edge = center;
      right_edge = center;
    } else if (difference * curvature > difference * difference / 6.0) {
      left_edge = 3.0 * center - 2.0 * right_edge;
    } else if (difference * curvature < -difference * difference / 6.0) {
      right_edge = 3.0 * center - 2.0 * left_edge;
    }
  }

  KOKKOS_INLINE_FUNCTION static void Reconstruct(
      const parthenon::Real qmmm, const parthenon::Real qmm, const parthenon::Real qm,
      const parthenon::Real q0, const parthenon::Real qp, const parthenon::Real qpp,
      parthenon::Real& left, parthenon::Real& right) {
    parthenon::Real left_cell_left = FourthOrderInterface(qmmm, qmm, qm, q0);
    parthenon::Real shared_edge = FourthOrderInterface(qmm, qm, q0, qp);
    parthenon::Real left_cell_right = shared_edge;
    LimitParabola(qm, left_cell_left, left_cell_right);
    parthenon::Real right_cell_left = shared_edge;
    parthenon::Real right_cell_right = FourthOrderInterface(qm, q0, qp, qpp);
    LimitParabola(q0, right_cell_left, right_cell_right);
    left = left_cell_right;
    right = right_cell_left;
  }
};

} // namespace pangu::reconstruct::model

#endif
