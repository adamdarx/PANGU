#ifndef PANGU_RECONSTRUCT_MODEL_PLM_H_
#define PANGU_RECONSTRUCT_MODEL_PLM_H_

#include <parthenon/parthenon.hpp>

namespace pangu::reconstruct::model {

struct PLM {
  // Reads two cells on each side of the face.
  static constexpr int stencil_radius = 2;
  // The slope limiter produces both edge states of one cell, so callers that
  // keep face-indexed buffers can write a cell's two edges in one pass.
  static constexpr bool cell_edge_states = true;

  KOKKOS_INLINE_FUNCTION static parthenon::Real MonotonizedCentral(
      const parthenon::Real left, const parthenon::Real center,
      const parthenon::Real right) {
    const parthenon::Real delta_left = center - left;
    const parthenon::Real delta_right = right - center;
    const parthenon::Real product = delta_left * delta_right;
    parthenon::Real increment = product / (delta_left + delta_right);
    if (product <= 0.0)
      increment = 0.0;
    return increment;
  }

  // Edge states of the cell whose neighbours are qm and qp.
  KOKKOS_INLINE_FUNCTION static void CellEdges(const parthenon::Real qm, const parthenon::Real q0,
                                               const parthenon::Real qp,
                                               parthenon::Real& low_edge,
                                               parthenon::Real& high_edge) {
    const parthenon::Real increment = MonotonizedCentral(qm, q0, qp);
    low_edge = q0 - increment;
    high_edge = q0 + increment;
  }

  KOKKOS_INLINE_FUNCTION static void Reconstruct(
      const parthenon::Real, const parthenon::Real qmm, const parthenon::Real qm,
      const parthenon::Real q0, const parthenon::Real qp, const parthenon::Real,
      parthenon::Real& left, parthenon::Real& right) {
    left = qm + MonotonizedCentral(qmm, qm, q0);
    right = q0 - MonotonizedCentral(qm, q0, qp);
  }
};

} // namespace pangu::reconstruct::model

#endif
