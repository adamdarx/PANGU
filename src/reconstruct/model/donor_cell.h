#ifndef PANGU_RECONSTRUCT_MODEL_DONOR_CELL_H_
#define PANGU_RECONSTRUCT_MODEL_DONOR_CELL_H_

#include <parthenon/parthenon.hpp>

namespace pangu::reconstruct::model {

struct DonorCell {
  // Reads the two cells adjacent to the face.
  static constexpr int stencil_radius = 1;
  static constexpr bool cell_edge_states = false;

  KOKKOS_INLINE_FUNCTION static void Reconstruct(
      const parthenon::Real, const parthenon::Real, const parthenon::Real qm,
      const parthenon::Real q0, const parthenon::Real, const parthenon::Real,
      parthenon::Real& left, parthenon::Real& right) {
    left = qm;
    right = q0;
  }
};

} // namespace pangu::reconstruct::model

#endif
