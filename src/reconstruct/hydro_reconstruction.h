#ifndef PANGU_RECONSTRUCT_HYDRO_RECONSTRUCTION_H_
#define PANGU_RECONSTRUCT_HYDRO_RECONSTRUCTION_H_

#include "hydro/hydro_types.h"
#include "reconstruct/registry.h"

namespace pangu::reconstruct {

using hydro::Reconstruction;
using parthenon::Real;

// Every model takes the same six-point stencil centred on the face between cells
// -1 and 0.  Slots outside a model's own radius are never read by it and stay
// zero, so a narrow model never touches cells beyond its ghost-zone budget.
inline constexpr int kStencilWidth = 6;
inline constexpr int kStencilCenter = 3;

template <typename ReconstructionModel>
KOKKOS_INLINE_FUNCTION void ReconstructWith(const Real qmmm, const Real qmm, const Real qm,
                                            const Real q0, const Real qp, const Real qpp,
                                            Real& left, Real& right) {
  ReconstructionModel::Reconstruct(qmmm, qmm, qm, q0, qp, qpp, left, right);
}

// Load exactly the cells a model's stencil radius covers and reconstruct the
// face states.  `read(offset)` returns the value at the cell `offset` away from
// the face's right neighbour, so the model sees offsets [-radius, radius).
template <typename ReconstructionModel, typename Read>
KOKKOS_INLINE_FUNCTION void ReconstructStencil(const Read& read, Real& left, Real& right) {
  constexpr int radius = ReconstructionModel::stencil_radius;
  static_assert(radius >= 1 && radius <= kStencilCenter,
                "reconstruction stencil radius exceeds the shared stencil width");
  Real stencil[kStencilWidth]{};
  for (int slot = kStencilCenter - radius; slot < kStencilCenter + radius; ++slot)
    stencil[slot] = read(slot - kStencilCenter);
  ReconstructWith<ReconstructionModel>(stencil[0], stencil[1], stencil[2], stencil[3], stencil[4],
                                       stencil[5], left, right);
}

template <Reconstruction Method, typename Read>
KOKKOS_INLINE_FUNCTION void ReconstructFaceStencil(const Read& read, Real& left, Real& right) {
  ReconstructStencil<Implementation<Method>>(read, left, right);
}

template <Reconstruction Method>
KOKKOS_INLINE_FUNCTION void ReconstructFace(const Real qmmm, const Real qmm, const Real qm,
                                            const Real q0, const Real qp, const Real qpp,
                                            Real& left, Real& right) {
  ReconstructWith<Implementation<Method>>(qmmm, qmm, qm, q0, qp, qpp, left, right);
}

} // namespace pangu::reconstruct

#endif
