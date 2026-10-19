#ifndef PANGU_Z4C_GAUGE_HARMONIC_HPP_
#define PANGU_Z4C_GAUGE_HARMONIC_HPP_

#include <parthenon/parthenon.hpp>

namespace pangu::nr::gauge::harmonic {
using Real = parthenon::Real;

KOKKOS_INLINE_FUNCTION Real LapseRhs(const Real alpha, const Real khat,
                                     const Real lie_alpha) {
  return lie_alpha - alpha * alpha * khat;
}

template <class SymmetricValue>
KOKKOS_INLINE_FUNCTION Real ShiftRhs(const int direction, const Real alpha,
                                     const Real chi, const Real gamma[3],
                                     const Real dalpha[3],
                                     const Real dchi[3], const Real inverse_metric[6],
                                     SymmetricValue symmetric_value) {
  // AwA harmonic spatial coordinates (Eq. 2.28 of arXiv:1810.12346).
  // The contracted conformal connection term is part of the gauge itself;
  // omitting it does not give harmonic shift even when the gradient terms
  // happen to vanish for a particular analytic solution.
  Real result = alpha * alpha * chi * gamma[direction];
  for (int derivative = 0; derivative < 3; ++derivative)
    result += alpha * (0.5 * alpha * dchi[derivative] - chi * dalpha[derivative]) *
              symmetric_value(inverse_metric, direction, derivative);
  return result;
}
} // namespace pangu::nr::gauge::harmonic

#endif
