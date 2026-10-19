#ifndef PANGU_Z4C_GAUGE_SHIFTED_GAUGE_WAVE_HPP_
#define PANGU_Z4C_GAUGE_SHIFTED_GAUGE_WAVE_HPP_

#include <parthenon/parthenon.hpp>

namespace pangu::nr::gauge::shifted_gauge_wave {
using Real = parthenon::Real;

// For H=H(x-t), alpha=H^(-1/2), and beta^x=alpha^2-1.
KOKKOS_INLINE_FUNCTION Real LapseRhs(const Real dalpha_x) { return -dalpha_x; }
KOKKOS_INLINE_FUNCTION Real ShiftRhs(const int direction, const Real alpha,
                                     const Real dalpha_x) {
  return direction == 0 ? -2.0 * alpha * dalpha_x : 0.0;
}
} // namespace pangu::nr::gauge::shifted_gauge_wave

#endif
