#ifndef PANGU_Z4C_GAUGE_MOVING_PUNCTURE_HPP_
#define PANGU_Z4C_GAUGE_MOVING_PUNCTURE_HPP_

#include <parthenon/parthenon.hpp>

#include "z4c/gauge/options.hpp"

namespace pangu::nr::gauge::moving_puncture {
using Real = parthenon::Real;

KOKKOS_INLINE_FUNCTION Real OnePlusLogLapseRhs(const Real alpha, const Real khat,
                                               const Real lie_alpha) {
  return lie_alpha - 2.0 * alpha * khat;
}

KOKKOS_INLINE_FUNCTION Real GammaDriverShiftRhs(const int direction, const Real alpha,
                                                const Real gamma[3], const Real beta[3],
                                                const Real lie_beta[3], const Options& options) {
  return options.shift_gamma * gamma[direction] + options.shift_advect * lie_beta[direction] -
         options.shift_eta * beta[direction] +
         options.shift_alpha2_gamma * alpha * alpha * gamma[direction];
}
} // namespace pangu::nr::gauge::moving_puncture

#endif
