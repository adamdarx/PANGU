#ifndef PANGU_Z4C_GAUGE_DISPATCH_HPP_
#define PANGU_Z4C_GAUGE_DISPATCH_HPP_

#include "z4c/gauge/frozen.hpp"
#include "z4c/gauge/harmonic.hpp"
#include "z4c/gauge/moving_puncture.hpp"
#include "z4c/gauge/options.hpp"
#include "z4c/gauge/shifted_gauge_wave.hpp"

namespace pangu::nr::gauge {
using Real = parthenon::Real;

KOKKOS_INLINE_FUNCTION Real LapseRhs(const Options& options, const Real alpha,
                                     const Real khat, const Real lie_alpha,
                                     const Real dalpha[3]) {
  if (options.lapse == Lapse::frozen) return frozen::LapseRhs();
  if (options.lapse == Lapse::harmonic) return harmonic::LapseRhs(alpha, khat, lie_alpha);
  if (options.lapse == Lapse::one_plus_log)
    return moving_puncture::OnePlusLogLapseRhs(alpha, khat, lie_alpha);
  if (options.lapse == Lapse::shifted_gauge_wave)
    return shifted_gauge_wave::LapseRhs(dalpha[0]);
  const Real factor =
      options.lapse_oplog * options.lapse_harmonic_factor + options.lapse_harmonic * alpha;
  return options.lapse_advect * lie_alpha - factor * alpha * khat;
}

template <class SymmetricValue>
KOKKOS_INLINE_FUNCTION Real ShiftRhs(const Options& options, const int direction,
                                     const Real alpha, const Real chi, const Real gamma[3],
                                     const Real beta[3], const Real lie_beta[3],
                                     const Real dalpha[3], const Real dchi[3],
                                     const Real inverse_metric[6], SymmetricValue symmetric_value) {
  if (options.shift == Shift::frozen) return frozen::ShiftRhs();
  if (options.shift == Shift::gamma_driver)
    return moving_puncture::GammaDriverShiftRhs(direction, alpha, gamma, beta, lie_beta, options);
  if (options.shift == Shift::harmonic)
    return options.shift_advect * lie_beta[direction] +
           harmonic::ShiftRhs(direction, alpha, chi, gamma, dalpha, dchi, inverse_metric,
                              symmetric_value);
  if (options.shift == Shift::shifted_gauge_wave)
    return shifted_gauge_wave::ShiftRhs(direction, alpha, dalpha[0]);

  Real result = moving_puncture::GammaDriverShiftRhs(direction, alpha, gamma, beta, lie_beta,
                                                     options);
  // Preserve the legacy coefficient-based gauge. The explicit harmonic
  // selector above implements Eq. (28); legacy shift_harmonic weights only
  // the original gradient term, while shift_alpha2_gamma weights Gammahat.
  if (options.shift_harmonic != 0.0)
    for (int derivative = 0; derivative < 3; ++derivative)
      result += options.shift_harmonic * alpha * chi *
                (0.5 * alpha * dchi[derivative] - dalpha[derivative]) *
                symmetric_value(inverse_metric, direction, derivative);
  return result;
}
} // namespace pangu::nr::gauge

#endif
