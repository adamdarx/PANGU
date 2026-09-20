#ifndef PANGU_ESTIMATOR_MODEL_WAVE_H_
#define PANGU_ESTIMATOR_MODEL_WAVE_H_

#include <limits>
#include <string_view>

#include <parthenon/parthenon.hpp>

namespace pangu::estimator::model {

// Summed characteristic limit dt = 1 / sum_d (c_d / dx_d).  GR signal speeds are
// the fastest fluid characteristics, and packages that solve face Riemann
// problems may supply their cached face speeds instead of cell-centred ones.
struct Wave {
  static constexpr std::string_view name = "wave";
  static constexpr bool characteristic_gr_speeds = true;
  static constexpr bool face_signal_speeds = true;

  KOKKOS_INLINE_FUNCTION static parthenon::Real Start() { return 0.0; }
  KOKKOS_INLINE_FUNCTION static parthenon::Real Add(const parthenon::Real rate,
                                                    const parthenon::Real speed,
                                                    const parthenon::Real dx) {
    return rate + speed / dx;
  }
  KOKKOS_INLINE_FUNCTION static parthenon::Real Finish(const parthenon::Real rate) {
    return rate > 0.0 ? 1.0 / rate : std::numeric_limits<parthenon::Real>::max();
  }
};

} // namespace pangu::estimator::model

#endif
