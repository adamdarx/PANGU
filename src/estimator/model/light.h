#ifndef PANGU_ESTIMATOR_MODEL_LIGHT_H_
#define PANGU_ESTIMATOR_MODEL_LIGHT_H_

#include <limits>
#include <string_view>

#include <parthenon/parthenon.hpp>

namespace pangu::estimator::model {

// Directional crossing limit dt = min_d dx_d / c_d.  In GR the signal speed is
// the unit coordinate light speed, so the limit is independent of the fluid.
struct Light {
  static constexpr std::string_view name = "light";
  static constexpr bool characteristic_gr_speeds = false;
  static constexpr bool face_signal_speeds = false;

  KOKKOS_INLINE_FUNCTION static parthenon::Real Start() {
    return std::numeric_limits<parthenon::Real>::max();
  }
  KOKKOS_INLINE_FUNCTION static parthenon::Real Add(const parthenon::Real dt,
                                                    const parthenon::Real speed,
                                                    const parthenon::Real dx) {
    return fmin(dt, dx / speed);
  }
  KOKKOS_INLINE_FUNCTION static parthenon::Real Finish(const parthenon::Real dt) { return dt; }
};

} // namespace pangu::estimator::model

#endif
