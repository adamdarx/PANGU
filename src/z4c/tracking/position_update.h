#pragma once

#include <stdexcept>
#include <string>

namespace pangu::nr::tracking {

enum class Mode { ode, walk };

inline Mode ParseMode(const std::string &name) {
  if (name == "ode")
    return Mode::ode;
  if (name == "walk")
    return Mode::walk;
  throw std::invalid_argument(
      "numerical_relativity/tracker_mode must be ode or walk");
}

// The velocity of a black-hole puncture is minus the interpolated shift.
// Match AthenaK's once-per-completed-step explicit ODE update. In particular,
// do not snap a stationary puncture to a neighboring cell center.
template <class Real>
inline Real AdvanceOdePosition(const Real position, const Real velocity,
                               const Real dt) {
  return position + dt * velocity;
}

} // namespace pangu::nr::tracking
