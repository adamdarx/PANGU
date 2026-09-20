#pragma once

#include "electron/heating.h"

namespace pangu::electron::model {

struct Werner {
  static constexpr const char* key = "werner";
  static constexpr const char* label = "Kel_Werner";
  static constexpr bool requires_magnetic_field = true;

  template <typename T>
  KOKKOS_INLINE_FUNCTION static T Fraction(const HeatingState<T>& state) {
    const T sigma = fmin(fmax(state.magnetic_field_squared, static_cast<T>(0.0)) /
                             fmax(state.density, static_cast<T>(1.0e-20)),
                         static_cast<T>(1.0e100));
    const T scaled = sigma / static_cast<T>(5.0);
    return ClampHeatingFraction(static_cast<T>(0.25) *
                                (static_cast<T>(1.0) +
                                 sqrt(scaled / (static_cast<T>(2.0) + scaled))));
  }
};

} // namespace pangu::electron::model
