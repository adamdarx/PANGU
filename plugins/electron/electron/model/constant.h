#pragma once

#include "electron/heating.h"

namespace pangu::electron::model {

struct Constant {
  static constexpr const char* key = "constant";
  static constexpr const char* label = "Kel_Constant";
  static constexpr bool requires_magnetic_field = false;

  template <typename T>
  KOKKOS_INLINE_FUNCTION static T Fraction(const HeatingState<T>& state) {
    return ClampHeatingFraction(state.constant_fraction);
  }
};

} // namespace pangu::electron::model
