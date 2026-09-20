#pragma once

#include "electron/heating.h"

namespace pangu::electron::model {

struct Sharma {
  static constexpr const char* key = "sharma";
  static constexpr const char* label = "Kel_Sharma";
  static constexpr bool requires_magnetic_field = false;

  template <typename T>
  KOKKOS_INLINE_FUNCTION static T Fraction(const HeatingState<T>& state) {
    constexpr T small = static_cast<T>(1.0e-20);
    const T electron_to_ion =
        static_cast<T>(0.33) * sqrt(static_cast<T>(1.0) / ProtonToElectronTemperature(state));
    return electron_to_ion <= small
               ? static_cast<T>(0.0)
               : ClampHeatingFraction(electron_to_ion / (static_cast<T>(1.0) + electron_to_ion));
  }
};

} // namespace pangu::electron::model
