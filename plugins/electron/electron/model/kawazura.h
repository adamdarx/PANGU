#pragma once

#include "electron/heating.h"

namespace pangu::electron::model {

struct Kawazura {
  static constexpr const char* key = "kawazura";
  static constexpr const char* label = "Kel_Kawazura";
  static constexpr bool requires_magnetic_field = true;

  template <typename T>
  KOKKOS_INLINE_FUNCTION static T Fraction(const HeatingState<T>& state) {
    constexpr T small = static_cast<T>(1.0e-20);
    constexpr T maximum = static_cast<T>(1.0e20);
    const T rho = fmax(state.density, small);
    const T ratio = ProtonToElectronTemperature(state);
    const T proton_temperature =
        fmax((state.gamma_proton - static_cast<T>(1.0)) * state.internal_energy / rho, small);
    const T bsq = fmax(state.magnetic_field_squared, static_cast<T>(0.0));
    const T beta = bsq > 0 ? fmin(static_cast<T>(2.0) * rho * proton_temperature / bsq, maximum)
                          : maximum;
    const T ion_to_electron =
        static_cast<T>(35.0) /
        (static_cast<T>(1.0) + pow(fmax(beta, small) / static_cast<T>(15.0),
                                    static_cast<T>(-1.4)) *
                                   exp(static_cast<T>(-0.1) / ratio));
    return ClampHeatingFraction(static_cast<T>(1.0) / (static_cast<T>(1.0) + ion_to_electron));
  }
};

} // namespace pangu::electron::model
