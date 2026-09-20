#pragma once

#include "electron/heating.h"

namespace pangu::electron::model {

struct Rowan {
  static constexpr const char* key = "rowan";
  static constexpr const char* label = "Kel_Rowan";
  static constexpr bool requires_magnetic_field = true;

  template <typename T>
  KOKKOS_INLINE_FUNCTION static T Fraction(const HeatingState<T>& state) {
    constexpr T small = static_cast<T>(1.0e-20);
    const T rho = fmax(state.density, small);
    const T proton_pressure = (state.gamma_proton - static_cast<T>(1.0)) * state.internal_energy;
    const T gas_pressure = (state.gamma_gas - static_cast<T>(1.0)) * state.internal_energy;
    const T enthalpy = fmax(rho + state.internal_energy + gas_pressure, small);
    const T sigma = fmin(fmax(state.magnetic_field_squared, static_cast<T>(0.0)) / enthalpy,
                         static_cast<T>(1.0e100));
    const T fit_base =
        fmax(static_cast<T>(1.0) - static_cast<T>(8.0) * proton_pressure / enthalpy,
             static_cast<T>(0.0));
    return ClampHeatingFraction(
        static_cast<T>(0.5) *
        exp(-pow(fit_base, static_cast<T>(3.3)) /
            (static_cast<T>(1.0) + static_cast<T>(1.2) * pow(sigma, static_cast<T>(0.7)))));
  }
};

} // namespace pangu::electron::model
