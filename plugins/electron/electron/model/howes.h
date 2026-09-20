#pragma once

#include "electron/heating.h"

namespace pangu::electron::model {

struct Howes {
  static constexpr const char* key = "howes";
  static constexpr const char* label = "Kel_Howes";
  static constexpr bool requires_magnetic_field = true;

  template <typename T>
  KOKKOS_INLINE_FUNCTION static T Fraction(const HeatingState<T>& state) {
    constexpr T small = static_cast<T>(1.0e-20);
    constexpr T maximum = static_cast<T>(1.0e20);
    constexpr T electron_mass = static_cast<T>(9.1093826e-28);
    constexpr T proton_mass = static_cast<T>(1.67262171e-24);
    const T rho = fmax(state.density, small);
    const T ratio = ProtonToElectronTemperature(state);
    const T proton_temperature =
        fmax((state.gamma_proton - static_cast<T>(1.0)) * state.internal_energy / rho, small);
    const T bsq = fmax(state.magnetic_field_squared, static_cast<T>(0.0));
    const T beta = bsq > 0 ? fmin(static_cast<T>(2.0) * rho * proton_temperature / bsq, maximum)
                          : maximum;
    const T safe_beta = fmax(beta, small);
    const T log_ratio = log10(ratio);
    const T exponent = static_cast<T>(2.0) - static_cast<T>(0.2) * log_ratio;
    const T c2 = (ratio <= 1 ? static_cast<T>(1.6) : static_cast<T>(1.2)) / ratio;
    const T c3 = ratio <= 1 ? static_cast<T>(18.0) + static_cast<T>(5.0) * log_ratio
                            : static_cast<T>(18.0);
    const T beta_power = pow(safe_beta, exponent);
    const T ion_to_electron = static_cast<T>(0.92) * (c2 * c2 + beta_power) /
                              (c3 * c3 + beta_power) * exp(-static_cast<T>(1.0) / safe_beta) *
                              sqrt(proton_mass / electron_mass * ratio);
    return ClampHeatingFraction(static_cast<T>(1.0) / (static_cast<T>(1.0) + ion_to_electron));
  }
};

} // namespace pangu::electron::model
