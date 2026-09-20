#pragma once

#include <cmath>

#include <Kokkos_Core.hpp>

namespace pangu::electron {

template <typename T> struct HeatingState {
  T density;
  T internal_energy;
  T magnetic_field_squared;
  T electron_entropy;
  T gamma_gas;
  T gamma_electron;
  T gamma_proton;
  T constant_fraction;
};

enum HeatingFlag : int {
  heating_none = 0,
  heating_negative_dissipation_clipped = 1 << 0,
  heating_high_magnetization_suppressed = 1 << 1,
  heating_kel_min_applied = 1 << 2,
  heating_kel_max_applied = 1 << 3
};

template <typename T> struct EntropyBounds {
  T minimum;
  T maximum;
};

template <typename T> struct ConstantHeatingResult {
  T total_entropy;
  T electron_entropy;
  T raw_dissipation;
  T applied_dissipation;
  T heating_fraction;
  T temperature_ratio;
  int flags;
};

template <typename T> struct ElectronEntropyUpdate {
  T electron_entropy;
  T heating_fraction;
  T temperature_ratio;
  int flags;
};

template <typename T> KOKKOS_INLINE_FUNCTION T ClampHeatingFraction(const T fraction) {
  return fmin(static_cast<T>(1.0), fmax(static_cast<T>(0.0), fraction));
}

template <typename T>
KOKKOS_INLINE_FUNCTION T ProtonToElectronTemperature(const HeatingState<T>& state) {
  constexpr T small = static_cast<T>(1.0e-20);
  constexpr T maximum_beta = static_cast<T>(1.0e20);
  const T rho = fmax(state.density, small);
  const T proton_temperature =
      fmax((state.gamma_proton - static_cast<T>(1.0)) * state.internal_energy / rho, small);
  const T electron_temperature =
      fmax(state.electron_entropy * pow(rho, state.gamma_electron - static_cast<T>(1.0)), small);
  return fmin(fmax(proton_temperature / electron_temperature, small), maximum_beta);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T EnergyConservingEntropy(const T density, const T internal_energy,
                                                 const T gamma_gas) {
  return (gamma_gas - 1.0) * internal_energy * pow(density, -gamma_gas);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T DissipativeElectronEntropy(const T old_density,
                                                    const T energy_conserving_entropy,
                                                    const T advected_total_entropy,
                                                    const T gamma_gas, const T gamma_electron) {
  return (gamma_electron - 1.0) / (gamma_gas - 1.0) * pow(old_density, gamma_gas - gamma_electron) *
         (energy_conserving_entropy - advected_total_entropy);
}

template <typename T>
KOKKOS_INLINE_FUNCTION EntropyBounds<T>
ElectronEntropyBounds(const T old_density, const T old_total_entropy, const T gamma_gas,
                      const T gamma_electron, const T gamma_proton, const T tp_over_te_min,
                      const T tp_over_te_max) {
  const T numerator = old_total_entropy * pow(old_density, gamma_gas - gamma_electron);
  const T gas_over_proton = (gamma_gas - 1.0) / (gamma_proton - 1.0);
  const T gas_over_electron = (gamma_gas - 1.0) / (gamma_electron - 1.0);
  return {numerator / (tp_over_te_max * gas_over_proton + gas_over_electron),
          numerator / (tp_over_te_min * gas_over_proton + gas_over_electron)};
}

template <typename T>
KOKKOS_INLINE_FUNCTION T TemperatureRatio(const T density, const T internal_energy,
                                          const T electron_entropy, const T gamma_electron,
                                          const T gamma_proton) {
  constexpr T small = static_cast<T>(1.0e-30);
  const T proton_temperature = fmax((gamma_proton - 1.0) * internal_energy / density, small);
  const T electron_temperature = fmax(electron_entropy * pow(density, gamma_electron - 1.0), small);
  return proton_temperature / electron_temperature;
}

template <typename T>
KOKKOS_INLINE_FUNCTION ElectronEntropyUpdate<T>
ApplyElectronEntropyUpdate(const T advected_electron_entropy, const T heating_fraction,
                           const T applied_dissipation, const EntropyBounds<T> bounds,
                           const bool limit_electron_entropy, const T new_density,
                           const T new_internal_energy, const T gamma_electron,
                           const T gamma_proton, const int inherited_flags = heating_none) {
  ElectronEntropyUpdate<T> result{};
  result.heating_fraction = ClampHeatingFraction(heating_fraction);
  result.electron_entropy =
      advected_electron_entropy + result.heating_fraction * applied_dissipation;
  result.flags = inherited_flags;
  if (limit_electron_entropy) {
    if (result.electron_entropy < bounds.minimum) {
      result.electron_entropy = bounds.minimum;
      result.flags |= heating_kel_min_applied;
    }
    if (result.electron_entropy > bounds.maximum) {
      result.electron_entropy = bounds.maximum;
      result.flags |= heating_kel_max_applied;
    }
  }
  result.temperature_ratio = TemperatureRatio(
      new_density, new_internal_energy, result.electron_entropy, gamma_electron, gamma_proton);
  return result;
}

template <typename T>
KOKKOS_INLINE_FUNCTION ConstantHeatingResult<T> ApplyConstantHeating(
    const T old_density, const T old_total_entropy, const T new_density,
    const T new_internal_energy, const T advected_total_entropy, const T advected_electron_entropy,
    const T gamma_gas, const T gamma_electron, const T gamma_proton, const T electron_fraction,
    const bool enforce_positive_dissipation, const bool suppress_high_magnetization,
    const T magnetization, const T magnetization_cutoff, const bool limit_electron_entropy,
    const T tp_over_te_min, const T tp_over_te_max) {
  ConstantHeatingResult<T> result{};
  result.total_entropy = EnergyConservingEntropy(new_density, new_internal_energy, gamma_gas);
  result.raw_dissipation = DissipativeElectronEntropy(
      old_density, result.total_entropy, advected_total_entropy, gamma_gas, gamma_electron);
  result.applied_dissipation = result.raw_dissipation;
  result.heating_fraction = electron_fraction;
  result.flags = heating_none;

  if (suppress_high_magnetization && magnetization > magnetization_cutoff) {
    result.applied_dissipation = 0.0;
    result.heating_fraction = 0.0;
    result.flags |= heating_high_magnetization_suppressed;
  } else if (enforce_positive_dissipation && result.applied_dissipation < 0.0) {
    result.applied_dissipation = 0.0;
    result.flags |= heating_negative_dissipation_clipped;
  }

  const auto bounds =
      ElectronEntropyBounds(old_density, old_total_entropy, gamma_gas, gamma_electron, gamma_proton,
                            tp_over_te_min, tp_over_te_max);
  const T effective_fraction = (result.flags & heating_high_magnetization_suppressed) != 0
                                   ? static_cast<T>(0.0)
                                   : electron_fraction;
  const auto update = ApplyElectronEntropyUpdate(
      advected_electron_entropy, effective_fraction, result.applied_dissipation, bounds,
      limit_electron_entropy, new_density, new_internal_energy, gamma_electron, gamma_proton,
      result.flags);
  result.electron_entropy = update.electron_entropy;
  result.heating_fraction = update.heating_fraction;
  result.temperature_ratio = update.temperature_ratio;
  result.flags = update.flags;
  return result;
}

} // namespace pangu::electron
