#pragma once

#include <array>
#include <tuple>

#include "electron/model/constant.h"
#include "electron/model/howes.h"
#include "electron/model/kawazura.h"
#include "electron/model/rowan.h"
#include "electron/model/sharma.h"
#include "electron/model/werner.h"

namespace pangu::electron::model {

template <typename... Models> struct Registry {
  static constexpr int size = sizeof...(Models);
  inline static constexpr std::array<const char*, size> keys{Models::key...};
  inline static constexpr std::array<const char*, size> labels{Models::label...};
  inline static constexpr std::array<bool, size> magnetic_requirements{
      Models::requires_magnetic_field...};

  template <int Index = 0, typename T>
  KOKKOS_INLINE_FUNCTION static T Fraction(const int model_index,
                                           const HeatingState<T>& state) {
    if constexpr (Index < size) {
      if (model_index == Index)
        return std::tuple_element_t<Index, std::tuple<Models...>>::template Fraction<T>(state);
      return Fraction<Index + 1>(model_index, state);
    }
    return static_cast<T>(0.0);
  }
};

// The sole model assembly point. Adding a model requires its implementation
// file and one entry here; the package, fields, and GPU dispatch remain unchanged.
using RegisteredModels = Registry<Constant, Howes, Kawazura, Werner, Rowan, Sharma>;

} // namespace pangu::electron::model

namespace pangu::electron {

inline constexpr int heating_model_count = model::RegisteredModels::size;

template <typename T>
KOKKOS_INLINE_FUNCTION T HeatingFraction(const int model_index, const T density,
                                         const T internal_energy, const T magnetic_field_squared,
                                         const T electron_entropy, const T gamma_gas,
                                         const T gamma_electron, const T gamma_proton,
                                         const T constant_fraction) {
  return model::RegisteredModels::Fraction(
      model_index, HeatingState<T>{density, internal_energy, magnetic_field_squared,
                                   electron_entropy, gamma_gas, gamma_electron, gamma_proton,
                                   constant_fraction});
}

} // namespace pangu::electron
