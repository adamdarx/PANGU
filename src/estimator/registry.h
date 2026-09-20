#ifndef PANGU_ESTIMATOR_REGISTRY_H_
#define PANGU_ESTIMATOR_REGISTRY_H_

#include <array>
#include <concepts>
#include <cstddef>
#include <string_view>
#include <tuple>

#include <parthenon/parthenon.hpp>

#include "estimator/model/light.h"
#include "estimator/model/wave.h"

namespace pangu::estimator {

// A cell timestep is Finish(Add(...Add(Start(), c_1, dx_1)..., c_n, dx_n)) over
// the active directions, scaled by the package CFL number.  Capabilities:
//   characteristic_gr_speeds  GR speeds are fluid characteristics bounded by the
//                             coordinate light cone; otherwise unit light speed.
//   face_signal_speeds        prefer face speeds cached by the Riemann solve.
template <typename T>
concept Model = requires(const parthenon::Real value) {
  { T::name } -> std::convertible_to<std::string_view>;
  { T::characteristic_gr_speeds } -> std::convertible_to<bool>;
  { T::face_signal_speeds } -> std::convertible_to<bool>;
  { T::Start() } -> std::same_as<parthenon::Real>;
  { T::Add(value, value, value) } -> std::same_as<parthenon::Real>;
  { T::Finish(value) } -> std::same_as<parthenon::Real>;
};

template <Model... Models> struct Registry {
  static constexpr std::size_t size = sizeof...(Models);
  static constexpr std::array<std::string_view, size> names{Models::name...};

  // Returns `size` for an unregistered name.
  static constexpr std::size_t Find(const std::string_view name) {
    for (std::size_t index = 0; index < size; ++index) {
      if (names[index] == name)
        return index;
    }
    return size;
  }

  template <std::size_t Index> using At = std::tuple_element_t<Index, std::tuple<Models...>>;
};

// The sole estimator assembly point.  ESTIMATOR selects an entry by name.
using Registered = Registry<model::Light, model::Wave>;

} // namespace pangu::estimator

#endif
