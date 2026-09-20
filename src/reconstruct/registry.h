#ifndef PANGU_RECONSTRUCT_REGISTRY_H_
#define PANGU_RECONSTRUCT_REGISTRY_H_

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

#include "reconstruct/model/donor_cell.h"
#include "reconstruct/model/plm.h"
#include "reconstruct/model/ppm.h"
#include "reconstruct/model/ppmc.h"
#include "reconstruct/model/wenoz.h"

namespace pangu::reconstruct {

// A reconstruction is selected by its position in a registry, so the enumeration
// carries no names of its own.  Configuration, restart files and package
// parameters keep storing that position as an integer.
enum class Method : int {};

// One registry line describes a model completely: its configuration name, the
// ghost-zone widths its stencil needs on the Newtonian and relativistic paths,
// and the physics it is valid for.  The stencil radius and the device kernel
// come from the model type itself.
template <typename Model> struct Entry {
  using Implementation = Model;

  std::string_view name;
  int ghost_zones;
  int relativistic_ghost_zones;
  bool relativistic_hydro;
  bool relativistic_mhd;
  bool passive_transport;
  bool radiation_transport;
};

// Adding a reconstruction means adding one line here and nothing else.
inline constexpr std::tuple registry{
    Entry<model::DonorCell>{"dc", 2, 2, true, false, false, true},
    Entry<model::PLM>{"plm", 2, 2, true, true, true, true},
    Entry<model::PPM>{"ppm", 3, 4, true, true, true, false},
    Entry<model::PPMC>{"ppmc", 3, 3, true, false, false, false},
    Entry<model::WENOZ>{"wenoz", 3, 3, true, true, false, false},
};

struct Descriptor {
  Method method;
  std::string_view name;
  int stencil_radius;
  int ghost_zones;
  int relativistic_ghost_zones;
  bool relativistic_hydro;
  bool relativistic_mhd;
  bool passive_transport;
  bool radiation_transport;
};

template <const auto& Registry>
inline constexpr std::size_t count = std::tuple_size_v<std::decay_t<decltype(Registry)>>;

template <const auto& Registry, std::size_t Index>
using ImplementationAt =
    typename std::tuple_element_t<Index, std::decay_t<decltype(Registry)>>::Implementation;

template <const auto& Registry, Method method>
using ImplementationOf = ImplementationAt<Registry, static_cast<std::size_t>(method)>;

namespace detail {

template <const auto& Registry, std::size_t Index> constexpr Descriptor Describe() {
  const auto& entry = std::get<Index>(Registry);
  return Descriptor{static_cast<Method>(Index),
                    entry.name,
                    ImplementationAt<Registry, Index>::stencil_radius,
                    entry.ghost_zones,
                    entry.relativistic_ghost_zones,
                    entry.relativistic_hydro,
                    entry.relativistic_mhd,
                    entry.passive_transport,
                    entry.radiation_transport};
}

template <const auto& Registry, std::size_t... Index>
constexpr auto BuildDescriptors(std::index_sequence<Index...>) {
  return std::array<Descriptor, sizeof...(Index)>{Describe<Registry, Index>()...};
}

} // namespace detail

template <const auto& Registry>
inline constexpr auto descriptorsOf =
    detail::BuildDescriptors<Registry>(std::make_index_sequence<count<Registry>>{});

template <const auto& Registry = registry>
constexpr const Descriptor& Describe(const Method method) {
  return descriptorsOf<Registry>[static_cast<std::size_t>(method)];
}

// `ppm4` names the same fourth-order PPM interpolation as `ppm`; the alias predates
// the registry and stays valid input.
template <const auto& Registry = registry>
constexpr std::optional<Method> Parse(const std::string_view name) {
  for (const auto& descriptor : descriptorsOf<Registry>) {
    if (descriptor.name == name || (descriptor.name == "ppm" && name == "ppm4"))
      return descriptor.method;
  }
  return std::nullopt;
}

namespace detail {

template <const auto& Registry, bool Descriptor::*Capability>
constexpr bool Supports(const std::size_t index) {
  if constexpr (Capability == nullptr)
    return true;
  else
    return descriptorsOf<Registry>[index].*Capability;
}

template <const auto& Registry, bool Descriptor::*Capability> constexpr std::size_t FirstSupported() {
  for (std::size_t index = 0; index < count<Registry>; ++index) {
    if (Supports<Registry, Capability>(index))
      return index;
  }
  return count<Registry>;
}

// Dispatch to the registry entry the configuration selected.  `Capability`
// restricts the instantiated set, so a physics path never builds a kernel for a
// model it rejects; an unsupported selection fails here instead of falling back
// on some other model.
template <const auto& Registry, bool Descriptor::*Capability, typename Visitor,
          std::size_t... Index>
decltype(auto) VisitIndexed(const Method method, Visitor&& visitor, const char* requirement,
                            std::index_sequence<Index...>) {
  constexpr std::size_t first = FirstSupported<Registry, Capability>();
  static_assert(first < count<Registry>, "no registered reconstruction provides this capability");
  using Result =
      decltype(std::forward<Visitor>(visitor).template operator()<static_cast<Method>(first)>());
  Result (*selected)(Visitor&&) = nullptr;
  (
      [&] {
        if constexpr (Supports<Registry, Capability>(Index)) {
          if (method == static_cast<Method>(Index)) {
            selected = [](Visitor&& forwarded) -> Result {
              return std::forward<Visitor>(forwarded)
                  .template operator()<static_cast<Method>(Index)>();
            };
          }
        }
      }(),
      ...);
  if (selected == nullptr)
    throw std::invalid_argument(requirement);
  return selected(std::forward<Visitor>(visitor));
}

} // namespace detail

template <const auto& Registry = registry, typename Visitor>
decltype(auto) Visit(const Method method, Visitor&& visitor) {
  return detail::VisitIndexed<Registry, nullptr>(
      method, std::forward<Visitor>(visitor), "invalid reconstruction method",
      std::make_index_sequence<count<Registry>>{});
}

template <const auto& Registry = registry, typename Visitor>
decltype(auto) VisitRelativisticMHD(const Method method, Visitor&& visitor) {
  return detail::VisitIndexed<Registry, &Descriptor::relativistic_mhd>(
      method, std::forward<Visitor>(visitor),
      "reconstruction does not support relativistic MHD",
      std::make_index_sequence<count<Registry>>{});
}

template <const auto& Registry = registry, typename Visitor>
decltype(auto) VisitPassiveTransport(const Method method, Visitor&& visitor) {
  return detail::VisitIndexed<Registry, &Descriptor::passive_transport>(
      method, std::forward<Visitor>(visitor),
      "reconstruction does not support passive transport",
      std::make_index_sequence<count<Registry>>{});
}

template <const auto& Registry = registry, typename Visitor>
decltype(auto) VisitRadiationTransport(const Method method, Visitor&& visitor) {
  return detail::VisitIndexed<Registry, &Descriptor::radiation_transport>(
      method, std::forward<Visitor>(visitor),
      "reconstruction does not support radiation transport",
      std::make_index_sequence<count<Registry>>{});
}

template <Method method> using Implementation = ImplementationOf<registry, method>;

} // namespace pangu::reconstruct

#endif
