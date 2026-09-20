#ifndef PANGU_RIEMANN_REGISTRY_H_
#define PANGU_RIEMANN_REGISTRY_H_

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "riemann/newtonian_hydro/hllc.h"
#include "riemann/newtonian_hydro/hlle.h"
#include "riemann/newtonian_hydro/llf.h"
#include "riemann/newtonian_hydro/roe.h"
#include "riemann/newtonian_mhd/hlld.h"
#include "riemann/newtonian_mhd/hlle.h"
#include "riemann/newtonian_mhd/llf.h"
#include "riemann/relativistic/hlle.h"
#include "riemann/relativistic/llf.h"
#include "riemann/relativistic_mhd/hlle.h"
#include "riemann/relativistic_mhd/llf.h"

namespace pangu::riemann {

// `none` is the kinematic mode: fluxes are zero and no solver is dispatched.
enum class Solver : int { llf = 0, hlle = 1, hllc = 2, roe = 3, hlld = 4, none = 5 };

enum class Physics : int { newtonian_hydro, newtonian_mhd, relativistic_hydro, relativistic_mhd };

struct Descriptor {
  Solver solver;
  std::string_view name;
  bool newtonian_hydro;
  bool newtonian_mhd;
  bool relativistic_hydro;
  bool relativistic_mhd;
  // Whether the solver closes without an energy equation.
  bool isothermal;

  constexpr bool Supports(const Physics physics) const {
    switch (physics) {
    case Physics::newtonian_hydro:
      return newtonian_hydro;
    case Physics::newtonian_mhd:
      return newtonian_mhd;
    case Physics::relativistic_hydro:
      return relativistic_hydro;
    case Physics::relativistic_mhd:
      return relativistic_mhd;
    }
    return false;
  }
};

inline constexpr std::array descriptors{
    Descriptor{Solver::llf, "llf", true, true, true, true, true},
    Descriptor{Solver::hlle, "hlle", true, true, true, true, true},
    Descriptor{Solver::hllc, "hllc", true, false, false, false, false},
    Descriptor{Solver::roe, "roe", true, false, false, false, true},
    Descriptor{Solver::hlld, "hlld", false, true, false, false, true},
    Descriptor{Solver::none, "none", true, true, false, false, true},
};

// First-order flux correction replaces flagged faces with this solver in every physics.
inline constexpr Solver first_order_fallback = Solver::llf;

template <Physics, Solver> struct Implementation;

template <> struct Implementation<Physics::newtonian_hydro, Solver::llf> : newtonian_hydro::LLF {};
template <>
struct Implementation<Physics::newtonian_hydro, Solver::hlle> : newtonian_hydro::HLLE {};
template <>
struct Implementation<Physics::newtonian_hydro, Solver::hllc> : newtonian_hydro::HLLC {};
template <> struct Implementation<Physics::newtonian_hydro, Solver::roe> : newtonian_hydro::Roe {};
template <> struct Implementation<Physics::newtonian_mhd, Solver::llf> : newtonian_mhd::LLF {};
template <> struct Implementation<Physics::newtonian_mhd, Solver::hlle> : newtonian_mhd::HLLE {};
template <> struct Implementation<Physics::newtonian_mhd, Solver::hlld> : newtonian_mhd::HLLD {};
template <> struct Implementation<Physics::relativistic_hydro, Solver::llf> : relativistic::LLF {};
template <>
struct Implementation<Physics::relativistic_hydro, Solver::hlle> : relativistic::HLLE {};
template <>
struct Implementation<Physics::relativistic_mhd, Solver::llf> : relativistic_mhd::LLF {};
template <>
struct Implementation<Physics::relativistic_mhd, Solver::hlle> : relativistic_mhd::HLLE {};

inline constexpr bool DescriptorsFollowSolverOrder() {
  for (int index = 0; index < static_cast<int>(descriptors.size()); ++index) {
    if (static_cast<int>(descriptors[index].solver) != index)
      return false;
  }
  return descriptors.size() == static_cast<std::size_t>(Solver::none) + 1;
}
static_assert(DescriptorsFollowSolverOrder(),
              "Riemann descriptors must list every solver in enum order, ending with none");
static_assert(descriptors[static_cast<int>(first_order_fallback)].newtonian_hydro &&
                  descriptors[static_cast<int>(first_order_fallback)].newtonian_mhd &&
                  descriptors[static_cast<int>(first_order_fallback)].relativistic_hydro &&
                  descriptors[static_cast<int>(first_order_fallback)].relativistic_mhd,
              "the first-order fallback must support every physics");

inline constexpr const Descriptor& Describe(const Solver solver) {
  return descriptors[static_cast<int>(solver)];
}

inline constexpr std::optional<Solver> Parse(const std::string_view name) {
  if (name == "advect")
    return Solver::none;
  for (const auto& descriptor : descriptors) {
    if (descriptor.name == name)
      return descriptor.solver;
  }
  return std::nullopt;
}

namespace detail {

template <Physics P, Solver S, typename Result, typename Visitor>
Result VisitSupported(Visitor&& visitor) {
  if constexpr (Describe(S).Supports(P))
    return std::forward<Visitor>(visitor).template operator()<S>();
  else
    throw std::invalid_argument("Riemann solver does not support this physics");
}

// Walks the registered solvers in enum order, so dispatch needs no per-solver case list.
template <Physics P, int Index, typename Result, typename Visitor>
Result VisitFrom(const Solver solver, Visitor&& visitor) {
  if constexpr (Index == static_cast<int>(Solver::none))
    throw std::invalid_argument("kinematic Riemann mode has no solver to dispatch");
  else if (static_cast<int>(solver) == Index)
    return VisitSupported<P, static_cast<Solver>(Index), Result>(std::forward<Visitor>(visitor));
  else
    return VisitFrom<P, Index + 1, Result>(solver, std::forward<Visitor>(visitor));
}

} // namespace detail

// Calls `visitor.template operator()<S>()` for the selected solver. Solvers that do not declare
// support for `P` are never instantiated, and `none` must be handled before dispatch.
template <Physics P, typename Visitor>
auto Visit(const Solver solver, Visitor&& visitor)
    -> decltype(std::forward<Visitor>(visitor).template operator()<Solver::llf>()) {
  using Result = decltype(std::forward<Visitor>(visitor).template operator()<Solver::llf>());
  return detail::VisitFrom<P, 0, Result>(solver, std::forward<Visitor>(visitor));
}

} // namespace pangu::riemann

#endif
