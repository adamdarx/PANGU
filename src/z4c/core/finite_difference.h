#ifndef PANGU_Z4C_FINITE_DIFFERENCE_H_
#define PANGU_Z4C_FINITE_DIFFERENCE_H_

#include <parthenon/parthenon.hpp>

namespace pangu::nr::fd {

template <int Order> struct Stencil;

template <> struct Stencil<2> {
  static constexpr int centered_radius = 1;
  static constexpr int transport_radius = 2;
  static constexpr int required_ghost_zones = 2;
};

template <> struct Stencil<4> {
  static constexpr int centered_radius = 2;
  static constexpr int transport_radius = 3;
  static constexpr int required_ghost_zones = 3;
};

template <> struct Stencil<6> {
  static constexpr int centered_radius = 3;
  static constexpr int transport_radius = 4;
  static constexpr int required_ghost_zones = 4;
};

template <int Order> inline constexpr bool kSupportedOrder = Order == 2 || Order == 4 || Order == 6;

constexpr int RequiredGhostZones(const int order) {
  if (order == 2)
    return Stencil<2>::required_ghost_zones;
  if (order == 4)
    return Stencil<4>::required_ghost_zones;
  if (order == 6)
    return Stencil<6>::required_ghost_zones;
  return -1;
}

template <class Accessor>
KOKKOS_INLINE_FUNCTION parthenon::Real DirectionalValue(const Accessor& value, const int direction,
                                                        const int offset) {
  return value(offset * (direction == 2), offset * (direction == 1), offset * (direction == 0));
}

template <int Order>
KOKKOS_INLINE_FUNCTION constexpr parthenon::Real FirstCoefficient(const int offset) {
  static_assert(kSupportedOrder<Order>);
  if constexpr (Order == 2) {
    return offset == -1 ? -1.0 / 2.0 : (offset == 1 ? 1.0 / 2.0 : 0.0);
  } else if constexpr (Order == 4) {
    if (offset == -2)
      return 1.0 / 12.0;
    if (offset == -1)
      return -2.0 / 3.0;
    if (offset == 1)
      return 2.0 / 3.0;
    return offset == 2 ? -1.0 / 12.0 : 0.0;
  } else {
    if (offset == -3)
      return -1.0 / 60.0;
    if (offset == -2)
      return 3.0 / 20.0;
    if (offset == -1)
      return -3.0 / 4.0;
    if (offset == 1)
      return 3.0 / 4.0;
    if (offset == 2)
      return -3.0 / 20.0;
    return offset == 3 ? 1.0 / 60.0 : 0.0;
  }
}

template <int Order, class Accessor>
KOKKOS_INLINE_FUNCTION parthenon::Real
First(const int direction, const parthenon::Real inverse_spacing, const Accessor& value) {
  static_assert(kSupportedOrder<Order>);
  // Subtract equal-background samples BEFORE applying non-binary weights.
  // This preserves a constant exactly, including with CUDA FMA contraction.
  const auto d1 = DirectionalValue(value, direction, 1) -
                  DirectionalValue(value, direction, -1);
  if constexpr (Order == 2) {
    return 0.5 * d1 * inverse_spacing;
  } else if constexpr (Order == 4) {
    const auto d2 = DirectionalValue(value, direction, 2) -
                    DirectionalValue(value, direction, -2);
    return (8.0 * d1 - d2) * (inverse_spacing / 12.0);
  } else {
    const auto d2 = DirectionalValue(value, direction, 2) -
                    DirectionalValue(value, direction, -2);
    const auto d3 = DirectionalValue(value, direction, 3) -
                    DirectionalValue(value, direction, -3);
    return (45.0 * d1 - 9.0 * d2 + d3) * (inverse_spacing / 60.0);
  }
}

template <class Accessor>
KOKKOS_INLINE_FUNCTION parthenon::Real
CenteredPair(const Accessor& value, const int direction, const int offset,
             const parthenon::Real center) {
  return (DirectionalValue(value, direction, -offset) - center) +
         (DirectionalValue(value, direction, offset) - center);
}

template <int Order, class Accessor>
KOKKOS_INLINE_FUNCTION parthenon::Real
Second(const int direction, const parthenon::Real inverse_spacing, const Accessor& value) {
  static_assert(kSupportedOrder<Order>);
  const auto center = value(0, 0, 0);
  const auto d1 = CenteredPair(value, direction, 1, center);
  if constexpr (Order == 2) {
    return d1 * inverse_spacing * inverse_spacing;
  } else if constexpr (Order == 4) {
    const auto d2 = CenteredPair(value, direction, 2, center);
    return (16.0 * d1 - d2) * (inverse_spacing * inverse_spacing / 12.0);
  } else {
    const auto d2 = CenteredPair(value, direction, 2, center);
    const auto d3 = CenteredPair(value, direction, 3, center);
    return (270.0 * d1 - 27.0 * d2 + 2.0 * d3) *
           (inverse_spacing * inverse_spacing / 180.0);
  }
}

template <int Order, class Accessor>
KOKKOS_INLINE_FUNCTION parthenon::Real
MixedSecond(const int first_direction, const int second_direction,
            const parthenon::Real first_inverse_spacing,
            const parthenon::Real second_inverse_spacing, const Accessor& value) {
  static_assert(kSupportedOrder<Order>);
  constexpr int radius = Stencil<Order>::centered_radius;
  parthenon::Real result = 0.0;
  // Tensor-product antisymmetry makes constants, and fields independent of
  // either differentiation direction, vanish exactly. The old weighted sum
  // of O(1) samples left an O(epsilon / dx^2) source even in flat spacetime.
  for (int first_offset = 1; first_offset <= radius; ++first_offset) {
    const parthenon::Real first_weight = FirstCoefficient<Order>(first_offset);
    for (int second_offset = 1; second_offset <= radius; ++second_offset) {
      const parthenon::Real second_weight = FirstCoefficient<Order>(second_offset);
      const int fk = first_offset * (first_direction == 2);
      const int fj = first_offset * (first_direction == 1);
      const int fi = first_offset * (first_direction == 0);
      const int sk = second_offset * (second_direction == 2);
      const int sj = second_offset * (second_direction == 1);
      const int si = second_offset * (second_direction == 0);
      const auto cross = (value(fk + sk, fj + sj, fi + si) -
                          value(-fk + sk, -fj + sj, -fi + si)) -
                         (value(fk - sk, fj - sj, fi - si) -
                          value(-fk - sk, -fj - sj, -fi - si));
      result += first_weight * second_weight * cross;
    }
  }
  return result * first_inverse_spacing * second_inverse_spacing;
}

template <int Order, class Accessor>
KOKKOS_INLINE_FUNCTION parthenon::Real
AdvectiveFirst(const int direction, const parthenon::Real inverse_spacing,
               const parthenon::Real velocity, const Accessor& value) {
  static_assert(kSupportedOrder<Order>);
  if (velocity == 0.0) return 0.0;
  const auto center = value(0, 0, 0);
  const int orientation = velocity < 0.0 ? 1 : -1;
  const auto deviation = [&](const int offset) {
    return DirectionalValue(value, direction, orientation * offset) - center;
  };
  parthenon::Real result;
  if constexpr (Order == 2) {
    result = (deviation(-2) - 4.0 * deviation(-1)) / 2.0;
  } else if constexpr (Order == 4) {
    result = (-deviation(-3) + 6.0 * deviation(-2) - 18.0 * deviation(-1) +
              3.0 * deviation(1)) / 12.0;
  } else {
    result = (deviation(-4) - 8.0 * deviation(-3) + 30.0 * deviation(-2) -
              80.0 * deviation(-1) + 24.0 * deviation(1) - 2.0 * deviation(2)) / 60.0;
  }
  return velocity * orientation * result * inverse_spacing;
}

template <int Order, class Accessor>
KOKKOS_INLINE_FUNCTION parthenon::Real
KreissOliger(const int direction, const parthenon::Real inverse_spacing, const Accessor& value) {
  static_assert(kSupportedOrder<Order>);
  const auto center = value(0, 0, 0);
  const auto d1 = CenteredPair(value, direction, 1, center);
  const auto d2 = CenteredPair(value, direction, 2, center);
  parthenon::Real result;
  if constexpr (Order == 2) {
    result = d2 - 4.0 * d1;
  } else if constexpr (Order == 4) {
    const auto d3 = CenteredPair(value, direction, 3, center);
    result = d3 - 6.0 * d2 + 15.0 * d1;
  } else {
    const auto d3 = CenteredPair(value, direction, 3, center);
    const auto d4 = CenteredPair(value, direction, 4, center);
    result = d4 - 8.0 * d3 + 28.0 * d2 - 56.0 * d1;
  }
  return result * inverse_spacing;
}

static_assert(Stencil<2>::required_ghost_zones == 2);
static_assert(Stencil<4>::required_ghost_zones == 3);
static_assert(Stencil<6>::required_ghost_zones == 4);
static_assert(Stencil<2>::transport_radius == Stencil<2>::centered_radius + 1);
static_assert(Stencil<4>::transport_radius == Stencil<4>::centered_radius + 1);
static_assert(Stencil<6>::transport_radius == Stencil<6>::centered_radius + 1);

} // namespace pangu::nr::fd

#endif
