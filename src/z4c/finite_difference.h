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
  if constexpr (Order == 2) {
    return (-0.5 * DirectionalValue(value, direction, -1) +
            0.5 * DirectionalValue(value, direction, 1)) *
           inverse_spacing;
  } else if constexpr (Order == 4) {
    return ((1.0 / 12.0 * DirectionalValue(value, direction, -2) -
             1.0 / 12.0 * DirectionalValue(value, direction, 2)) +
            (-2.0 / 3.0 * DirectionalValue(value, direction, -1) +
             2.0 / 3.0 * DirectionalValue(value, direction, 1))) *
           inverse_spacing;
  } else {
    return ((-1.0 / 60.0 * DirectionalValue(value, direction, -3) +
             1.0 / 60.0 * DirectionalValue(value, direction, 3)) +
            (3.0 / 20.0 * DirectionalValue(value, direction, -2) -
             3.0 / 20.0 * DirectionalValue(value, direction, 2)) +
            (-3.0 / 4.0 * DirectionalValue(value, direction, -1) +
             3.0 / 4.0 * DirectionalValue(value, direction, 1))) *
           inverse_spacing;
  }
}

template <int Order, class Accessor>
KOKKOS_INLINE_FUNCTION parthenon::Real
Second(const int direction, const parthenon::Real inverse_spacing, const Accessor& value) {
  static_assert(kSupportedOrder<Order>);
  parthenon::Real result;
  if constexpr (Order == 2) {
    result = DirectionalValue(value, direction, -1) + DirectionalValue(value, direction, 1) -
             2.0 * value(0, 0, 0);
  } else if constexpr (Order == 4) {
    result = (-1.0 / 12.0 * DirectionalValue(value, direction, -2) -
              1.0 / 12.0 * DirectionalValue(value, direction, 2)) +
             (4.0 / 3.0 * DirectionalValue(value, direction, -1) +
              4.0 / 3.0 * DirectionalValue(value, direction, 1)) -
             5.0 / 2.0 * value(0, 0, 0);
  } else {
    result = (1.0 / 90.0 * DirectionalValue(value, direction, -3) +
              1.0 / 90.0 * DirectionalValue(value, direction, 3)) +
             (-3.0 / 20.0 * DirectionalValue(value, direction, -2) -
              3.0 / 20.0 * DirectionalValue(value, direction, 2)) +
             (3.0 / 2.0 * DirectionalValue(value, direction, -1) +
              3.0 / 2.0 * DirectionalValue(value, direction, 1)) -
             49.0 / 18.0 * value(0, 0, 0);
  }
  return result * inverse_spacing * inverse_spacing;
}

template <int Order, class Accessor>
KOKKOS_INLINE_FUNCTION parthenon::Real
MixedSecond(const int first_direction, const int second_direction,
            const parthenon::Real first_inverse_spacing,
            const parthenon::Real second_inverse_spacing, const Accessor& value) {
  static_assert(kSupportedOrder<Order>);
  constexpr int radius = Stencil<Order>::centered_radius;
  parthenon::Real result = 0.0;
  for (int first_offset = -radius; first_offset <= radius; ++first_offset) {
    const parthenon::Real first_weight = FirstCoefficient<Order>(first_offset);
    for (int second_offset = -radius; second_offset <= radius; ++second_offset) {
      const parthenon::Real second_weight = FirstCoefficient<Order>(second_offset);
      const int dk =
          first_offset * (first_direction == 2) + second_offset * (second_direction == 2);
      const int dj =
          first_offset * (first_direction == 1) + second_offset * (second_direction == 1);
      const int di =
          first_offset * (first_direction == 0) + second_offset * (second_direction == 0);
      result += first_weight * second_weight * value(dk, dj, di);
    }
  }
  return result * first_inverse_spacing * second_inverse_spacing;
}

template <int Order, class Accessor>
KOKKOS_INLINE_FUNCTION parthenon::Real
AdvectiveFirst(const int direction, const parthenon::Real inverse_spacing,
               const parthenon::Real velocity, const Accessor& value) {
  static_assert(kSupportedOrder<Order>);
  parthenon::Real left;
  parthenon::Real right;
  if constexpr (Order == 2) {
    left = 0.5 * DirectionalValue(value, direction, -2) -
           2.0 * DirectionalValue(value, direction, -1) + 1.5 * value(0, 0, 0);
    right = -0.5 * DirectionalValue(value, direction, 2) +
            2.0 * DirectionalValue(value, direction, 1) - 1.5 * value(0, 0, 0);
  } else if constexpr (Order == 4) {
    left = -1.0 / 12.0 * DirectionalValue(value, direction, -3) +
           6.0 / 12.0 * DirectionalValue(value, direction, -2) -
           18.0 / 12.0 * DirectionalValue(value, direction, -1) + 10.0 / 12.0 * value(0, 0, 0) +
           3.0 / 12.0 * DirectionalValue(value, direction, 1);
    right = 1.0 / 12.0 * DirectionalValue(value, direction, 3) -
            6.0 / 12.0 * DirectionalValue(value, direction, 2) +
            18.0 / 12.0 * DirectionalValue(value, direction, 1) - 10.0 / 12.0 * value(0, 0, 0) -
            3.0 / 12.0 * DirectionalValue(value, direction, -1);
  } else {
    left = 1.0 / 60.0 * DirectionalValue(value, direction, -4) -
           2.0 / 15.0 * DirectionalValue(value, direction, -3) +
           1.0 / 2.0 * DirectionalValue(value, direction, -2) -
           4.0 / 3.0 * DirectionalValue(value, direction, -1) + 7.0 / 12.0 * value(0, 0, 0) +
           2.0 / 5.0 * DirectionalValue(value, direction, 1) -
           1.0 / 30.0 * DirectionalValue(value, direction, 2);
    right = -1.0 / 60.0 * DirectionalValue(value, direction, 4) +
            2.0 / 15.0 * DirectionalValue(value, direction, 3) -
            1.0 / 2.0 * DirectionalValue(value, direction, 2) +
            4.0 / 3.0 * DirectionalValue(value, direction, 1) - 7.0 / 12.0 * value(0, 0, 0) -
            2.0 / 5.0 * DirectionalValue(value, direction, -1) +
            1.0 / 30.0 * DirectionalValue(value, direction, -2);
  }
  return velocity * (velocity < 0.0 ? left : right) * inverse_spacing;
}

template <int Order, class Accessor>
KOKKOS_INLINE_FUNCTION parthenon::Real
KreissOliger(const int direction, const parthenon::Real inverse_spacing, const Accessor& value) {
  static_assert(kSupportedOrder<Order>);
  parthenon::Real result;
  if constexpr (Order == 2) {
    result = DirectionalValue(value, direction, -2) + DirectionalValue(value, direction, 2) -
             4.0 * DirectionalValue(value, direction, -1) -
             4.0 * DirectionalValue(value, direction, 1) + 6.0 * value(0, 0, 0);
  } else if constexpr (Order == 4) {
    result = DirectionalValue(value, direction, -3) + DirectionalValue(value, direction, 3) -
             6.0 * DirectionalValue(value, direction, -2) -
             6.0 * DirectionalValue(value, direction, 2) +
             15.0 * DirectionalValue(value, direction, -1) +
             15.0 * DirectionalValue(value, direction, 1) - 20.0 * value(0, 0, 0);
  } else {
    result = DirectionalValue(value, direction, -4) + DirectionalValue(value, direction, 4) -
             8.0 * DirectionalValue(value, direction, -3) -
             8.0 * DirectionalValue(value, direction, 3) +
             28.0 * DirectionalValue(value, direction, -2) +
             28.0 * DirectionalValue(value, direction, 2) -
             56.0 * DirectionalValue(value, direction, -1) -
             56.0 * DirectionalValue(value, direction, 1) + 70.0 * value(0, 0, 0);
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
