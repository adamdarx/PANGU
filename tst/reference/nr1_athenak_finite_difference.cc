#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>

#include <Kokkos_Core.hpp>
#include <parthenon/parthenon.hpp>

using Real = parthenon::Real;

#include "athenak_finite_difference_reference.h"

#include "z4c/core/finite_difference.h"

namespace {

struct ScalarField {
  Kokkos::View<Real****> values;

  KOKKOS_INLINE_FUNCTION Real& operator()(const int block, const int k, const int j,
                                          const int i) const {
    return values(block, k, j, i);
  }
};

struct VectorField {
  Kokkos::View<Real*****> values;

  KOKKOS_INLINE_FUNCTION Real& operator()(const int block, const int component, const int k,
                                          const int j, const int i) const {
    return values(block, component, k, j, i);
  }
};

struct Accessor {
  ScalarField field;
  int k;
  int j;
  int i;

  KOKKOS_INLINE_FUNCTION Real operator()(const int dk, const int dj, const int di) const {
    return field(0, k + dk, j + dj, i + di);
  }
};

std::uint64_t OrderedBits(const Real value) {
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
  return (bits & (std::uint64_t{1} << 63)) != 0 ? ~bits + 1 : bits | (std::uint64_t{1} << 63);
}

template <int Order>
void Evaluate(const ScalarField field, const VectorField vector, const VectorField velocity,
              const Kokkos::View<Real***>& results, const int order_index) {
  constexpr int kCenter = 4;
  Kokkos::parallel_for(
      "NR-1 PANGU AthenaK operator comparison", 1, KOKKOS_LAMBDA(const int) {
        Real inverse_spacing[3]{1.3, 0.7, 2.1};
        const Accessor value{field, kCenter, kCenter, kCenter};
        for (int direction = 0; direction < 3; ++direction) {
          results(order_index, direction, 0) =
              pangu::nr::fd::First<Order>(direction, inverse_spacing[direction], value);
          results(order_index, direction, 1) =
              Dx<pangu::nr::fd::Stencil<Order>::required_ghost_zones>(
                  direction, inverse_spacing, field, 0, kCenter, kCenter, kCenter);
          results(order_index, 3 + direction, 0) =
              pangu::nr::fd::Second<Order>(direction, inverse_spacing[direction], value);
          results(order_index, 3 + direction, 1) =
              Dxx<pangu::nr::fd::Stencil<Order>::required_ghost_zones>(
                  direction, inverse_spacing, field, 0, kCenter, kCenter, kCenter);
          const Real speed = velocity(0, direction, kCenter, kCenter, kCenter);
          results(order_index, 9 + direction, 0) = pangu::nr::fd::AdvectiveFirst<Order>(
              direction, inverse_spacing[direction], speed, value);
          results(order_index, 9 + direction, 1) =
              Lx<pangu::nr::fd::Stencil<Order>::required_ghost_zones>(direction, inverse_spacing,
                                                                      velocity, field, 0, direction,
                                                                      kCenter, kCenter, kCenter);
          results(order_index, 12 + direction, 0) =
              pangu::nr::fd::KreissOliger<Order>(direction, inverse_spacing[direction], value);
          results(order_index, 12 + direction, 1) =
              Diss<pangu::nr::fd::Stencil<Order>::required_ghost_zones>(
                  direction, inverse_spacing, vector, 0, 0, kCenter, kCenter, kCenter);
        }
        int mixed = 6;
        for (int first = 0; first < 3; ++first) {
          for (int second = first + 1; second < 3; ++second) {
            results(order_index, mixed, 0) = pangu::nr::fd::MixedSecond<Order>(
                first, second, inverse_spacing[first], inverse_spacing[second], value);
            results(order_index, mixed, 1) =
                Dxy<pangu::nr::fd::Stencil<Order>::required_ghost_zones>(
                    first, second, inverse_spacing, field, 0, kCenter, kCenter, kCenter);
            ++mixed;
          }
        }
      });
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int return_code = 0;
  {
    constexpr int kCells = 9;
    constexpr int kOrders = 3;
    constexpr int kOperators = 15;
    ScalarField scalar{
        Kokkos::View<Real****>("NR-1 scalar random field", 1, kCells, kCells, kCells)};
    VectorField vector{
        Kokkos::View<Real*****>("NR-1 vector random field", 1, 1, kCells, kCells, kCells)};
    VectorField velocity{Kokkos::View<Real*****>("NR-1 velocity", 1, 3, kCells, kCells, kCells)};
    auto scalar_host = Kokkos::create_mirror_view(scalar.values);
    auto vector_host = Kokkos::create_mirror_view(vector.values);
    auto velocity_host = Kokkos::create_mirror_view(velocity.values);
    std::mt19937_64 generator(0x415448454e414b31ULL);
    std::uniform_real_distribution<Real> distribution(-1.0, 1.0);
    for (int k = 0; k < kCells; ++k) {
      for (int j = 0; j < kCells; ++j) {
        for (int i = 0; i < kCells; ++i) {
          const Real value = distribution(generator);
          scalar_host(0, k, j, i) = value;
          vector_host(0, 0, k, j, i) = value;
          velocity_host(0, 0, k, j, i) = -0.75;
          velocity_host(0, 1, k, j, i) = 0.875;
          velocity_host(0, 2, k, j, i) = 0.0;
        }
      }
    }
    Kokkos::deep_copy(scalar.values, scalar_host);
    Kokkos::deep_copy(vector.values, vector_host);
    Kokkos::deep_copy(velocity.values, velocity_host);
    Kokkos::View<Real***> results("NR-1 cross-code operator results", kOrders, kOperators, 2);
    Evaluate<2>(scalar, vector, velocity, results, 0);
    Evaluate<4>(scalar, vector, velocity, results, 1);
    Evaluate<6>(scalar, vector, velocity, results, 2);
    Kokkos::fence();
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
    Real maximum_absolute = 0.0;
    Real maximum_scaled = 0.0;
    std::uint64_t maximum_ulp = 0;
    for (int order = 0; order < kOrders; ++order) {
      for (int operation = 0; operation < kOperators; ++operation) {
        const Real pangu = host(order, operation, 0);
        const Real athenak = host(order, operation, 1);
        const Real absolute = std::abs(pangu - athenak);
        maximum_absolute = std::max(maximum_absolute, absolute);
        maximum_scaled =
            std::max(maximum_scaled, absolute / std::max(Real{1.0}, std::abs(athenak)));
        const std::uint64_t pangu_bits = OrderedBits(pangu);
        const std::uint64_t athenak_bits = OrderedBits(athenak);
        maximum_ulp = std::max(maximum_ulp, pangu_bits > athenak_bits ? pangu_bits - athenak_bits
                                                                      : athenak_bits - pangu_bits);
      }
    }
    const Real tolerance = 128.0 * std::numeric_limits<Real>::epsilon();
    if (!std::isfinite(maximum_scaled) || maximum_scaled > tolerance) {
      std::cerr << "NR-1 AthenaK comparison failure: absolute=" << maximum_absolute
                << " scaled=" << maximum_scaled << " ulp=" << maximum_ulp
                << " tolerance=" << tolerance << '\n';
      return_code = 1;
    } else {
      std::cout << "NR-1 AthenaK comparison PASS: absolute=" << maximum_absolute
                << " scaled=" << maximum_scaled << " ulp=" << maximum_ulp << '\n';
    }
  }
  Kokkos::finalize();
  return return_code;
}
