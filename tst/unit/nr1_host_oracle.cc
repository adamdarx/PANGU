#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <parthenon/parthenon.hpp>

#include "z4c/core/finite_difference.h"

namespace {

using Real = parthenon::Real;
constexpr int kCells = 9;
constexpr int kCenter = kCells / 2;

struct HostAccessor {
  const std::array<Real, kCells * kCells * kCells>& values;

  Real operator()(const int dk, const int dj, const int di) const {
    const int k = kCenter + dk;
    const int j = kCenter + dj;
    const int i = kCenter + di;
    return values[(k * kCells + j) * kCells + i];
  }
};

long double Factorial(const int value) {
  long double result = 1.0;
  for (int factor = 2; factor <= value; ++factor)
    result *= factor;
  return result;
}

std::vector<long double> DerivativeWeights(const std::vector<int>& nodes, const int derivative) {
  const int count = static_cast<int>(nodes.size());
  std::vector<std::vector<long double>> augmented(count, std::vector<long double>(count + 1, 0.0));
  for (int power = 0; power < count; ++power) {
    for (int node = 0; node < count; ++node)
      augmented[power][node] = std::pow(static_cast<long double>(nodes[node]), power);
    augmented[power][count] = power == derivative ? Factorial(derivative) : 0.0;
  }
  for (int pivot = 0; pivot < count; ++pivot) {
    int largest = pivot;
    for (int row = pivot + 1; row < count; ++row) {
      if (std::abs(augmented[row][pivot]) > std::abs(augmented[largest][pivot]))
        largest = row;
    }
    std::swap(augmented[pivot], augmented[largest]);
    const long double divisor = augmented[pivot][pivot];
    for (int column = pivot; column <= count; ++column)
      augmented[pivot][column] /= divisor;
    for (int row = 0; row < count; ++row) {
      if (row == pivot)
        continue;
      const long double multiplier = augmented[row][pivot];
      for (int column = pivot; column <= count; ++column)
        augmented[row][column] -= multiplier * augmented[pivot][column];
    }
  }
  std::vector<long double> weights(count);
  for (int node = 0; node < count; ++node)
    weights[node] = augmented[node][count];
  return weights;
}

long double DirectionalOracle(const HostAccessor& field, const int direction,
                              const std::vector<int>& nodes,
                              const std::vector<long double>& weights) {
  long double result = 0.0;
  for (std::size_t node = 0; node < nodes.size(); ++node) {
    const int offset = nodes[node];
    result += weights[node] * field(offset * (direction == 2), offset * (direction == 1),
                                    offset * (direction == 0));
  }
  return result;
}

long double MixedOracle(const HostAccessor& field, const int first_direction,
                        const int second_direction, const std::vector<int>& nodes,
                        const std::vector<long double>& weights) {
  long double result = 0.0;
  for (std::size_t first = 0; first < nodes.size(); ++first) {
    for (std::size_t second = 0; second < nodes.size(); ++second) {
      const int dk =
          nodes[first] * (first_direction == 2) + nodes[second] * (second_direction == 2);
      const int dj =
          nodes[first] * (first_direction == 1) + nodes[second] * (second_direction == 1);
      const int di =
          nodes[first] * (first_direction == 0) + nodes[second] * (second_direction == 0);
      result += weights[first] * weights[second] * field(dk, dj, di);
    }
  }
  return result;
}

long double Binomial(const int count, int selected) {
  selected = std::min(selected, count - selected);
  long double result = 1.0;
  for (int index = 1; index <= selected; ++index)
    result = result * (count - selected + index) / index;
  return result;
}

template <int Order> Real CompareOrder(const HostAccessor& field) {
  constexpr int radius = pangu::nr::fd::Stencil<Order>::centered_radius;
  constexpr int transport_radius = pangu::nr::fd::Stencil<Order>::transport_radius;
  const Real inverse_spacing[3]{1.3, 0.7, 2.1};
  std::vector<int> centered;
  for (int offset = -radius; offset <= radius; ++offset)
    centered.push_back(offset);
  const auto first_weights = DerivativeWeights(centered, 1);
  const auto second_weights = DerivativeWeights(centered, 2);
  std::vector<int> negative;
  std::vector<int> nonnegative;
  for (int offset = -transport_radius; offset <= radius - 1; ++offset)
    negative.push_back(offset);
  for (int offset = -(radius - 1); offset <= transport_radius; ++offset)
    nonnegative.push_back(offset);
  const auto negative_weights = DerivativeWeights(negative, 1);
  const auto nonnegative_weights = DerivativeWeights(nonnegative, 1);

  Real maximum_scaled = 0.0;
  for (int direction = 0; direction < 3; ++direction) {
    const auto compare = [&](const Real actual, const long double oracle) {
      const Real expected = static_cast<Real>(oracle);
      maximum_scaled = std::max(maximum_scaled, std::abs(actual - expected) /
                                                    std::max(Real{1.0}, std::abs(expected)));
    };
    compare(pangu::nr::fd::First<Order>(direction, inverse_spacing[direction], field),
            DirectionalOracle(field, direction, centered, first_weights) *
                inverse_spacing[direction]);
    compare(pangu::nr::fd::Second<Order>(direction, inverse_spacing[direction], field),
            DirectionalOracle(field, direction, centered, second_weights) *
                inverse_spacing[direction] * inverse_spacing[direction]);
    compare(
        pangu::nr::fd::AdvectiveFirst<Order>(direction, inverse_spacing[direction], -0.75, field),
        -0.75 * DirectionalOracle(field, direction, negative, negative_weights) *
            inverse_spacing[direction]);
    compare(pangu::nr::fd::AdvectiveFirst<Order>(direction, inverse_spacing[direction], 0.0, field),
            0.0 * DirectionalOracle(field, direction, nonnegative, nonnegative_weights) *
                inverse_spacing[direction]);
    compare(
        pangu::nr::fd::AdvectiveFirst<Order>(direction, inverse_spacing[direction], 0.875, field),
        0.875 * DirectionalOracle(field, direction, nonnegative, nonnegative_weights) *
            inverse_spacing[direction]);

    long double dissipation = 0.0;
    for (int offset = -transport_radius; offset <= transport_radius; ++offset) {
      const int binomial_index = offset + transport_radius;
      const long double sign = binomial_index % 2 == 0 ? 1.0 : -1.0;
      dissipation +=
          sign * Binomial(2 * transport_radius, binomial_index) *
          field(offset * (direction == 2), offset * (direction == 1), offset * (direction == 0));
    }
    compare(pangu::nr::fd::KreissOliger<Order>(direction, inverse_spacing[direction], field),
            dissipation * inverse_spacing[direction]);
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = first + 1; second < 3; ++second) {
      const Real actual = pangu::nr::fd::MixedSecond<Order>(first, second, inverse_spacing[first],
                                                            inverse_spacing[second], field);
      const Real expected =
          static_cast<Real>(MixedOracle(field, first, second, centered, first_weights) *
                            inverse_spacing[first] * inverse_spacing[second]);
      maximum_scaled = std::max(maximum_scaled, std::abs(actual - expected) /
                                                    std::max(Real{1.0}, std::abs(expected)));
    }
  }
  return maximum_scaled;
}

} // namespace

int main() {
  std::array<Real, kCells * kCells * kCells> values{};
  std::mt19937_64 generator(0x484f53544f524331ULL);
  std::uniform_real_distribution<Real> distribution(-1.0, 1.0);
  for (auto& value : values)
    value = distribution(generator);
  const HostAccessor field{values};
  const Real order2 = CompareOrder<2>(field);
  const Real order4 = CompareOrder<4>(field);
  const Real order6 = CompareOrder<6>(field);
  const Real maximum = std::max({order2, order4, order6});
  const Real tolerance = 256.0 * std::numeric_limits<Real>::epsilon();
  if (!std::isfinite(maximum) || maximum > tolerance) {
    std::cerr << "NR-1 independent host oracle failure: order2=" << order2 << " order4=" << order4
              << " order6=" << order6 << " tolerance=" << tolerance << '\n';
    return 1;
  }
  std::cout << "NR-1 independent host oracle PASS: order2=" << order2 << " order4=" << order4
            << " order6=" << order6 << '\n';
  return 0;
}
