#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>

#include <Kokkos_Core.hpp>
#include <parthenon/parthenon.hpp>

#include "z4c/core/finite_difference.h"

namespace {

using Real = parthenon::Real;
constexpr int kMetrics = 4;

template <class View> struct Accessor {
  View values;
  int k;
  int j;
  int i;

  KOKKOS_INLINE_FUNCTION Real operator()(const int dk, const int dj, const int di) const {
    return values(k + dk, j + dj, i + di);
  }
};

struct ErrorNorms {
  Real first;
  Real second;
  Real mixed;
  Real advective;
};

template <int Order> ErrorNorms Measure(const int cells) {
  constexpr int ghost = pangu::nr::fd::Stencil<Order>::transport_radius;
  const int extent = cells + 2 * ghost;
  const Real spacing = 2.0 * std::numbers::pi_v<Real> / cells;
  const Real inverse_spacing = 1.0 / spacing;
  Kokkos::View<Real***> field("NR-1 convergence field", extent, extent, extent);
  Kokkos::parallel_for(
      "NR-1 initialize convergence field",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {extent, extent, extent}),
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = (i - ghost) * spacing;
        const Real y = (j - ghost) * spacing;
        const Real z = (k - ghost) * spacing;
        field(k, j, i) = sin(x + 2.0 * y + 3.0 * z);
      });
  Kokkos::View<Real****> squared("NR-1 squared convergence errors", kMetrics, cells, cells, cells);
  Kokkos::parallel_for(
      "NR-1 evaluate convergence operators",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {cells, cells, cells}),
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int kk = k + ghost;
        const int jj = j + ghost;
        const int ii = i + ghost;
        const Real phase = i * spacing + 2.0 * j * spacing + 3.0 * k * spacing;
        const Real sine = sin(phase);
        const Real cosine = cos(phase);
        const Real wave_number[3]{1.0, 2.0, 3.0};
        const Accessor<decltype(field)> value{field, kk, jj, ii};
        Real first_error = 0.0;
        Real second_error = 0.0;
        Real advective_error = 0.0;
        for (int direction = 0; direction < 3; ++direction) {
          const Real exact_first = wave_number[direction] * cosine;
          const Real exact_second = -wave_number[direction] * wave_number[direction] * sine;
          const Real first =
              pangu::nr::fd::First<Order>(direction, inverse_spacing, value) - exact_first;
          const Real second =
              pangu::nr::fd::Second<Order>(direction, inverse_spacing, value) - exact_second;
          const Real negative =
              pangu::nr::fd::AdvectiveFirst<Order>(direction, inverse_spacing, -0.7, value) +
              0.7 * exact_first;
          const Real positive =
              pangu::nr::fd::AdvectiveFirst<Order>(direction, inverse_spacing, 0.9, value) -
              0.9 * exact_first;
          first_error += first * first;
          second_error += second * second;
          advective_error += negative * negative + positive * positive;
        }
        Real mixed_error = 0.0;
        for (int first_direction = 0; first_direction < 3; ++first_direction) {
          for (int second_direction = first_direction + 1; second_direction < 3;
               ++second_direction) {
            const Real exact = -wave_number[first_direction] * wave_number[second_direction] * sine;
            const Real error =
                pangu::nr::fd::MixedSecond<Order>(first_direction, second_direction,
                                                  inverse_spacing, inverse_spacing, value) -
                exact;
            mixed_error += error * error;
          }
        }
        squared(0, k, j, i) = first_error / 3.0;
        squared(1, k, j, i) = second_error / 3.0;
        squared(2, k, j, i) = mixed_error / 3.0;
        squared(3, k, j, i) = advective_error / 6.0;
      });
  Kokkos::fence();

  const int total_cells = cells * cells * cells;
  std::array<Real, kMetrics> rms{};
  for (int metric = 0; metric < kMetrics; ++metric) {
    Real sum = 0.0;
    Kokkos::parallel_reduce(
        "NR-1 reduce convergence error", total_cells,
        KOKKOS_LAMBDA(const int index, Real& update) {
          const int i = index % cells;
          const int j = (index / cells) % cells;
          const int k = index / (cells * cells);
          update += squared(metric, k, j, i);
        },
        sum);
    rms[metric] = sqrt(sum / total_cells);
  }
  return {rms[0], rms[1], rms[2], rms[3]};
}

template <std::size_t Size>
Real FitOrder(const std::array<int, Size>& cells, const std::array<Real, Size>& errors) {
  Real mean_x = 0.0;
  Real mean_y = 0.0;
  for (std::size_t index = 0; index < Size; ++index) {
    mean_x += log(1.0 / cells[index]);
    mean_y += log(errors[index]);
  }
  mean_x /= Size;
  mean_y /= Size;
  Real numerator = 0.0;
  Real denominator = 0.0;
  for (std::size_t index = 0; index < Size; ++index) {
    const Real delta_x = log(1.0 / cells[index]) - mean_x;
    numerator += delta_x * (log(errors[index]) - mean_y);
    denominator += delta_x * delta_x;
  }
  return numerator / denominator;
}

template <int Order> bool CheckConvergence() {
  constexpr std::array<int, 4> cells{16, 32, 64, 128};
  std::array<Real, cells.size()> first{};
  std::array<Real, cells.size()> second{};
  std::array<Real, cells.size()> mixed{};
  std::array<Real, cells.size()> advective{};
  for (std::size_t index = 0; index < cells.size(); ++index) {
    const ErrorNorms errors = Measure<Order>(cells[index]);
    first[index] = errors.first;
    second[index] = errors.second;
    mixed[index] = errors.mixed;
    advective[index] = errors.advective;
  }
  const Real first_order = FitOrder(cells, first);
  const Real second_order = FitOrder(cells, second);
  const Real mixed_order = FitOrder(cells, mixed);
  const Real advective_order = FitOrder(cells, advective);
  const Real minimum = std::min({first_order, second_order, mixed_order, advective_order});
  std::cout << std::setprecision(10) << "NR-1 convergence order=" << Order
            << " first=" << first_order << " second=" << second_order << " mixed=" << mixed_order
            << " advective=" << advective_order << " error128={" << first.back() << ','
            << second.back() << ',' << mixed.back() << ',' << advective.back() << "}\n";
  return std::isfinite(minimum) && minimum >= Order - 0.25;
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int return_code = 0;
  {
    const bool second = CheckConvergence<2>();
    const bool fourth = CheckConvergence<4>();
    const bool sixth = CheckConvergence<6>();
    if (!(second && fourth && sixth)) {
      std::cerr << "NR-1 finite-difference convergence failure\n";
      return_code = 1;
    }
  }
  Kokkos::finalize();
  return return_code;
}
