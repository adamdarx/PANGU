#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>

#include <Kokkos_Core.hpp>
#include <parthenon/parthenon.hpp>

#include "z4c/finite_difference.h"

namespace {

using Real = parthenon::Real;

struct Accessor {
  Kokkos::View<Real***> values;
  int k;
  int j;
  int i;

  KOKKOS_INLINE_FUNCTION Real operator()(const int dk, const int dj, const int di) const {
    return values(k + dk, j + dj, i + di);
  }
};

void ApplyOperators(const Kokkos::View<Real***>& input, const Kokkos::View<Real***>& output,
                    const int cells, const Real inverse_spacing) {
  constexpr int ghost = pangu::nr::fd::Stencil<6>::transport_radius;
  Kokkos::parallel_for(
      "PANGU_NR1_fused_operator_benchmark", cells * cells * cells, KOKKOS_LAMBDA(const int index) {
        const int i = index % cells;
        const int j = (index / cells) % cells;
        const int k = index / (cells * cells);
        const Accessor value{input, k + ghost, j + ghost, i + ghost};
        Real result = 0.0;
        for (int direction = 0; direction < 3; ++direction) {
          const Real speed = direction == 0 ? -0.75 : (direction == 1 ? 0.0 : 0.875);
          result += pangu::nr::fd::First<6>(direction, inverse_spacing, value);
          result += pangu::nr::fd::Second<6>(direction, inverse_spacing, value);
          result += pangu::nr::fd::AdvectiveFirst<6>(direction, inverse_spacing, speed, value);
          result += pangu::nr::fd::KreissOliger<6>(direction, inverse_spacing, value);
        }
        result += pangu::nr::fd::MixedSecond<6>(0, 1, inverse_spacing, inverse_spacing, value);
        result += pangu::nr::fd::MixedSecond<6>(0, 2, inverse_spacing, inverse_spacing, value);
        result += pangu::nr::fd::MixedSecond<6>(1, 2, inverse_spacing, inverse_spacing, value);
        output(k, j, i) = result;
      });
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    constexpr int cells = 128;
    constexpr int ghost = pangu::nr::fd::Stencil<6>::transport_radius;
    constexpr int warmup_iterations = 20;
    constexpr int timed_iterations = 20;
    constexpr int repetitions = 7;
    const int extent = cells + 2 * ghost;
    const Real spacing = 2.0 * std::numbers::pi_v<Real> / cells;
    const Real inverse_spacing = 1.0 / spacing;
    Kokkos::View<Real***> input("NR-1 benchmark input", extent, extent, extent);
    Kokkos::View<Real***> output("NR-1 benchmark output", cells, cells, cells);
    Kokkos::parallel_for(
        "NR-1 initialize benchmark", extent * extent * extent, KOKKOS_LAMBDA(const int index) {
          const int i = index % extent;
          const int j = (index / extent) % extent;
          const int k = index / (extent * extent);
          input(k, j, i) = sin((i + 2.0 * j + 3.0 * k) * spacing);
        });
    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
      ApplyOperators(input, output, cells, inverse_spacing);
    Kokkos::fence();

    std::array<double, repetitions> times{};
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      Kokkos::Timer timer;
      for (int iteration = 0; iteration < timed_iterations; ++iteration)
        ApplyOperators(input, output, cells, inverse_spacing);
      Kokkos::fence();
      times[repetition] = timer.seconds() / timed_iterations;
    }
    std::sort(times.begin(), times.end());
    const double median = times[repetitions / 2];
    const double lower_quartile = times[1];
    const double upper_quartile = times[5];
    const double cell_updates = static_cast<double>(cells) * cells * cells;
    const double giga_cell_updates = cell_updates / median / 1.0e9;
    const double giga_operator_values = 15.0 * cell_updates / median / 1.0e9;

    Real checksum = 0.0;
    Kokkos::parallel_reduce(
        "NR-1 benchmark checksum", cells * cells * cells,
        KOKKOS_LAMBDA(const int index, Real& update) {
          const int i = index % cells;
          const int j = (index / cells) % cells;
          const int k = index / (cells * cells);
          update += output(k, j, i);
        },
        checksum);
    std::cout << std::setprecision(10)
              << "NR-1 fused finite-difference benchmark: cells=128^3 warmup=" << warmup_iterations
              << " repetitions=" << repetitions << " iterations_per_repetition=" << timed_iterations
              << " median_seconds=" << median << " iqr_seconds=" << upper_quartile - lower_quartile
              << " Gcell_updates_per_second=" << giga_cell_updates
              << " Goperator_values_per_second=" << giga_operator_values << " checksum=" << checksum
              << '\n';
  }
  Kokkos::finalize();
  return 0;
}
