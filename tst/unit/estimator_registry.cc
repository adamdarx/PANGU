#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "estimator/registry.h"

namespace {

using parthenon::Real;
namespace estimator = pangu::estimator;

// A user estimator outside the production tree: dt = min_d dx_d / max_d c_d.
struct SlowestCell {
  static constexpr std::string_view name = "slowest_cell";
  static constexpr bool characteristic_gr_speeds = true;
  static constexpr bool face_signal_speeds = false;

  KOKKOS_INLINE_FUNCTION static Real Start() { return 0.0; }
  KOKKOS_INLINE_FUNCTION static Real Add(const Real rate, const Real speed, const Real dx) {
    return fmax(rate, speed / dx);
  }
  KOKKOS_INLINE_FUNCTION static Real Finish(const Real rate) {
    return rate > 0.0 ? 1.0 / rate : std::numeric_limits<Real>::max();
  }
};

using TestRegistry = estimator::Registry<estimator::model::Light, estimator::model::Wave,
                                         SlowestCell>;

void Require(const bool condition, const std::string& label) {
  if (!condition)
    throw std::runtime_error(label);
}

// Evaluates a model in a device kernel, exactly as the fluid packages do.
template <estimator::Model Estimator>
Real CellTimestep(const Real speeds[3], const Real dx[3], const int ndim) {
  Kokkos::View<Real[3]> device_speeds("estimator speeds");
  Kokkos::View<Real[3]> device_dx("estimator spacings");
  auto host_speeds = Kokkos::create_mirror_view(device_speeds);
  auto host_dx = Kokkos::create_mirror_view(device_dx);
  for (int d = 0; d < 3; ++d) {
    host_speeds(d) = speeds[d];
    host_dx(d) = dx[d];
  }
  Kokkos::deep_copy(device_speeds, host_speeds);
  Kokkos::deep_copy(device_dx, host_dx);
  Real dt = 0.0;
  Kokkos::parallel_reduce(
      "estimator registry cell", 1,
      KOKKOS_LAMBDA(const int, Real& local) {
        Real value = Estimator::Start();
        for (int d = 0; d < ndim; ++d)
          value = Estimator::Add(value, device_speeds(d), device_dx(d));
        local = Estimator::Finish(value);
      },
      dt);
  return dt;
}

void CheckRegistry() {
  static_assert(estimator::Registered::size == 2);
  static_assert(TestRegistry::size == 3);
  static_assert(TestRegistry::Find("light") == 0);
  static_assert(TestRegistry::Find("wave") == 1);
  static_assert(TestRegistry::Find("slowest_cell") == 2);
  static_assert(TestRegistry::Find("missing") == TestRegistry::size);
  static_assert(std::is_same_v<TestRegistry::At<2>, SlowestCell>);
  static_assert(SlowestCell::characteristic_gr_speeds && !SlowestCell::face_signal_speeds);
  static_assert(!estimator::Model<int>);
}

void CheckFormulas() {
  constexpr Real speeds[3] = {0.5, 0.25, 0.8};
  constexpr Real dx[3] = {0.1, 0.05, 0.2};
  const Real light = CellTimestep<TestRegistry::At<0>>(speeds, dx, 3);
  const Real wave = CellTimestep<TestRegistry::At<1>>(speeds, dx, 3);
  const Real slowest = CellTimestep<TestRegistry::At<2>>(speeds, dx, 3);
  Require(light == fmin(fmin(dx[0] / speeds[0], dx[1] / speeds[1]), dx[2] / speeds[2]),
          "light directional minimum");
  Require(wave == 1.0 / (speeds[0] / dx[0] + speeds[1] / dx[1] + speeds[2] / dx[2]),
          "wave summed inverse");
  Require(slowest == 1.0 / fmax(fmax(speeds[0] / dx[0], speeds[1] / dx[1]), speeds[2] / dx[2]),
          "user model maximum rate");
  const Real rest = CellTimestep<TestRegistry::At<1>>(std::array<Real, 3>{}.data(), dx, 3);
  Require(rest == std::numeric_limits<Real>::max(), "wave without signal speeds");
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  try {
    CheckRegistry();
    CheckFormulas();
    std::cout << "Estimator registry and models PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Estimator registry and models FAIL: " << error.what() << '\n';
    return 1;
  }
}
