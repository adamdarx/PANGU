#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <Kokkos_Core.hpp>

#include "electron/model/registry.h"

namespace {

using Real = double;
using pangu::electron::HeatingFraction;

constexpr int cases = 3;
constexpr int kinetic_models = 5;

struct TestModel {
  static constexpr const char* key = "test";
  static constexpr const char* label = "Kel_test";
  static constexpr bool requires_magnetic_field = false;

  template <typename T>
  KOKKOS_INLINE_FUNCTION static T Fraction(const pangu::electron::HeatingState<T>& state) {
    return state.constant_fraction + static_cast<T>(0.125);
  }
};

using TestRegistry = pangu::electron::model::Registry<TestModel>;

constexpr std::array<std::array<Real, kinetic_models>, cases> kharma_reference{{
    {0.11114012775155177, 0.10407524866677852, 0.26116917629021924, 0.445919544657038,
     0.11332460789480643},
    {0.9973400707488927, 0.8539247099221324, 0.299029033784546, 0.4626412976864275,
     0.10690716290089985},
    {0.007399463079642298, 0.03230792498167544, 0.2555887729251237, 0.49889741310865277,
     0.06692922938425871},
}};

Real CheckReferenceVectors() {
  Kokkos::View<Real[cases][kinetic_models]> fractions("electron heating fractions");
  Kokkos::parallel_for(
      "PANGU EH-3 KHARMA model vectors", cases, KOKKOS_LAMBDA(const int sample) {
        Real density = 0.0;
        Real internal = 0.0;
        Real magnetic_squared = 0.0;
        Real electron_entropy = 0.0;
        if (sample == 0) {
          density = 1.0;
          internal = 0.1;
          magnetic_squared = 0.02;
          electron_entropy = 0.01;
        } else if (sample == 1) {
          density = 0.2;
          internal = 0.02;
          magnetic_squared = 0.08;
          electron_entropy = 0.015;
        } else {
          density = 2.0;
          internal = 0.4;
          magnetic_squared = 0.01;
          electron_entropy = 0.005;
        }
        constexpr Real gamma = 4.0 / 3.0;
        constexpr Real gamma_e = 4.0 / 3.0;
        constexpr Real gamma_p = 5.0 / 3.0;
        for (int model = 0; model < kinetic_models; ++model) {
          fractions(sample, model) =
              HeatingFraction(model + 1, density, internal, magnetic_squared, electron_entropy,
                              gamma, gamma_e, gamma_p, 0.1);
        }
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fractions);
  Real maximum_relative = 0.0;
  for (int sample = 0; sample < cases; ++sample) {
    for (int model = 0; model < kinetic_models; ++model) {
      const Real actual = host(sample, model);
      const Real expected = kharma_reference[sample][model];
      const Real scale = fmax(fabs(expected), 1.0e-300);
      const Real relative = fabs(actual - expected) / scale;
      maximum_relative = fmax(maximum_relative, relative);
      if (!std::isfinite(actual) || relative > 1.0e-13) {
        throw std::runtime_error("KHARMA fraction mismatch in sample " + std::to_string(sample) +
                                 ", model " + std::to_string(model) +
                                 ": relative error=" + std::to_string(relative));
      }
    }
  }
  return maximum_relative;
}

void CheckFiniteLimits() {
  Kokkos::View<Real[4][kinetic_models]> fractions("electron heating limit fractions");
  Kokkos::parallel_for(
      "PANGU EH-3 finite model limits", 4, KOKKOS_LAMBDA(const int sample) {
        const Real density = sample == 0 ? 1.0e-30 : (sample == 1 ? 1.0 : 1.0e6);
        const Real internal = sample == 2 ? 1.0e12 : 1.0e-30;
        const Real magnetic_squared = sample == 1 ? 0.0 : (sample == 3 ? 1.0e100 : 1.0e-30);
        const Real electron_entropy = sample == 3 ? 1.0e100 : 1.0e-30;
        constexpr Real gamma = 4.0 / 3.0;
        constexpr Real gamma_e = 4.0 / 3.0;
        constexpr Real gamma_p = 5.0 / 3.0;
        for (int model = 0; model < kinetic_models; ++model) {
          fractions(sample, model) =
              HeatingFraction(model + 1, density, internal, magnetic_squared, electron_entropy,
                              gamma, gamma_e, gamma_p, 0.1);
        }
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fractions);
  for (int sample = 0; sample < 4; ++sample) {
    for (int model = 0; model < kinetic_models; ++model) {
      const Real fraction = host(sample, model);
      if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0)
        throw std::runtime_error("non-finite or out-of-range kinetic heating fraction");
    }
  }
}

void CheckSinglePointExtension() {
  static_assert(TestRegistry::size == 1);
  static_assert(!TestRegistry::magnetic_requirements[0]);
  if (std::string_view(TestRegistry::keys[0]) != "test" ||
      std::string_view(TestRegistry::labels[0]) != "Kel_test")
    throw std::runtime_error("single-point model metadata registration failed");
  Kokkos::View<Real> result("single-point model result");
  Kokkos::parallel_for(
      "PANGU electron model extension", 1, KOKKOS_LAMBDA(const int) {
        const pangu::electron::HeatingState<Real> state{1.0, 0.1, 0.0, 0.01,
                                                        4.0 / 3.0, 4.0 / 3.0,
                                                        5.0 / 3.0, 0.25};
        result() = TestRegistry::Fraction(0, state);
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  if (fabs(host() - 0.375) > 1.0e-15)
    throw std::runtime_error("single-point model registry extension failed");
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  try {
    const Real maximum_relative = CheckReferenceVectors();
    CheckFiniteLimits();
    CheckSinglePointExtension();
    std::cout << "Electron kinetic heating prescriptions PASS: max KHARMA relative error="
              << std::scientific << std::setprecision(17) << maximum_relative << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Electron kinetic heating prescriptions FAIL: " << error.what() << '\n';
    return 1;
  }
}
