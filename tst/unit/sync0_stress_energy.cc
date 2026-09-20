#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "z4c/constraints.h"
#include "z4c/stress_energy.h"
#include "z4c/z4c_rhs.h"

namespace {
using Real = parthenon::Real;
using pangu::nr::Index;
using pangu::nr::SpatialSymmetricComponent;
using pangu::nr::StressEnergyComponent;
using pangu::nr::Z4cComponent;

Real ScaledDifference(const Real measured, const Real expected) {
  return std::abs(measured - expected) / std::max({1.0, std::abs(measured), std::abs(expected)});
}

void InvertSymmetric(const Real metric[6], Real inverse[6]) {
  const Real determinant = metric[0] * (metric[3] * metric[5] - metric[4] * metric[4]) -
                           metric[1] * (metric[1] * metric[5] - metric[2] * metric[4]) +
                           metric[2] * (metric[1] * metric[4] - metric[2] * metric[3]);
  inverse[0] = (metric[3] * metric[5] - metric[4] * metric[4]) / determinant;
  inverse[1] = (metric[2] * metric[4] - metric[1] * metric[5]) / determinant;
  inverse[2] = (metric[1] * metric[4] - metric[2] * metric[3]) / determinant;
  inverse[3] = (metric[0] * metric[5] - metric[2] * metric[2]) / determinant;
  inverse[4] = (metric[1] * metric[2] - metric[0] * metric[4]) / determinant;
  inverse[5] = (metric[0] * metric[3] - metric[1] * metric[1]) / determinant;
}
} // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  constexpr int cases = pangu::nr::kStressEnergyComponents;
  Kokkos::View<Real * [pangu::nr::kZ4cComponents]> result("SYNC-0 matter RHS", cases);
  Kokkos::View<Real * [pangu::nr::kConstraintComponents]> constraints("SYNC-0 constraints", cases);
  Kokkos::parallel_for(
      "SYNC-0 AthenaK Tmunu component oracle", Kokkos::RangePolicy<>(0, cases),
      KOKKOS_LAMBDA(const int test) {
        pangu::nr::rhs::PointState state{};
        state.chi = 0.83;
        state.metric[0] = 1.20;
        state.metric[1] = 0.10;
        state.metric[2] = -0.06;
        state.metric[3] = 0.90;
        state.metric[4] = 0.04;
        state.metric[5] = 1.10;
        state.alpha = 0.91;
        pangu::nr::Z4cOptions options{};
        pangu::nr::StressEnergy matter{};
        const Real amplitude = (test % 2 == 0 ? 1.0 : -1.0) * 0.031 * (test + 1);
        if (test < 6)
          matter.stress[test] = amplitude;
        else if (test == Index(StressEnergyComponent::energy))
          matter.energy = amplitude;
        else
          matter.momentum[test - Index(StressEnergyComponent::momentum_x)] = amplitude;
        Real local[pangu::nr::kZ4cComponents]{};
        pangu::nr::rhs::ApplyMatterSources(state, options, matter, local);
        for (int component = 0; component < pangu::nr::kZ4cComponents; ++component)
          result(test, component) = local[component];

        pangu::nr::ADMConstraintInput input{};
        input.metric[0] = input.metric[3] = input.metric[5] = 1.0;
        input.energy = matter.energy;
        for (int axis = 0; axis < 3; ++axis)
          input.momentum[axis] = matter.momentum[axis];
        pangu::nr::ConstraintValues values{};
        pangu::nr::ComputeADMConstraints(input, values);
        for (int component = 0; component < pangu::nr::kConstraintComponents; ++component)
          constraints(test, component) = values.values[component];
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  const auto host_constraints =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), constraints);

  const Real metric[6] = {1.20, 0.10, -0.06, 0.90, 0.04, 1.10};
  Real inverse[6]{};
  InvertSymmetric(metric, inverse);
  constexpr Real chi = 0.83;
  constexpr Real alpha = 0.91;
  const Real inverse_psi4 = chi;
  Real maximum = 0.0;
  for (int test = 0; test < cases; ++test) {
    const Real amplitude = (test % 2 == 0 ? 1.0 : -1.0) * 0.031 * (test + 1);
    Real stress[6]{};
    Real energy = 0.0;
    Real momentum[3]{};
    if (test < 6)
      stress[test] = amplitude;
    else if (test == Index(StressEnergyComponent::energy))
      energy = amplitude;
    else
      momentum[test - Index(StressEnergyComponent::momentum_x)] = amplitude;

    Real trace = 0.0;
    for (int first = 0; first < 3; ++first)
      for (int second = 0; second < 3; ++second)
        trace += inverse_psi4 * inverse[SpatialSymmetricComponent(first, second)] *
                 stress[SpatialSymmetricComponent(first, second)];
    Real expected[pangu::nr::kZ4cComponents]{};
    expected[Index(Z4cComponent::khat)] = 4.0 * M_PI * alpha * (trace + energy);
    expected[Index(Z4cComponent::theta)] = -8.0 * M_PI * alpha * energy;
    for (int first = 0; first < 3; ++first)
      for (int second = 0; second < 3; ++second)
        expected[Index(Z4cComponent::gamx) + first] -=
            16.0 * M_PI * alpha * inverse[SpatialSymmetricComponent(first, second)] *
            momentum[second];
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        const int pair = SpatialSymmetricComponent(first, second);
        expected[Index(Z4cComponent::axx) + pair] =
            -8.0 * M_PI * alpha *
            (inverse_psi4 * stress[pair] - trace * metric[pair] / 3.0);
      }
    }
    for (int component = 0; component < pangu::nr::kZ4cComponents; ++component)
      maximum = std::max(maximum, ScaledDifference(host(test, component), expected[component]));

    const Real expected_hamiltonian = -16.0 * M_PI * energy;
    maximum = std::max(
        maximum,
        ScaledDifference(host_constraints(test, Index(pangu::nr::ConstraintComponent::hamiltonian)),
                         expected_hamiltonian));
    for (int axis = 0; axis < 3; ++axis) {
      maximum = std::max(
          maximum,
          ScaledDifference(host_constraints(
                               test, Index(pangu::nr::ConstraintComponent::momentum_x) + axis),
                           -8.0 * M_PI * momentum[axis]));
    }
  }

  const Real tolerance = 512.0 * std::numeric_limits<Real>::epsilon();
  if (!(std::isfinite(maximum) && maximum <= tolerance)) {
    std::cerr << "SYNC-0 stress-energy oracle failed: maximum scaled error=" << maximum << '\n';
    return 1;
  }
  std::cout << "SYNC-0 stress-energy oracle PASS: maximum scaled error=" << maximum << '\n';
  return 0;
}
