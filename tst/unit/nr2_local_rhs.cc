#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "z4c/evolution/z4c_rhs.h"

namespace {
using Real = parthenon::Real;
using pangu::nr::Index;
using pangu::nr::SpatialSymmetricComponent;
using pangu::nr::Z4cComponent;

Real ScaledDifference(const Real measured, const Real expected) {
  return std::abs(measured - expected) / std::max({1.0, std::abs(measured), std::abs(expected)});
}
} // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  constexpr int cases = 3;
  Kokkos::View<Real * [pangu::nr::kZ4cComponents]> result("nr2 local rhs", cases);
  Kokkos::parallel_for(
      "NR-2 local RHS oracle cases", Kokkos::RangePolicy<>(0, cases),
      KOKKOS_LAMBDA(const int test) {
        pangu::nr::rhs::PointState state{};
        pangu::nr::rhs::PointDerivatives derivatives{};
        pangu::nr::Z4cOptions options{};
        Real local[pangu::nr::kZ4cComponents]{};
        state.chi = 1.0;
        state.metric[0] = 1.0;
        state.metric[3] = 1.0;
        state.metric[5] = 1.0;
        state.alpha = 1.0;

        if (test == 0) {
          state.chi = 0.8;
          state.khat = 0.17;
          state.theta = -0.03;
          state.alpha = 0.91;
          state.a[0] = 0.07;
          state.a[3] = -0.07;
          state.gamma[0] = 0.04;
          state.gamma[1] = -0.05;
          state.gamma[2] = 0.06;
          state.beta[0] = -0.02;
          state.beta[1] = 0.03;
          state.beta[2] = -0.04;
          options.damp_kappa1 = 0.13;
          options.damp_kappa2 = -0.2;
          options.shift_gamma = 0.8;
          options.shift_alpha2_gamma = 0.17;
          options.shift_eta = 1.4;
        } else if (test == 1) {
          derivatives.dgamma[0][0] = 0.11;
          derivatives.dgamma[0][1] = -0.07;
          derivatives.dgamma[0][2] = 0.03;
          derivatives.dgamma[1][0] = 0.05;
          derivatives.dgamma[1][1] = -0.02;
          derivatives.dgamma[1][2] = 0.09;
          derivatives.dgamma[2][0] = -0.04;
          derivatives.dgamma[2][1] = 0.08;
          derivatives.dgamma[2][2] = 0.06;
          for (int derivative = 0; derivative < 3; ++derivative) {
            const int pair = SpatialSymmetricComponent(derivative, derivative);
            for (int component = 0; component < 6; ++component)
              derivatives.ddmetric[pair][component] = 0.013 * (1 + derivative) * (1 + component);
          }
        } else {
          state.chi = 0.73;
          derivatives.dchi[0] = 0.08;
          derivatives.dchi[1] = -0.06;
          derivatives.dchi[2] = 0.04;
          for (int first = 0; first < 3; ++first) {
            for (int second = first; second < 3; ++second) {
              derivatives.ddchi[SpatialSymmetricComponent(first, second)] =
                  0.017 * (1 + first + 2 * second);
            }
          }
        }
        pangu::nr::rhs::ComputeVacuum(state, derivatives, options, 0.37, local);
        for (int component = 0; component < pangu::nr::kZ4cComponents; ++component)
          result(test, component) = local[component];
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

  Real expected[cases][pangu::nr::kZ4cComponents]{};
  {
    const Real chi = 0.8;
    const Real khat = 0.17;
    const Real theta = -0.03;
    const Real alpha = 0.91;
    const Real avalue = 0.07;
    const Real trace_k = khat + 2.0 * theta;
    const Real aa = 2.0 * avalue * avalue;
    const Real kappa1 = 0.13;
    const Real kappa2 = -0.2;
    expected[0][Index(Z4cComponent::khat)] =
        alpha * (aa + trace_k * trace_k / 3.0) + kappa1 * (1.0 - kappa2) * alpha * theta;
    expected[0][Index(Z4cComponent::chi)] = (2.0 / 3.0) * chi * alpha * trace_k;
    expected[0][Index(Z4cComponent::theta)] =
        alpha * (0.5 * ((2.0 / 3.0) * trace_k * trace_k - aa) - (2.0 + kappa2) * kappa1 * theta);
    const Real gamma[3] = {0.04, -0.05, 0.06};
    const Real beta[3] = {-0.02, 0.03, -0.04};
    for (int axis = 0; axis < 3; ++axis) {
      expected[0][Index(Z4cComponent::gamx) + axis] = -2.0 * alpha * kappa1 * gamma[axis];
      expected[0][Index(Z4cComponent::betax) + axis] =
          0.8 * gamma[axis] + 0.17 * alpha * alpha * gamma[axis] - 1.4 * beta[axis];
    }
    expected[0][Index(Z4cComponent::gxx)] = -2.0 * alpha * avalue;
    expected[0][Index(Z4cComponent::gyy)] = 2.0 * alpha * avalue;
    expected[0][Index(Z4cComponent::axx)] = alpha * (trace_k * avalue - 2.0 * avalue * avalue);
    expected[0][Index(Z4cComponent::ayy)] = alpha * (-trace_k * avalue - 2.0 * avalue * avalue);
    expected[0][Index(Z4cComponent::alpha)] = -(2.0 * 1.0) * alpha * khat;
  }
  {
    const Real dgamma[3][3] = {{0.11, -0.07, 0.03}, {0.05, -0.02, 0.09}, {-0.04, 0.08, 0.06}};
    Real ricci[6]{};
    Real trace = 0.0;
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        const int component = SpatialSymmetricComponent(first, second);
        Real laplacian = 0.0;
        for (int derivative = 0; derivative < 3; ++derivative)
          laplacian += 0.013 * (1 + derivative) * (1 + component);
        ricci[component] = 0.5 * (dgamma[second][first] + dgamma[first][second] - laplacian);
        if (first == second)
          trace += ricci[component];
      }
    }
    expected[1][Index(Z4cComponent::theta)] = 0.5 * trace;
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        const int component = SpatialSymmetricComponent(first, second);
        expected[1][Index(Z4cComponent::axx) + component] =
            ricci[component] - (first == second ? trace / 3.0 : 0.0);
      }
    }
  }
  {
    const Real chi = 0.73;
    const Real dchi[3] = {0.08, -0.06, 0.04};
    Real dphi[3]{};
    for (int axis = 0; axis < 3; ++axis)
      dphi[axis] = dchi[axis] / (-4.0 * chi);
    Real ddphi[6]{};
    Real rphi[6]{};
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        const int component = SpatialSymmetricComponent(first, second);
        const Real ddchi = 0.017 * (1 + first + 2 * second);
        ddphi[component] = ddchi / (-4.0 * chi) + 4.0 * dphi[first] * dphi[second];
      }
    }
    Real contracted = 0.0;
    for (int axis = 0; axis < 3; ++axis)
      contracted += ddphi[SpatialSymmetricComponent(axis, axis)] + 2.0 * dphi[axis] * dphi[axis];
    Real trace = 0.0;
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        const int component = SpatialSymmetricComponent(first, second);
        rphi[component] = 4.0 * dphi[first] * dphi[second] - 2.0 * ddphi[component] -
                          (first == second ? 2.0 * contracted : 0.0);
        if (first == second)
          trace += chi * rphi[component];
      }
    }
    expected[2][Index(Z4cComponent::theta)] = 0.5 * trace;
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        const int component = SpatialSymmetricComponent(first, second);
        expected[2][Index(Z4cComponent::axx) + component] =
            chi * rphi[component] - (first == second ? trace / 3.0 : 0.0);
      }
    }
  }

  Real maximum = 0.0;
  for (int test = 0; test < cases; ++test) {
    for (int component = 0; component < pangu::nr::kZ4cComponents; ++component)
      maximum =
          std::max(maximum, ScaledDifference(host(test, component), expected[test][component]));
  }
  const Real tolerance = 512.0 * std::numeric_limits<Real>::epsilon();
  if (!(std::isfinite(maximum) && maximum <= tolerance)) {
    std::cerr << "NR-2 local RHS oracle failed: maximum scaled error=" << maximum << '\n';
    return 1;
  }
  std::cout << "NR-2 local RHS oracle PASS: maximum scaled error=" << maximum << '\n';
  return 0;
}
