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
struct OracleSymmetric {
  KOKKOS_INLINE_FUNCTION Real operator()(const Real *metric, int a, int b) const {
    return metric[pangu::nr::SpatialSymmetricComponent(a,b)];
  }
};

struct CopyCountPack {
  int* copies;
  Real* data;
  KOKKOS_INLINE_FUNCTION CopyCountPack(int* count, Real* values)
      : copies(count), data(values) {}
  KOKKOS_INLINE_FUNCTION CopyCountPack(const CopyCountPack& other)
      : copies(other.copies), data(other.data) { ++*copies; }
  KOKKOS_INLINE_FUNCTION Real& operator()(int, int component, int, int, int) const {
    return data[component];
  }
};

bool CheckPointPackStorage() {
  int copies = 0;
  Real values[2] = {2.0, 3.0};
  const CopyCountPack pack(&copies, values);
  const auto& cached = pangu::nr::rhs::CachePointPack(pack);
  const pangu::nr::rhs::ComponentAccessor<decltype(cached)> read{cached, 0, 0, 0, 0, 0};
  const pangu::nr::rhs::PointOutput<decltype(cached)> write{cached, 0, 0, 0, 0};
  write[1] = read(0, 0, 0) + 5.0;
  if (values[1] != 7.0) return false;
  if constexpr (std::is_same_v<parthenon::DevExecSpace, parthenon::HostExecSpace>) {
    if (copies != 0) return false;
  }
  static_assert(!std::is_same_v<parthenon::DevExecSpace, parthenon::HostExecSpace> ||
                std::is_reference_v<pangu::nr::rhs::PointPackStorage<CopyCountPack>>);
  return true;
}

Real ScaledDifference(const Real measured, const Real expected) {
  if (!std::isfinite(measured) || !std::isfinite(expected))
    return std::numeric_limits<Real>::infinity();
  return std::abs(measured - expected) / std::max({1.0, std::abs(measured), std::abs(expected)});
}
} // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (!CheckPointPackStorage()) {
    std::cerr << "Point-local pack storage copied a host pack or lost its alias\n";
    return 1;
  }
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
          options.gauge.shift_gamma = 0.8;
          options.gauge.shift_alpha2_gamma = 0.17;
          options.gauge.shift_eta = 1.4;
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
  // Eq. (28) with chi != 1 and nonzero advection detects all three terms
  // independently; flat-space/zero-shift waves cannot expose these errors.
  Kokkos::View<Real*> harmonic_result("AwA harmonic gauge oracle", 3);
  Kokkos::parallel_for("AwA harmonic gauge oracle", 3, KOKKOS_LAMBDA(const int axis) {
    pangu::nr::gauge::Options options;
    options.shift = pangu::nr::gauge::Shift::harmonic;
    options.shift_advect = 1.0;
    const Real gamma[3] = {1.0, -1.0, 2.0};
    const Real beta[3] = {0.0, 0.0, 0.0};
    const Real lie[3] = {7.0, 8.0, 9.0};
    const Real dalpha[3] = {0.1, 0.2, 0.3};
    const Real dchi[3] = {0.4, 0.5, 0.6};
    const Real inverse[6] = {1.0, 0.0, 0.0, 2.0, 0.0, 3.0};
    harmonic_result(axis) = pangu::nr::gauge::ShiftRhs(
        options, axis, 2.0, 0.5, gamma, beta, lie, dalpha, dchi, inverse, OracleSymmetric{});
  });
  const auto harmonic_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), harmonic_result);
  const Real harmonic_expected[3] = {9.7, 7.6, 15.7};
  for (int axis = 0; axis < 3; ++axis)
    maximum = std::max(maximum, ScaledDifference(harmonic_host(axis), harmonic_expected[axis]));
  // Changing the evolved conformal variable must preserve the physical RHS.
  // Transform exact first/second derivatives by the chain rule, independently
  // of the implementation's log-conformal-factor intermediates.
  Kokkos::View<Real* [pangu::nr::kZ4cComponents]> variable_errors("chi/W RHS invariance", 4);
  Kokkos::parallel_for("chi/W RHS invariance", 4, KOKKOS_LAMBDA(const int test) {
    pangu::nr::rhs::PointState chi{};
    pangu::nr::rhs::PointDerivatives d{};
    pangu::nr::Z4cOptions options{};
    chi.chi=0.73;chi.alpha=0.91;chi.khat=0.17;chi.theta=-0.03;
    chi.metric[0]=chi.metric[3]=chi.metric[5]=1;
    chi.a[0]=0.07;chi.a[3]=-0.07;
    options.gauge.slow_start_lapse=true;
    options.gauge.shift= test==0 ? pangu::nr::gauge::Shift::gamma_driver :
        (test==1 ? pangu::nr::gauge::Shift::harmonic :
         (test==2 ? pangu::nr::gauge::Shift::legacy : pangu::nr::gauge::Shift::frozen));
    options.gauge.shift_harmonic=0.8;
    d.lie_chi=0.04;d.lie_alpha=-0.02;
    for(int axis=0;axis<3;++axis) {
      chi.gamma[axis]=0.04*(axis+1);chi.beta[axis]=-0.02*(axis+1);
      d.dchi[axis]=0.08-0.03*axis;d.dalpha[axis]=0.02*(axis+1);
      d.lie_beta[axis]=0.01*(axis+1);
      for(int other=0;other<3;++other) {
        d.dbeta[axis][other]=0.003*(1+axis+other);
        d.dgamma[axis][other]=0.004*(1+axis-other);
      }
    }
    for(int pair=0;pair<6;++pair) d.ddchi[pair]=0.017*(pair+1);
    auto w=chi;auto dw=d;auto w_options=options;
    w.chi=sqrt(chi.chi);w_options.chi_psi_power=-2;
    dw.lie_chi=d.lie_chi/(2*w.chi);
    for(int first=0;first<3;++first) {
      dw.dchi[first]=d.dchi[first]/(2*w.chi);
      for(int second=first;second<3;++second) {
        const int pair=SpatialSymmetricComponent(first,second);
        dw.ddchi[pair]=d.ddchi[pair]/(2*w.chi)-
            d.dchi[first]*d.dchi[second]/(4*w.chi*w.chi*w.chi);
      }
    }
    Real chi_rhs[pangu::nr::kZ4cComponents]{},w_rhs[pangu::nr::kZ4cComponents]{};
    pangu::nr::rhs::ComputeVacuum(chi,d,options,0.37,chi_rhs);
    pangu::nr::rhs::ComputeVacuum(w,dw,w_options,0.37,w_rhs);
    pangu::nr::StressEnergy matter{};
    matter.energy=0.17;matter.stress[0]=0.2;matter.stress[3]=0.1;matter.stress[5]=-0.03;
    matter.momentum[0]=0.003;matter.momentum[1]=-0.002;
    pangu::nr::rhs::ApplyMatterSources(chi,options,matter,chi_rhs);
    pangu::nr::rhs::ApplyMatterSources(w,w_options,matter,w_rhs);
    for(int component=0;component<pangu::nr::kZ4cComponents;++component) {
      const Real expected=component==Index(Z4cComponent::chi) ?
          chi_rhs[component]/(2*w.chi) : chi_rhs[component];
      variable_errors(test,component)=Kokkos::isfinite(w_rhs[component]) && Kokkos::isfinite(expected) ?
          fabs(w_rhs[component]-expected)/fmax(1.0,fabs(expected)) : 1.0;
    }
  });
  const auto variable_host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),variable_errors);
  for(int test=0;test<4;++test)
    for(int component=0;component<pangu::nr::kZ4cComponents;++component)
      maximum=std::max(maximum,variable_host(test,component));
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
