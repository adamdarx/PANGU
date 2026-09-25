#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "z4c/core/adm_conversion.h"

namespace {
using Real = parthenon::Real;
using pangu::nr::ADMMetricDerivatives;
using pangu::nr::ADMState;
using pangu::nr::ConvertADMToZ4c;
using pangu::nr::ConvertZ4cToADM;
using pangu::nr::Index;
using pangu::nr::Z4cComponent;
using pangu::nr::Z4cState;

KOKKOS_INLINE_FUNCTION Real Difference(const Real measured, const Real expected) {
  const Real scale = fmax(1.0, fmax(fabs(measured), fabs(expected)));
  return fabs(measured - expected) / scale;
}
} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  constexpr int kCases = 66;
  Kokkos::View<Real* [2]> errors("NR-3 ADM conversion errors", kCases);
  Kokkos::parallel_for(
      "NR-3 ADM/Z4c round-trip", kCases, KOKKOS_LAMBDA(const int sample) {
        ADMState adm{};
        ADMMetricDerivatives derivatives{};
        if (sample == 0) {
          adm.metric[0] = adm.metric[3] = adm.metric[5] = 1.0;
          adm.extrinsic[0] = 0.2;
          adm.extrinsic[3] = -0.1;
          adm.extrinsic[5] = 0.05;
          adm.lapse = 1.0;
          adm.theta = 0.03;
        } else {
          // Deterministic positive-definite metric; no random device state is used.
          const Real x = static_cast<Real>(sample);
          const Real l00 = 1.0 + 0.01 * fmod(x * 17.0, 31.0);
          const Real l10 = -0.2 + 0.01 * fmod(x * 13.0, 37.0);
          const Real l11 = 1.1 + 0.01 * fmod(x * 19.0, 29.0);
          const Real l20 = -0.15 + 0.01 * fmod(x * 23.0, 23.0);
          const Real l21 = -0.1 + 0.01 * fmod(x * 29.0, 19.0);
          const Real l22 = 0.9 + 0.01 * fmod(x * 31.0, 17.0);
          adm.metric[0] = l00 * l00;
          adm.metric[1] = l00 * l10;
          adm.metric[2] = l00 * l20;
          adm.metric[3] = l10 * l10 + l11 * l11;
          adm.metric[4] = l10 * l20 + l11 * l21;
          adm.metric[5] = l20 * l20 + l21 * l21 + l22 * l22;
          for (int component = 0; component < 6; ++component) {
            adm.extrinsic[component] = 0.03 * sin(0.17 * x * (component + 1));
            for (int derivative = 0; derivative < 3; ++derivative)
              derivatives.dmetric[derivative][component] =
                  0.02 * cos(0.11 * x * (component + derivative + 1));
          }
          adm.lapse = 0.7 + 0.01 * fmod(x, 41.0);
          adm.shift[0] = 0.2 * sin(0.13 * x);
          adm.shift[1] = 0.2 * cos(0.19 * x);
          adm.shift[2] = 0.1 * sin(0.23 * x);
          adm.theta = 0.02 * cos(0.07 * x);
        }

        Z4cState z4c{};
        ADMState round_trip{};
        const bool forward = ConvertADMToZ4c(adm, derivatives, -4.0, z4c);
        const bool backward = ConvertZ4cToADM(z4c, -4.0, round_trip);
        Real round_error = forward && backward ? 0.0 : 1.0;
        for (int component = 0; component < 6; ++component) {
          round_error =
              fmax(round_error, Difference(round_trip.metric[component], adm.metric[component]));
          round_error = fmax(round_error,
                             Difference(round_trip.extrinsic[component], adm.extrinsic[component]));
        }
        round_error = fmax(round_error, Difference(round_trip.lapse, adm.lapse));
        round_error = fmax(round_error, Difference(round_trip.theta, adm.theta));
        for (int axis = 0; axis < 3; ++axis)
          round_error = fmax(round_error, Difference(round_trip.shift[axis], adm.shift[axis]));

        Real conformal_metric[6]{};
        Real conformal_a[6]{};
        for (int component = 0; component < 6; ++component) {
          conformal_metric[component] = z4c.values[Index(Z4cComponent::gxx) + component];
          conformal_a[component] = z4c.values[Index(Z4cComponent::axx) + component];
        }
        const Real conformal_det = pangu::nr::rhs::SpatialDeterminant(conformal_metric);
        Real conformal_inverse[6]{};
        pangu::nr::rhs::SpatialInverse(1.0 / conformal_det, conformal_metric, conformal_inverse);
        Real trace_a = 0.0;
        for (int first = 0; first < 3; ++first)
          for (int second = 0; second < 3; ++second)
            trace_a += pangu::nr::rhs::SymmetricValue(conformal_inverse, first, second) *
                       pangu::nr::rhs::SymmetricValue(conformal_a, first, second);
        errors(sample, 0) = round_error;
        errors(sample, 1) = fmax(fabs(conformal_det - 1.0), fabs(trace_a));
      });
  Kokkos::fence();
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
  Real round_max = 0.0;
  Real algebraic_max = 0.0;
  for (int sample = 0; sample < kCases; ++sample) {
    round_max = std::max(round_max, host(sample, 0));
    algebraic_max = std::max(algebraic_max, host(sample, 1));
  }
  const Real tolerance = 4096.0 * std::numeric_limits<Real>::epsilon();
  if (!(std::isfinite(round_max) && std::isfinite(algebraic_max)) || round_max > tolerance ||
      algebraic_max > tolerance) {
    std::cerr << "NR-3 ADM conversion failure: round-trip=" << round_max
              << " algebraic=" << algebraic_max << " tolerance=" << tolerance << '\n';
    return 1;
  }
  std::cout << "NR-3 ADM conversion PASS: round-trip=" << round_max
            << " algebraic=" << algebraic_max << '\n';
  return 0;
}
