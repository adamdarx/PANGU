#ifndef PANGU_RIEMANN_RELATIVISTIC_HLLE_H_
#define PANGU_RIEMANN_RELATIVISTIC_HLLE_H_

#include <cmath>

#include <parthenon/parthenon.hpp>

namespace pangu::riemann::relativistic {

// HLLE flux from two face states. A state provides five-component `flux` and `conserved`
// arrays and the characteristic bounds `lambda_minus` and `lambda_plus`.
struct HLLE {
  template <typename State>
  KOKKOS_INLINE_FUNCTION static void Solve(const State& left, const State& right,
                                           parthenon::Real output[5]) {
    const parthenon::Real lambda_left = fmin(left.lambda_minus, right.lambda_minus);
    const parthenon::Real lambda_right = fmax(left.lambda_plus, right.lambda_plus);
    if (lambda_left >= 0.0) {
      for (int n = 0; n < 5; ++n)
        output[n] = left.flux[n];
    } else if (lambda_right <= 0.0) {
      for (int n = 0; n < 5; ++n)
        output[n] = right.flux[n];
    } else {
      const parthenon::Real inverse = 1.0 / (lambda_right - lambda_left);
      for (int n = 0; n < 5; ++n) {
        output[n] = (lambda_right * left.flux[n] - lambda_left * right.flux[n] +
                     lambda_left * lambda_right * (right.conserved[n] - left.conserved[n])) *
                    inverse;
      }
    }
  }
};

} // namespace pangu::riemann::relativistic

#endif
