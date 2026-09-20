#ifndef PANGU_RIEMANN_RELATIVISTIC_LLF_H_
#define PANGU_RIEMANN_RELATIVISTIC_LLF_H_

#include <cmath>

#include <parthenon/parthenon.hpp>

namespace pangu::riemann::relativistic {

// Local Lax-Friedrichs flux from two face states. A state provides five-component `flux` and
// `conserved` arrays and the characteristic bounds `lambda_minus` and `lambda_plus`.
struct LLF {
  template <typename State>
  KOKKOS_INLINE_FUNCTION static void Solve(const State& left, const State& right,
                                           parthenon::Real output[5]) {
    const parthenon::Real lambda = fmax(fmax(fabs(left.lambda_minus), fabs(left.lambda_plus)),
                                        fmax(fabs(right.lambda_minus), fabs(right.lambda_plus)));
    for (int n = 0; n < 5; ++n)
      output[n] =
          0.5 * (left.flux[n] + right.flux[n] - lambda * (right.conserved[n] - left.conserved[n]));
  }
};

} // namespace pangu::riemann::relativistic

#endif
