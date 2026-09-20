#ifndef PANGU_RIEMANN_RELATIVISTIC_MHD_LLF_H_
#define PANGU_RIEMANN_RELATIVISTIC_MHD_LLF_H_

#include <cmath>

#include <parthenon/parthenon.hpp>

namespace pangu::riemann::relativistic_mhd {

// Local Lax-Friedrichs flux in 3+1 form with the extremal characteristic speed.
// A state provides five-component `flux` and `conserved` arrays, three-component `induction`
// and `magnetic` arrays, and the characteristic bounds `lambda_minus` and `lambda_plus`; the
// metric provides `lapse` and `spatial_det`. Fluxes are densitized by the spatial volume.
struct LLF {
  template <typename State, typename Metric>
  KOKKOS_INLINE_FUNCTION static State Solve(const State& left, const State& right,
                                            const Metric& metric) {
    using parthenon::Real;
    State result{};
    const Real lambda_left = fmin(left.lambda_minus, right.lambda_minus);
    const Real lambda_right = fmax(left.lambda_plus, right.lambda_plus);
    const Real speed = fmax(lambda_right, -lambda_left);
    const Real spatial_volume = sqrt(metric.spatial_det);
    for (int component = 0; component < 5; ++component) {
      result.flux[component] = 0.5 * spatial_volume *
                               (metric.lapse * (left.flux[component] + right.flux[component]) -
                                speed * (right.conserved[component] - left.conserved[component]));
    }
    for (int axis = 0; axis < 3; ++axis) {
      result.induction[axis] = 0.5 * spatial_volume *
                               (metric.lapse * (left.induction[axis] + right.induction[axis]) -
                                speed * (right.magnetic[axis] - left.magnetic[axis]));
    }
    result.lambda_minus = lambda_left;
    result.lambda_plus = lambda_right;
    return result;
  }
};

} // namespace pangu::riemann::relativistic_mhd

#endif
