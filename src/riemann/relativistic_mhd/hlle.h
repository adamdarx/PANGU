#ifndef PANGU_RIEMANN_RELATIVISTIC_MHD_HLLE_H_
#define PANGU_RIEMANN_RELATIVISTIC_MHD_HLLE_H_

#include <cmath>

#include <parthenon/parthenon.hpp>

namespace pangu::riemann::relativistic_mhd {

// HLLE flux in 3+1 form bounded by the extremal characteristic speeds.
// A state provides five-component `flux` and `conserved` arrays, three-component `induction`
// and `magnetic` arrays, and the characteristic bounds `lambda_minus` and `lambda_plus`; the
// metric provides `lapse` and `spatial_det`. Fluxes are densitized by the spatial volume.
struct HLLE {
  template <typename State, typename Metric>
  KOKKOS_INLINE_FUNCTION static State Solve(const State& left, const State& right,
                                            const Metric& metric) {
    using parthenon::Real;
    const Real lambda_left = fmin(left.lambda_minus, right.lambda_minus);
    const Real lambda_right = fmax(left.lambda_plus, right.lambda_plus);
    const State* selected = nullptr;
    if (lambda_left >= 0.0)
      selected = &left;
    else if (lambda_right <= 0.0)
      selected = &right;
    State result{};
    const Real spatial_volume = sqrt(metric.spatial_det);
    if (selected != nullptr) {
      for (int component = 0; component < 5; ++component)
        result.flux[component] = spatial_volume * metric.lapse * selected->flux[component];
      for (int axis = 0; axis < 3; ++axis)
        result.induction[axis] = spatial_volume * metric.lapse * selected->induction[axis];
    } else {
      const Real inverse = 1.0 / (lambda_right - lambda_left);
      const Real jump_factor = lambda_right * lambda_left / metric.lapse;
      for (int component = 0; component < 5; ++component) {
        const Real hll =
            (lambda_right * left.flux[component] - lambda_left * right.flux[component] +
             jump_factor * (right.conserved[component] - left.conserved[component])) *
            inverse;
        result.flux[component] = spatial_volume * metric.lapse * hll;
      }
      for (int axis = 0; axis < 3; ++axis) {
        const Real hll =
            (lambda_right * left.induction[axis] - lambda_left * right.induction[axis] +
             jump_factor * (right.magnetic[axis] - left.magnetic[axis])) *
            inverse;
        result.induction[axis] = spatial_volume * metric.lapse * hll;
      }
    }
    result.lambda_minus = lambda_left;
    result.lambda_plus = lambda_right;
    return result;
  }
};

} // namespace pangu::riemann::relativistic_mhd

#endif
