#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "z4c/core/component_indices.h"
#include "z4c/puncture/bowen_york.h"

namespace {
using Real = parthenon::Real;
using pangu::nr::SpatialSymmetricComponent;
using pangu::nr::puncture::EvaluateFreeData;
using pangu::nr::puncture::EvaluateHamiltonianPoint;
using pangu::nr::puncture::FreeData;
using pangu::nr::puncture::HamiltonianPoint;
using pangu::nr::puncture::Puncture;

struct Results {
  int valid = 1;
  Real stationary_error = 0.0;
  Real momentum_error = 0.0;
  Real spin_error = 0.0;
  Real trace_error = 0.0;
  Real source_error = 0.0;
};
} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  Kokkos::View<Results> result("native Bowen-York checks");
  Kokkos::parallel_for(
      "native Bowen-York analytic oracle", 1, KOKKOS_LAMBDA(const int) {
        Results value{};
        Puncture body{};
        body.mass = 2.0;

        FreeData stationary{};
        value.valid &= EvaluateFreeData(2.0, 0.0, 0.0, &body, 1, stationary);
        value.stationary_error = fabs(stationary.singular_psi - 1.5);
        for (int component = 0; component < 6; ++component)
          value.stationary_error =
              fmax(value.stationary_error, fabs(stationary.conformal_extrinsic[component]));

        body.momentum[0] = 0.25;
        FreeData momentum{};
        value.valid &= EvaluateFreeData(2.0, 0.0, 0.0, &body, 1, momentum);
        const Real factor = 1.5 * body.momentum[0] / (2.0 * 2.0);
        value.momentum_error =
            fmax(fabs(momentum.conformal_extrinsic[SpatialSymmetricComponent(0, 0)] -
                      2.0 * factor),
                 fabs(momentum.conformal_extrinsic[SpatialSymmetricComponent(1, 1)] + factor));
        value.momentum_error =
            fmax(value.momentum_error,
                 fabs(momentum.conformal_extrinsic[SpatialSymmetricComponent(2, 2)] + factor));

        body.momentum[0] = 0.0;
        body.spin[2] = 0.5;
        FreeData spinning{};
        value.valid &= EvaluateFreeData(2.0, 0.0, 0.0, &body, 1, spinning);
        const Real expected_xy = 3.0 * body.spin[2] / (2.0 * 2.0 * 2.0);
        value.spin_error =
            fabs(spinning.conformal_extrinsic[SpatialSymmetricComponent(0, 1)] - expected_xy);
        const Real expected_squared = 2.0 * expected_xy * expected_xy;
        value.spin_error =
            fmax(value.spin_error, fabs(spinning.conformal_extrinsic_squared - expected_squared));

        for (int sample = 0; sample < 4; ++sample) {
          const Real points[4][3] = {{1.0, 2.0, 3.0}, {-2.0, 1.0, 0.5},
                                     {0.5, -1.5, 2.5}, {3.0, 0.25, -0.75}};
          FreeData general{};
          value.valid &= EvaluateFreeData(points[sample][0], points[sample][1], points[sample][2],
                                          &body, 1, general);
          const Real trace = general.conformal_extrinsic[0] + general.conformal_extrinsic[3] +
                             general.conformal_extrinsic[5];
          value.trace_error = fmax(value.trace_error, fabs(trace));
        }

        HamiltonianPoint hamiltonian{};
        value.valid &= EvaluateHamiltonianPoint(2.0, 0.0, 0.0, &body, 1, 0.1, hamiltonian);
        const Real expected_source =
            -0.125 * pow(hamiltonian.psi, -7.0) * spinning.conformal_extrinsic_squared;
        value.source_error = fabs(hamiltonian.source - expected_source);
        result() = value;
      });
  Kokkos::fence();

  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  const Real tolerance = 4096.0 * std::numeric_limits<Real>::epsilon();
  const Real maximum = std::max({host().stationary_error, host().momentum_error,
                                 host().spin_error, host().trace_error, host().source_error});
  if (!host().valid || !std::isfinite(maximum) || maximum > tolerance) {
    std::cerr << "native Bowen-York oracle failure: valid=" << host().valid
              << " stationary=" << host().stationary_error
              << " momentum=" << host().momentum_error << " spin=" << host().spin_error
              << " trace=" << host().trace_error << " source=" << host().source_error
              << " tolerance=" << tolerance << '\n';
    return 1;
  }
  std::cout << "native Bowen-York oracle PASS: maximum_error=" << maximum
            << " tolerance=" << tolerance << '\n';
  return 0;
}
