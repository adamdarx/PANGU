#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "radiation/cooling_model.h"

namespace {

using Real = parthenon::Real;
using pangu::geometry::MetricPoint;
using pangu::radiation::AddCoolingIncrement;
using pangu::radiation::BuildCoolingIncrement;
using pangu::radiation::BuildFourVelocity;
using pangu::radiation::EvaluateTargetThicknessCooling;

KOKKOS_INLINE_FUNCTION MetricPoint MinkowskiMetric() {
  MetricPoint metric{};
  for (int axis = 0; axis < 4; ++axis) {
    metric.lower[axis][axis] = axis == 0 ? -1.0 : 1.0;
    metric.upper[axis][axis] = axis == 0 ? -1.0 : 1.0;
  }
  metric.lapse = 1.0;
  metric.gdet = 1.0;
  metric.spatial_det = 1.0;
  return metric;
}

KOKKOS_INLINE_FUNCTION Real RelativeDifference(const Real actual, const Real expected) {
  return fabs(actual - expected) / fmax(1.0, fmax(fabs(actual), fabs(expected)));
}

void Require(const bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  try {
    constexpr int checks = 24;
    Kokkos::View<Real[checks]> errors("target-thickness errors");
    Kokkos::parallel_for(
        "PANGU RC-4 target-thickness model", 1, KOKKOS_LAMBDA(const int) {
          int check = 0;
          constexpr Real gamma = 4.0 / 3.0;
          const pangu::eos::IdealGas eos{gamma};
          constexpr Real density = 1.7;
          constexpr Real internal = 0.8;
          constexpr Real radius = 12.0;
          constexpr Real spin = 0.9375;
          constexpr Real h_target = 0.1;
          constexpr Real dt = 0.25;
          constexpr Real u_time = 1.3;
          constexpr Real beta_fast = 0.2 * 3.1415926535897932384626433832795;
          constexpr Real beta_medium = 2.0 * 3.1415926535897932384626433832795;
          constexpr Real beta_slow = 20.0 * 3.1415926535897932384626433832795;

          const auto fast = EvaluateTargetThicknessCooling(density, internal, eos, radius, spin,
                                                           h_target, beta_fast, dt, u_time, 1.0);
          const auto medium = EvaluateTargetThicknessCooling(
              density, internal, eos, radius, spin, h_target, beta_medium, dt, u_time, 1.0);
          const auto slow = EvaluateTargetThicknessCooling(density, internal, eos, radius, spin,
                                                           h_target, beta_slow, dt, u_time, 1.0);
          const Real omega = 1.0 / (radius * sqrt(radius) + spin);
          const Real expected_target =
              density * h_target * h_target * (radius * omega) * (radius * omega) / (gamma - 1.0);
          errors(check++) = RelativeDifference(fast.target_internal_energy, expected_target);
          const auto double_h = EvaluateTargetThicknessCooling(
              density, internal, eos, radius, spin, 2.0 * h_target, beta_fast, dt, u_time, 1.0);
          errors(check++) =
              RelativeDifference(double_h.target_internal_energy, 4.0 * expected_target);
          errors(check++) = fast.removed_fraction > medium.removed_fraction &&
                                    medium.removed_fraction > slow.removed_fraction
                                ? 0.0
                                : 1.0;
          errors(check++) = fast.removed_internal_energy > medium.removed_internal_energy &&
                                    medium.removed_internal_energy > slow.removed_internal_energy
                                ? 0.0
                                : 1.0;

          constexpr int steps = 80;
          Real evolved = internal;
          bool monotone = true;
          bool above_target = true;
          for (int step = 0; step < steps; ++step) {
            const Real previous = evolved;
            const auto cooling = EvaluateTargetThicknessCooling(
                density, evolved, eos, radius, spin, h_target, beta_medium, dt, u_time, 1.0);
            evolved -= cooling.removed_internal_energy;
            monotone = monotone && evolved <= previous;
            above_target = above_target && evolved >= expected_target;
          }
          const Real expected_evolved =
              expected_target +
              (internal - expected_target) * exp(-steps * dt / u_time / (beta_medium / omega));
          errors(check++) = RelativeDifference(evolved, expected_evolved);
          errors(check++) = monotone ? 0.0 : 1.0;
          errors(check++) = above_target ? 0.0 : 1.0;

          const Real inverse_sqrt2 = 1.0 / sqrt(2.0);
          const Real inverse_sqrt3 = 1.0 / sqrt(3.0);
          const Real points[4][3]{
              {radius, 0.0, 0.0},
              {radius * inverse_sqrt2, radius * inverse_sqrt2, 0.0},
              {radius * inverse_sqrt3, radius * inverse_sqrt3, radius * inverse_sqrt3},
              {-radius, 0.0, 0.0}};
          for (int point = 0; point < 4; ++point) {
            const Real rotated_radius =
                sqrt(points[point][0] * points[point][0] + points[point][1] * points[point][1] +
                     points[point][2] * points[point][2]);
            const auto rotated =
                EvaluateTargetThicknessCooling(density, internal, eos, rotated_radius, spin,
                                               h_target, beta_fast, dt, u_time, 1.0);
            errors(check++) =
                RelativeDifference(rotated.target_internal_energy, fast.target_internal_energy);
            errors(check++) =
                RelativeDifference(rotated.removed_internal_energy, fast.removed_internal_energy);
          }

          const auto flat = MinkowskiMetric();
          pangu::relativity::MHDPrimitiveState primitive{};
          primitive.fluid = {1.0, {0.35, -0.12, 0.08}, (gamma - 1.0) * 0.4};
          const pangu::eos::RelativisticEOS recovery_eos{{gamma}, 1.0e-12, 1.0e-12, 1.0e-30};
          const auto base = pangu::relativity::ConvertGRMHDP2CWithInternalEnergy(primitive, 0.4,
                                                                                recovery_eos, flat);
          const auto increment = BuildCoolingIncrement(primitive.fluid, flat, 0.03);
          const auto cooled = AddCoolingIncrement(base, increment);
          errors(check++) = RelativeDifference(cooled.energy - base.energy, increment.energy);
          for (int axis = 0; axis < 3; ++axis)
            errors(check++) = RelativeDifference(cooled.momentum[axis] - base.momentum[axis],
                                                 increment.momentum[axis]);
          errors(check++) = RelativeDifference(cooled.density, base.density);
          const auto recovered = pangu::relativity::SolveGRMHDC2P(cooled, primitive.magnetic,
                                                                  recovery_eos, 100.0, 1.0e6, flat);
          errors(check++) = recovered.success ? 0.0 : 1.0;
          errors(check++) = recovered.success && recovered.internal_energy > 0.0 &&
                                    recovered.internal_energy < 0.4
                                ? 0.0
                                : 1.0;
          errors(check++) = recovered.success && isfinite(recovered.primitive.fluid.u[0]) &&
                                    isfinite(recovered.primitive.fluid.u[1]) &&
                                    isfinite(recovered.primitive.fluid.u[2])
                                ? 0.0
                                : 1.0;
          const auto four_velocity = BuildFourVelocity(primitive.fluid, flat);
          Real projection = four_velocity.upper[0] * increment.energy;
          for (int axis = 0; axis < 3; ++axis)
            projection += four_velocity.upper[axis + 1] * increment.momentum[axis];
          errors(check++) =
              RelativeDifference(projection, 0.03 * four_velocity.upper[0] * flat.gdet);
        });
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
    Real maximum = 0.0;
    for (int check = 0; check < checks; ++check) {
      Require(std::isfinite(host(check)), "non-finite target-thickness test result");
      if (host(check) > 1.0e-14)
        std::cerr << "RC-4 check[" << check << "] error=" << std::scientific << host(check) << '\n';
      maximum = fmax(maximum, host(check));
    }
    Require(maximum <= 2.0e-13, "target-thickness model exceeds the roundoff tolerance");
    std::cout << "PANGU RC-4 target-thickness cooling: PASS, max error=" << std::scientific
              << maximum << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "PANGU RC-4 target-thickness cooling: FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
