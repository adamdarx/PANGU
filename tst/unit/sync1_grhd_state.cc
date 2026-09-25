#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "z4c/coupling/sync_grhd.h"

namespace {
using Real = parthenon::Real;

KOKKOS_INLINE_FUNCTION pangu::geometry::MetricPoint TestMetric(const int test) {
  pangu::geometry::MetricPoint metric{};
  const Real spatial[3][3] = {{1.2 + 0.01 * test, 0.08, -0.03},
                              {0.08, 0.95 + 0.005 * test, 0.04},
                              {-0.03, 0.04, 1.1}};
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      metric.lower[first + 1][second + 1] = spatial[first][second];
  metric.spatial_det =
      spatial[0][0] * (spatial[1][1] * spatial[2][2] - spatial[1][2] * spatial[2][1]) -
      spatial[0][1] * (spatial[1][0] * spatial[2][2] - spatial[1][2] * spatial[2][0]) +
      spatial[0][2] * (spatial[1][0] * spatial[2][1] - spatial[1][1] * spatial[2][0]);
  metric.lapse = 0.91 + 0.002 * test;
  metric.shift[0] = 0.03;
  metric.shift[1] = -0.015;
  metric.shift[2] = 0.01;
  Real inverse[3][3]{};
  inverse[0][0] = (spatial[1][1] * spatial[2][2] - spatial[1][2] * spatial[2][1]) /
                  metric.spatial_det;
  inverse[0][1] = (spatial[0][2] * spatial[2][1] - spatial[0][1] * spatial[2][2]) /
                  metric.spatial_det;
  inverse[0][2] = (spatial[0][1] * spatial[1][2] - spatial[0][2] * spatial[1][1]) /
                  metric.spatial_det;
  inverse[1][0] = inverse[0][1];
  inverse[1][1] = (spatial[0][0] * spatial[2][2] - spatial[0][2] * spatial[2][0]) /
                  metric.spatial_det;
  inverse[1][2] = (spatial[0][2] * spatial[1][0] - spatial[0][0] * spatial[1][2]) /
                  metric.spatial_det;
  inverse[2][0] = inverse[0][2];
  inverse[2][1] = inverse[1][2];
  inverse[2][2] = (spatial[0][0] * spatial[1][1] - spatial[0][1] * spatial[1][0]) /
                  metric.spatial_det;
  Real lowered_shift[3]{};
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      lowered_shift[first] += spatial[first][second] * metric.shift[second];
  metric.lower[0][0] = -metric.lapse * metric.lapse;
  for (int axis = 0; axis < 3; ++axis)
    metric.lower[0][0] += lowered_shift[axis] * metric.shift[axis];
  const Real inverse_lapse2 = 1.0 / (metric.lapse * metric.lapse);
  metric.upper[0][0] = -inverse_lapse2;
  for (int axis = 0; axis < 3; ++axis) {
    metric.lower[0][axis + 1] = lowered_shift[axis];
    metric.lower[axis + 1][0] = lowered_shift[axis];
    metric.upper[0][axis + 1] = metric.shift[axis] * inverse_lapse2;
    metric.upper[axis + 1][0] = metric.upper[0][axis + 1];
    for (int second = 0; second < 3; ++second)
      metric.upper[axis + 1][second + 1] =
          inverse[axis][second] - metric.shift[axis] * metric.shift[second] * inverse_lapse2;
  }
  metric.gdet = metric.lapse * sqrt(metric.spatial_det);
  return metric;
}

Real ScaledDifference(const Real measured, const Real expected) {
  return std::abs(measured - expected) / std::max({1.0, std::abs(measured), std::abs(expected)});
}
} // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  constexpr int cases = 16;
  Kokkos::View<Real * [37]> device("SYNC-1 GRHD state", cases);
  Kokkos::parallel_for(
      "SYNC-1 AthenaK GRHD state oracle", Kokkos::RangePolicy<>(0, cases),
      KOKKOS_LAMBDA(const int test) {
        const auto metric = TestMetric(test);
        const pangu::relativity::HydroPrimitiveState primitive{
            0.7 + 0.03 * test,
            {0.04 * (test + 1), -0.015 * (test + 2), 0.01 * (test - 3)},
            0.08 + 0.004 * test};
        const pangu::eos::RelativisticEOS eos{{5.0 / 3.0}, 1.0e-12, 1.0e-12};
        const auto conserved = pangu::nr::ConvertSyncGRHDP2C(primitive, eos, metric);
        const auto matter = pangu::nr::BuildSyncGRHDStressEnergy(primitive, conserved, metric);
        device(test, 0) = conserved.density;
        for (int axis = 0; axis < 3; ++axis)
          device(test, 1 + axis) = conserved.momentum[axis];
        device(test, 4) = conserved.energy;
        for (int component = 0; component < 6; ++component)
          device(test, 5 + component) = matter.stress[component];
        device(test, 11) = matter.energy;
        for (int axis = 0; axis < 3; ++axis)
          device(test, 12 + axis) = matter.momentum[axis];
        pangu::nr::SyncGRHDGeometryDerivatives derivatives{};
        for (int direction = 0; direction < 3; ++direction) {
          derivatives.lapse[direction] = 0.003 * (direction + 1) * (test + 1);
          for (int component = 0; component < 3; ++component)
            derivatives.shift[direction][component] =
                -0.002 * (direction + 1) * (component + 2);
          for (int component = 0; component < 6; ++component)
            derivatives.spatial_metric[direction][component] =
                0.001 * (direction + 1) * (component + 1);
        }
        for (int component = 0; component < 6; ++component)
          derivatives.extrinsic[component] = -0.004 * (component + 1);
        Real source[5]{};
        pangu::nr::ComputeSyncGRHDSource(matter, metric, derivatives, source);
        for (int component = 0; component < 5; ++component)
          device(test, 15 + component) = source[component];
        const auto recovered = pangu::nr::SolveSyncGRHDC2P(conserved, eos, 1000.0, metric);
        device(test, 20) = recovered.primitive.density;
        for (int axis = 0; axis < 3; ++axis)
          device(test, 21 + axis) = recovered.primitive.u[axis];
        device(test, 24) = recovered.primitive.pressure;
        const int direction = test % 3;
        const auto flux = pangu::nr::BuildSyncGRHDFluxState(primitive, eos, direction, metric);
        for (int component = 0; component < 5; ++component) {
          device(test, 25 + component) = flux.conserved[component];
          device(test, 30 + component) = flux.flux[component];
        }
        device(test, 35) = flux.lambda_plus;
        device(test, 36) = flux.lambda_minus;
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);

  Real maximum = 0.0;
  Real maximum_round_trip = 0.0;
  for (int test = 0; test < cases; ++test) {
    const auto metric = TestMetric(test);
    const pangu::relativity::HydroPrimitiveState primitive{
        0.7 + 0.03 * test,
        {0.04 * (test + 1), -0.015 * (test + 2), 0.01 * (test - 3)},
        0.08 + 0.004 * test};
    const Real gamma = 5.0 / 3.0;
    Real lower[3]{};
    Real projected2 = 0.0;
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second)
        lower[first] += metric.lower[first + 1][second + 1] * primitive.u[second];
      projected2 += primitive.u[first] * lower[first];
    }
    const Real lorentz = std::sqrt(1.0 + projected2);
    const Real volume = std::sqrt(metric.spatial_det);
    const Real enthalpy = primitive.density + gamma * primitive.pressure / (gamma - 1.0);
    const Real density = volume * primitive.density * lorentz;
    Real momentum[3]{};
    for (int axis = 0; axis < 3; ++axis)
      momentum[axis] = volume * enthalpy * lorentz * lower[axis];
    const Real energy = volume * (enthalpy * lorentz * lorentz - primitive.pressure) - density;
    maximum = std::max(maximum, ScaledDifference(host(test, 0), density));
    for (int axis = 0; axis < 3; ++axis)
      maximum = std::max(maximum, ScaledDifference(host(test, 1 + axis), momentum[axis]));
    maximum = std::max(maximum, ScaledDifference(host(test, 4), energy));

    const Real expected_energy = (energy + density) / volume;
    maximum = std::max(maximum, ScaledDifference(host(test, 11), expected_energy));
    for (int first = 0; first < 3; ++first) {
      maximum = std::max(maximum,
                         ScaledDifference(host(test, 12 + first), momentum[first] / volume));
      for (int second = first; second < 3; ++second) {
        const int pair = pangu::nr::SpatialSymmetricComponent(first, second);
        const Real stress = momentum[first] / volume * lower[second] / lorentz +
                            primitive.pressure * metric.lower[first + 1][second + 1];
        maximum = std::max(maximum, ScaledDifference(host(test, 5 + pair), stress));
      }
    }

    Real inverse[3][3]{};
    inverse[0][0] = (metric.lower[2][2] * metric.lower[3][3] -
                     metric.lower[2][3] * metric.lower[3][2]) /
                    metric.spatial_det;
    inverse[0][1] = (metric.lower[1][3] * metric.lower[3][2] -
                     metric.lower[1][2] * metric.lower[3][3]) /
                    metric.spatial_det;
    inverse[0][2] = (metric.lower[1][2] * metric.lower[2][3] -
                     metric.lower[1][3] * metric.lower[2][2]) /
                    metric.spatial_det;
    inverse[1][0] = inverse[0][1];
    inverse[1][1] = (metric.lower[1][1] * metric.lower[3][3] -
                     metric.lower[1][3] * metric.lower[3][1]) /
                    metric.spatial_det;
    inverse[1][2] = (metric.lower[1][3] * metric.lower[2][1] -
                     metric.lower[1][1] * metric.lower[2][3]) /
                    metric.spatial_det;
    inverse[2][0] = inverse[0][2];
    inverse[2][1] = inverse[1][2];
    inverse[2][2] = (metric.lower[1][1] * metric.lower[2][2] -
                     metric.lower[1][2] * metric.lower[2][1]) /
                    metric.spatial_det;
    Real stress_upper[3][3]{};
    for (int first = 0; first < 3; ++first)
      for (int second = 0; second < 3; ++second)
        for (int lower_first = 0; lower_first < 3; ++lower_first)
          for (int lower_second = 0; lower_second < 3; ++lower_second)
            stress_upper[first][second] +=
                inverse[first][lower_first] * inverse[second][lower_second] *
                host(test, 5 + pangu::nr::SpatialSymmetricComponent(lower_first, lower_second));
    Real lapse_derivative[3]{};
    Real shift_derivative[3][3]{};
    Real metric_derivative[3][6]{};
    Real extrinsic[6]{};
    for (int direction = 0; direction < 3; ++direction) {
      lapse_derivative[direction] = 0.003 * (direction + 1) * (test + 1);
      for (int component = 0; component < 3; ++component)
        shift_derivative[direction][component] =
            -0.002 * (direction + 1) * (component + 2);
      for (int component = 0; component < 6; ++component)
        metric_derivative[direction][component] =
            0.001 * (direction + 1) * (component + 1);
    }
    for (int component = 0; component < 6; ++component)
      extrinsic[component] = -0.004 * (component + 1);
    Real source[5]{};
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
        source[4] += volume *
                     (metric.lapse *
                          extrinsic[pangu::nr::SpatialSymmetricComponent(first, second)] *
                          stress_upper[first][second] -
                      inverse[first][second] * (momentum[first] / volume) *
                          lapse_derivative[second]);
      }
    }
    for (int direction = 0; direction < 3; ++direction) {
      source[1 + direction] = -volume * expected_energy * lapse_derivative[direction];
      for (int first = 0; first < 3; ++first) {
        source[1 + direction] += momentum[first] * shift_derivative[direction][first];
        for (int second = 0; second < 3; ++second) {
          source[1 + direction] +=
              0.5 * metric.lapse * volume * stress_upper[first][second] *
              metric_derivative[direction]
                               [pangu::nr::SpatialSymmetricComponent(first, second)];
        }
      }
    }
    for (int component = 0; component < 5; ++component)
      maximum = std::max(maximum, ScaledDifference(host(test, 15 + component), source[component]));

    pangu::relativity::HydroConservedState local{};
    local.density = density / volume;
    local.energy = energy / volume;
    Real momentum_squared = 0.0;
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
        local.momentum[first] += inverse[first][second] * momentum[second] / volume;
        momentum_squared += inverse[first][second] * momentum[first] * momentum[second] /
                            (volume * volume);
      }
    }
    const pangu::eos::RelativisticEOS eos{{gamma}, 1.0e-12, 1.0e-12};
    const auto reference_recovered =
        pangu::relativity::SolveSRHDC2PFromInvariants(local, momentum_squared, eos, 1000.0);
    maximum = std::max(
        maximum, ScaledDifference(host(test, 20), reference_recovered.primitive.density));
    for (int axis = 0; axis < 3; ++axis) {
      maximum = std::max(
          maximum,
          ScaledDifference(host(test, 21 + axis), reference_recovered.primitive.u[axis]));
    }
    maximum = std::max(
        maximum, ScaledDifference(host(test, 24), reference_recovered.primitive.pressure));

    const int direction = test % 3;
    maximum = std::max(maximum, ScaledDifference(host(test, 25), density));
    for (int axis = 0; axis < 3; ++axis)
      maximum = std::max(maximum, ScaledDifference(host(test, 26 + axis), momentum[axis]));
    maximum = std::max(maximum, ScaledDifference(host(test, 29), energy));
    const Real transport =
        primitive.u[direction] / lorentz - metric.shift[direction] / metric.lapse;
    const Real expected_flux[5]{
        metric.lapse * density * transport,
        metric.lapse *
            (momentum[0] * transport + (direction == 0 ? volume * primitive.pressure : 0.0)),
        metric.lapse *
            (momentum[1] * transport + (direction == 1 ? volume * primitive.pressure : 0.0)),
        metric.lapse *
            (momentum[2] * transport + (direction == 2 ? volume * primitive.pressure : 0.0)),
        metric.lapse *
            (energy * transport + volume * primitive.pressure * primitive.u[direction] / lorentz)};
    for (int component = 0; component < 5; ++component)
      maximum = std::max(maximum, ScaledDifference(host(test, 30 + component),
                                                   expected_flux[component]));
    Real upper_four[4]{lorentz / metric.lapse, 0.0, 0.0, 0.0};
    for (int axis = 0; axis < 3; ++axis)
      upper_four[axis + 1] =
          primitive.u[axis] - lorentz * metric.shift[axis] / metric.lapse;
    const Real sound2 = gamma * primitive.pressure / enthalpy;
    const int normal = direction + 1;
    const Real aa = upper_four[0] * upper_four[0] -
                    (metric.upper[0][0] + upper_four[0] * upper_four[0]) * sound2;
    const Real bb = -2.0 * (upper_four[0] * upper_four[normal] -
                            (metric.upper[0][normal] + upper_four[0] * upper_four[normal]) *
                                sound2);
    const Real cc = upper_four[normal] * upper_four[normal] -
                    (metric.upper[normal][normal] + upper_four[normal] * upper_four[normal]) *
                        sound2;
    const Real discriminant = std::max(0.0, bb * bb - 4.0 * aa * cc);
    const Real root_first = (-bb + std::sqrt(discriminant)) / (2.0 * aa);
    const Real root_second = (-bb - std::sqrt(discriminant)) / (2.0 * aa);
    maximum = std::max(maximum,
                       ScaledDifference(host(test, 35), std::max(root_first, root_second)));
    maximum = std::max(maximum,
                       ScaledDifference(host(test, 36), std::min(root_first, root_second)));
    maximum_round_trip =
        std::max(maximum_round_trip, ScaledDifference(host(test, 20), primitive.density));
    for (int axis = 0; axis < 3; ++axis) {
      maximum_round_trip = std::max(
          maximum_round_trip, ScaledDifference(host(test, 21 + axis), primitive.u[axis]));
    }
    maximum_round_trip =
        std::max(maximum_round_trip, ScaledDifference(host(test, 24), primitive.pressure));
  }

  const Real tolerance = 512.0 * std::numeric_limits<Real>::epsilon();
  const Real recovery_tolerance = 1.0e-11;
  if (!(std::isfinite(maximum) && maximum <= tolerance &&
        std::isfinite(maximum_round_trip) && maximum_round_trip <= recovery_tolerance)) {
    std::cerr << "SYNC-1 GRHD state oracle failed: maximum scaled error=" << maximum
              << " round-trip error=" << maximum_round_trip << '\n';
    return 1;
  }
  std::cout << "SYNC-1 GRHD state oracle PASS: maximum scaled error=" << maximum
            << " round-trip error=" << maximum_round_trip << '\n';
  return 0;
}
