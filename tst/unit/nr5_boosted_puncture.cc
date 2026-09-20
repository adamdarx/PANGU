#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "z4c/analytic_initial_data.h"

namespace {
using Real = parthenon::Real;
using pangu::nr::BoostedPunctureData;
using pangu::nr::BuildBoostedPuncture;
using pangu::nr::BuildSchwarzschildIsotropic;
using pangu::nr::SpatialSymmetricComponent;

struct PointResult {
  int valid = 0;
  int finite = 0;
  int positive = 0;
  int symmetry = 0;
  int nonzero_extrinsic = 0;
  Real rest_metric_error = 0.0;
  Real rest_extrinsic_error = 0.0;
};
} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const Real coordinates[4][3] = {
      {3.0, 0.0, 0.0}, {2.0, 1.0, 0.0}, {-2.0, 1.0, 1.0}, {0.0, 0.0, 4.0}};
  Kokkos::View<PointResult[4]> results("NR-5 boosted puncture results");
  Kokkos::parallel_for(
      "NR-5 boosted puncture analytic oracle", 4, KOKKOS_LAMBDA(const int point) {
        const Real x = coordinates[point][0];
        const Real y = coordinates[point][1];
        const Real z = coordinates[point][2];
        BoostedPunctureData boosted{};
        results(point).valid = BuildBoostedPuncture(x, y, z, 1.0, 0.5, boosted) ? 1 : 0;
        if (!results(point).valid) return;
        results(point).finite = 1;
        results(point).positive = boosted.adm.lapse > 0.0 ? 1 : 0;
        for (int component = 0; component < 6; ++component) {
          results(point).finite = results(point).finite &&
                                  Kokkos::isfinite(boosted.adm.metric[component]) &&
                                  Kokkos::isfinite(boosted.adm.extrinsic[component]);
          if (component == SpatialSymmetricComponent(0, 0) ||
              component == SpatialSymmetricComponent(1, 1) ||
              component == SpatialSymmetricComponent(2, 2))
            results(point).positive =
                results(point).positive && boosted.adm.metric[component] > 0.0;
        }
        for (int axis = 0; axis < 3; ++axis) {
          results(point).finite = results(point).finite && Kokkos::isfinite(boosted.adm.shift[axis]);
          results(point).positive = results(point).positive && boosted.adm.lapse > 0.0;
        }
        results(point).symmetry =
            fabs(boosted.adm.metric[SpatialSymmetricComponent(1, 1)] -
                 boosted.adm.metric[SpatialSymmetricComponent(2, 2)]) < 1.0e-14 &&
            fabs(boosted.adm.metric[SpatialSymmetricComponent(1, 2)]) < 1.0e-14;
        Real extrinsic_norm = 0.0;
        for (int component = 0; component < 6; ++component)
          extrinsic_norm = fmax(extrinsic_norm, fabs(boosted.adm.extrinsic[component]));
        results(point).nonzero_extrinsic = extrinsic_norm > 1.0e-12;

        BoostedPunctureData rest{};
        pangu::nr::SchwarzschildIsotropicData isotropic{};
        const bool rest_ok = BuildBoostedPuncture(x, y, z, 1.0, 0.0, rest) &&
                             BuildSchwarzschildIsotropic(x, y, z, 1.0, -4.0, isotropic);
        if (!rest_ok) return;
        for (int component = 0; component < 6; ++component) {
          results(point).rest_metric_error =
              fmax(results(point).rest_metric_error,
                   fabs(rest.adm.metric[component] - isotropic.adm.metric[component]));
          results(point).rest_extrinsic_error =
              fmax(results(point).rest_extrinsic_error,
                   fabs(rest.adm.extrinsic[component] - isotropic.adm.extrinsic[component]));
        }
      });
  Kokkos::fence();
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
  const Real tolerance = 4096.0 * std::numeric_limits<Real>::epsilon();
  Real max_metric_error = 0.0;
  Real max_extrinsic_error = 0.0;
  int all_valid = 1;
  int all_finite = 1;
  int all_positive = 1;
  int all_symmetric = 1;
  int all_moving = 1;
  for (int point = 0; point < 4; ++point) {
    all_valid = all_valid && host(point).valid;
    all_finite = all_finite && host(point).finite;
    all_positive = all_positive && host(point).positive;
    all_symmetric = all_symmetric && host(point).symmetry;
    all_moving = all_moving && host(point).nonzero_extrinsic;
    max_metric_error = std::max(max_metric_error, host(point).rest_metric_error);
    max_extrinsic_error = std::max(max_extrinsic_error, host(point).rest_extrinsic_error);
  }
  if (!all_valid || !all_finite || !all_positive || !all_symmetric || !all_moving ||
      !std::isfinite(max_metric_error) || !std::isfinite(max_extrinsic_error) ||
      max_metric_error > tolerance || max_extrinsic_error > tolerance) {
    std::cerr << "NR-5 boosted puncture oracle failure: valid=" << all_valid
              << " finite=" << all_finite << " positive=" << all_positive
              << " symmetry=" << all_symmetric << " moving_extrinsic=" << all_moving
              << " rest_metric_error=" << max_metric_error
              << " rest_extrinsic_error=" << max_extrinsic_error
              << " tolerance=" << tolerance << '\n';
    return 1;
  }
  std::cout << "NR-5 boosted puncture oracle PASS: rest_metric_error=" << max_metric_error
            << " rest_extrinsic_error=" << max_extrinsic_error << " tolerance=" << tolerance
            << '\n';
  return 0;
}
