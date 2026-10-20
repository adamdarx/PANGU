#include <algorithm>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>
#include <parthenon/parthenon.hpp>

#include "z4c/evolution/amr.h"

using parthenon::Real;
namespace amr = pangu::nr::amr;

// Independent transcription of RefineTracker in the inspected AthenaK
// reference (8a5d398, src/z4c/z4c_amr.cpp). AthenaK evaluates this on the host;
// the oracle below checks its geometric/level semantics, not CPU/GPU bitwise
// equivalence or either framework's subsequent coarsening/tree policy.
template <class Scalar, class Bounds, class Centers, class Radii, class Levels>
KOKKOS_INLINE_FUNCTION int AthenaTrackerTagOracle(const Bounds& bounds, const int level,
                                                 const int count, const Centers& centers,
                                                 const Radii& radii, const Levels& levels) {
  int flag = -1;
  for (int tracker = 0; tracker < count; ++tracker) {
    const Scalar x = centers[3 * tracker];
    const Scalar y = centers[3 * tracker + 1];
    const Scalar z = centers[3 * tracker + 2];
    const Scalar cx = fmax(bounds[0], fmin(x, bounds[1]));
    const Scalar cy = fmax(bounds[2], fmin(y, bounds[3]));
    const Scalar cz = fmax(bounds[4], fmin(z, bounds[5]));
    const Scalar dmin2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
    const bool contained = x >= bounds[0] && x <= bounds[1] &&
                           y >= bounds[2] && y <= bounds[3] &&
                           z >= bounds[4] && z <= bounds[5];
    int requested = -1;
    if (dmin2 < radii[tracker] * radii[tracker] || contained) {
      if (levels[tracker] < 0 || level < levels[tracker])
        requested = 1;
      else if (level == levels[tracker])
        requested = 0;
    }
    flag = requested > flag ? requested : flag;
  }
  return flag;
}

template <class Scalar> int CheckTrackerTangency(const char* precision) {
  constexpr int cases = 3 * 3 * 3 * 8;
  constexpr int checks = 7;
  Kokkos::View<int**> errors("NR tracker tangency roundoff", cases, checks);
  Kokkos::View<int*> intentional_differences("Tracker tangency strict oracle differences", cases);
  Kokkos::parallel_for(
      "NR tracker tangency at faces edges corners", cases, KOKKOS_LAMBDA(const int test) {
        int index = test;
        const int signs = index % 8;
        index /= 8;
        const int translation = index % 3;
        index /= 3;
        const int scale_index = index % 3;
        const int shape = index / 3;
        const Scalar scale = scale_index == 0 ? Scalar(1.e-6)
                            : scale_index == 1 ? Scalar(1.) : Scalar(1.e6);
        const Scalar origin = Scalar(translation - 1) * Scalar(64.) * scale;
        // Exact Pythagorean distances: face 4, edge 3-4-5, corner 2-3-6-7.
        const Scalar distances[3] = {Scalar(shape == 0 ? 4 : shape == 1 ? 3 : 2),
                                     Scalar(shape == 0 ? 0 : shape == 1 ? 4 : 3),
                                     Scalar(shape == 2 ? 6 : 0)};
        const Scalar radius = Scalar(shape == 0 ? 4 : shape == 1 ? 5 : 7) * scale;
        Kokkos::Array<Scalar, 6> bounds{};
        Kokkos::Array<Scalar, 3> center{};
        const Kokkos::Array<Scalar, 1> radii{radius};
        const Kokkos::Array<int, 1> levels{2};
        for (int axis = 0; axis < 3; ++axis) {
          const Scalar sign = (signs & (1 << axis)) ? Scalar(-1.) : Scalar(1.);
          if (distances[axis] == Scalar(0.)) {
            bounds[2 * axis] = origin - Scalar(4.) * scale;
            bounds[2 * axis + 1] = origin + Scalar(4.) * scale;
          } else {
            const Scalar near = origin + sign * distances[axis] * scale;
            const Scalar far = near + sign * Scalar(4.) * scale;
            bounds[2 * axis] = fmin(near, far);
            bounds[2 * axis + 1] = fmax(near, far);
          }
          center[axis] = origin;
        }
        errors(test, 0) = amr::TrackerTag<Scalar>(bounds, 1, 1, center, radii, levels) !=
                          amr::kDerefine;
        const Scalar coordinate_scale = fmax(fabs(origin), radius);
        const Scalar jitter = std::numeric_limits<Scalar>::epsilon() * coordinate_scale;
        for (int direction = 0; direction < 2; ++direction) {
          for (int axis = 0; axis < 3; ++axis) {
            const Scalar sign = (signs & (1 << axis)) ? Scalar(-1.) : Scalar(1.);
            center[axis] = origin + sign * (direction == 0 ? jitter : -jitter);
          }
          const int robust = amr::TrackerTag<Scalar>(bounds, 1, 1, center, radii, levels);
          const int strict = AthenaTrackerTagOracle<Scalar>(bounds, 1, 1, center, radii, levels);
          errors(test, 1 + direction) = robust != amr::kDerefine;
          intentional_differences(test) += robust != strict;
        }
        // A resolvable displacement must still cross the boundary, in every
        // reflected/translated case and in both float and double arithmetic.
        const Scalar displacement = Scalar(4096.) * jitter;
        for (int axis = 0; axis < 3; ++axis) {
          const Scalar sign = (signs & (1 << axis)) ? Scalar(-1.) : Scalar(1.);
          center[axis] = origin + sign * displacement;
        }
        errors(test, 3) = amr::TrackerTag<Scalar>(bounds, 1, 1, center, radii, levels) !=
                          amr::kRefine ||
                          amr::TrackerTag<Scalar>(bounds, 1, 1, center, radii, levels) !=
                          AthenaTrackerTagOracle<Scalar>(bounds, 1, 1, center, radii, levels);
        errors(test, 4) = amr::TrackerTag<Scalar>(bounds, 2, 1, center, radii, levels) != amr::kSame ||
                          amr::TrackerTag<Scalar>(bounds, 2, 1, center, radii, levels) !=
                          AthenaTrackerTagOracle<Scalar>(bounds, 2, 1, center, radii, levels);
        errors(test, 5) = amr::TrackerTag<Scalar>(bounds, 3, 1, center, radii, levels) !=
                          amr::kDerefine;
        for (int axis = 0; axis < 3; ++axis) {
          const Scalar sign = (signs & (1 << axis)) ? Scalar(-1.) : Scalar(1.);
          center[axis] = origin - sign * displacement;
        }
        errors(test, 6) = amr::TrackerTag<Scalar>(bounds, 1, 1, center, radii, levels) !=
                          amr::kDerefine ||
                          amr::TrackerTag<Scalar>(bounds, 1, 1, center, radii, levels) !=
                          AthenaTrackerTagOracle<Scalar>(bounds, 1, 1, center, radii, levels);
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
  int failures = 0;
  for (int test = 0; test < cases; ++test)
    for (int check = 0; check < checks; ++check) {
      failures += host(test, check);
      if (host(test, check))
        std::cerr << "Tracker tangency " << precision << " case=" << test
                  << " check=" << check << '\n';
    }
  std::cout << "Tracker tangency " << precision << ": " << cases * checks - failures << '/'
            << cases * checks << " checks passed\n";
  const auto differences = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                              intentional_differences);
  int difference_count = 0;
  for (int test = 0; test < cases; ++test)
    difference_count += differences(test);
  std::cout << "AthenaK strict oracle " << precision << ": " << difference_count
            << " intentional roundoff-band differences; resolved inside/outside checks match\n";
  return failures;
}

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  constexpr int cases = 24;
  Kokkos::View<int*> errors("NR-6 AthenaK AMR oracle", cases);
  Kokkos::parallel_for(
      "NR-6 AthenaK AMR decisions", 1, KOKKOS_LAMBDA(const int) {
        const Kokkos::Array<Real, 6> unit_box{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
        Kokkos::Array<Real, 6> centers{};
        Kokkos::Array<Real, 2> radii{};
        Kokkos::Array<int, 2> levels{};
        centers[0] = 1.1;
        centers[1] = 0.5;
        centers[2] = 0.5;
        radii[0] = 0.11;
        levels[0] = 1;
        errors(0) = amr::TrackerTag<Real>(unit_box, 0, 1, centers, radii, levels) != amr::kRefine;
        centers[1] = 1.1;
        radii[0] = 0.15;
        errors(1) = amr::TrackerTag<Real>(unit_box, 0, 1, centers, radii, levels) != amr::kRefine;
        centers[2] = 1.1;
        radii[0] = 0.18;
        errors(2) = amr::TrackerTag<Real>(unit_box, 0, 1, centers, radii, levels) != amr::kRefine;
        radii[0] = 0.17;
        errors(3) = amr::TrackerTag<Real>(unit_box, 0, 1, centers, radii, levels) != amr::kDerefine;
        centers[0] = centers[1] = centers[2] = 0.5;
        radii[0] = 0.0;
        errors(4) = amr::TrackerTag<Real>(unit_box, 0, 1, centers, radii, levels) != amr::kRefine;
        errors(5) = amr::TrackerTag<Real>(unit_box, 1, 1, centers, radii, levels) != amr::kSame;
        errors(6) = amr::TrackerTag<Real>(unit_box, 2, 1, centers, radii, levels) != amr::kDerefine;
        levels[0] = -1;
        errors(7) = amr::TrackerTag<Real>(unit_box, 7, 1, centers, radii, levels) != amr::kRefine;
        centers[0] = 2.0;
        centers[1] = 2.0;
        centers[2] = 2.0;
        centers[3] = 0.5;
        centers[4] = 0.5;
        centers[5] = 0.5;
        radii[0] = 0.1;
        radii[1] = 0.0;
        levels[0] = 2;
        levels[1] = 1;
        errors(8) = amr::TrackerTag<Real>(unit_box, 1, 2, centers, radii, levels) != amr::kSame;

        errors(9) = amr::ChiTag<Real>(0.199, 0.2) != amr::kRefine;
        errors(10) = amr::ChiTag<Real>(0.2, 0.2) != amr::kSame;
        errors(11) = amr::ChiTag<Real>(0.25, 0.2) != amr::kSame;
        errors(12) = amr::ChiTag<Real>(0.251, 0.2) != amr::kDerefine;

        const Real indicator = amr::DchiIndicator<Real>(3.0, 4.0, 12.0, 2.0, false);
        const Real normalized = amr::DchiIndicator<Real>(3.0, 4.0, 12.0, 2.0, true);
        errors(13) = fabs(indicator - 13.0) > 8.0 * std::numeric_limits<Real>::epsilon();
        errors(14) = fabs(normalized - 3.25) > 8.0 * std::numeric_limits<Real>::epsilon();
        errors(15) = amr::DchiTag<Real>(1.01, 1.0, 0.5) != amr::kRefine;
        errors(16) = amr::DchiTag<Real>(1.0, 1.0, 0.5) != amr::kSame;
        errors(17) = amr::DchiTag<Real>(0.5, 1.0, 0.5) != amr::kSame;
        errors(18) = amr::DchiTag<Real>(0.49, 1.0, 0.5) != amr::kDerefine;

        const Kokkos::Array<Real, 6> near_origin{-0.2, 0.4, -0.3, 0.5, -0.1, 0.6};
        Kokkos::Array<Real, 2> fixed_radii{0.5, 0.0};
        Kokkos::Array<int, 2> fixed_levels{2, 0};
        errors(19) = amr::ApplyFixedRadii<Real>(amr::kDerefine, near_origin, 1, 1, fixed_radii,
                                                fixed_levels) != amr::kRefine;
        errors(20) = amr::ApplyFixedRadii<Real>(amr::kDerefine, near_origin, 2, 1, fixed_radii,
                                                fixed_levels) != amr::kSame;
        errors(21) = amr::ApplyFixedRadii<Real>(amr::kRefine, near_origin, 3, 1, fixed_radii,
                                                fixed_levels) != amr::kRefine;
        const Kokkos::Array<Real, 6> symmetric_box{-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
        fixed_radii[0] = 1.1;
        errors(22) = amr::ApplyFixedRadii<Real>(amr::kDerefine, symmetric_box, 0, 1, fixed_radii,
                                                fixed_levels) != amr::kDerefine;
        fixed_radii[0] = sqrt(3.0);
        errors(23) = amr::ApplyFixedRadii<Real>(amr::kDerefine, symmetric_box, 0, 1, fixed_radii,
                                                fixed_levels) != amr::kDerefine;
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
  int failures = 0;
  for (int test = 0; test < cases; ++test)
    failures += host(test);
  failures += CheckTrackerTangency<float>("float");
  failures += CheckTrackerTangency<double>("double");
  if (failures != 0) {
    std::cerr << "NR-6 AMR oracle failed " << failures << " base/tangency checks\n";
    return 1;
  }
  std::cout << "NR-6 AthenaK AMR GPU oracle PASS: " << cases
            << " tracker/chi/dchi/fixed-radius decisions\n";
  return 0;
}
