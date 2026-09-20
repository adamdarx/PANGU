#include <algorithm>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>
#include <parthenon/parthenon.hpp>

#include "z4c/nr_amr.h"

using parthenon::Real;
namespace amr = pangu::nr::amr;

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
  if (failures != 0) {
    std::cerr << "NR-6 AthenaK AMR oracle failed " << failures << "/" << cases << " cases\n";
    return 1;
  }
  std::cout << "NR-6 AthenaK AMR GPU oracle PASS: " << cases
            << " tracker/chi/dchi/fixed-radius decisions\n";
  return 0;
}
