#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "z4c/analytic_initial_data.h"

namespace {
using Real = parthenon::Real;
using pangu::nr::BuildSchwarzschildIsotropic;
using pangu::nr::ComputeADMConstraints;
using pangu::nr::ConstraintComponent;
using pangu::nr::Index;
using pangu::nr::SchwarzschildIsotropicData;

KOKKOS_INLINE_FUNCTION Real ConstraintError(const SchwarzschildIsotropicData& data,
                                            pangu::nr::ConstraintValues& values) {
  ComputeADMConstraints(data.constraints, values);
  Real error = 0.0;
  for (int component = Index(ConstraintComponent::hamiltonian);
       component <= Index(ConstraintComponent::z_norm); ++component)
    error = fmax(error, fabs(values.values[component]));
  return error;
}
} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const Real coordinates[4][3] = {
      {3.0, 4.0, 12.0}, {2.0, 0.0, 0.0}, {-2.0, 1.0, 0.0}, {0.0, 0.0, 4.0}};
  Kokkos::View<Real[4]> errors("NR-3 Schwarzschild errors");
  Kokkos::View<int[4]> valid("NR-3 Schwarzschild valid");
  Kokkos::parallel_for(
      "NR-3 Schwarzschild initial data", 4, KOKKOS_LAMBDA(const int point) {
        SchwarzschildIsotropicData data{};
        const bool ok = BuildSchwarzschildIsotropic(coordinates[point][0], coordinates[point][1],
                                                    coordinates[point][2], 1.0, -4.0, data);
        valid(point) = ok ? 1 : 0;
        pangu::nr::ConstraintValues local{};
        errors(point) = ok ? ConstraintError(data, local) : 1.0e300;
      });
  Kokkos::fence();
  const auto host_errors = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
  const auto host_valid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), valid);
  Real maximum = 0.0;
  int all_valid = 1;
  for (int point = 0; point < 4; ++point) {
    maximum = std::max(maximum, host_errors(point));
    all_valid = all_valid && host_valid(point);
  }
  const Real tolerance = 4096.0 * std::numeric_limits<Real>::epsilon();
  if (!all_valid || !std::isfinite(maximum) || maximum > tolerance) {
    std::cerr << "NR-3 Schwarzschild initial-data failure: error=" << maximum
              << " tolerance=" << tolerance << " valid=" << all_valid << '\n';
    return 1;
  }
  std::cout << "NR-3 Schwarzschild initial-data PASS: max constraint=" << maximum
            << " tolerance=" << tolerance << '\n';
  return 0;
}
