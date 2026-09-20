#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "z4c/constraints.h"

namespace {
using Real = parthenon::Real;
using pangu::nr::ADMConstraintInput;
using pangu::nr::ComputeADMConstraints;
using pangu::nr::ConstraintComponent;
using pangu::nr::ConstraintValues;
using pangu::nr::ContractedConformalChristoffel;
using pangu::nr::Index;

KOKKOS_INLINE_FUNCTION Real Difference(const Real measured, const Real expected) {
  return fabs(measured - expected) / fmax(1.0, fmax(fabs(measured), fabs(expected)));
}
} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  Kokkos::View<Real* [2]> errors("NR-3 constraint errors", 3);
  Kokkos::parallel_for(
      "NR-3 ADM constraints", 3, KOKKOS_LAMBDA(const int test) {
        if (test == 2) {
          const Real physical_metric[6] = {2.0, 0.1, 0.2, 1.5, -0.05, 1.2};
          const Real dmetric[3][6] = {{0.11, 0.02, -0.03, 0.07, 0.04, 0.05},
                                      {-0.06, 0.01, 0.08, 0.09, -0.02, 0.03},
                                      {0.04, -0.07, 0.06, 0.02, 0.01, -0.05}};
          const Real dpsi4[3] = {0.03, -0.02, 0.01};
          Real inverse[6]{};
          pangu::nr::rhs::SpatialInverse(1.0 / pangu::nr::rhs::SpatialDeterminant(physical_metric),
                                         physical_metric, inverse);
          Real contracted[3]{};
          ContractedConformalChristoffel(inverse, dmetric, 1.3, dpsi4, contracted);
          const Real expected[3] = {0.039181419324762266, 0.039100878761996115,
                                    -0.084353485227314259};
          Real error = 0.0;
          for (int axis = 0; axis < 3; ++axis)
            error = fmax(error, Difference(contracted[axis], expected[axis]));
          errors(test, 0) = error;
          errors(test, 1) = 0.0;
          return;
        }
        ADMConstraintInput input{};
        input.metric[0] = input.metric[3] = input.metric[5] = 1.0;
        if (test == 1) {
          input.extrinsic[0] = 0.2;
          input.extrinsic[3] = -0.1;
          input.extrinsic[5] = 0.05;
        }
        ConstraintValues result{};
        ComputeADMConstraints(input, result);
        Real error = 0.0;
        if (test == 0) {
          for (int component = 0; component < 4; ++component)
            error = fmax(error, fabs(result.values[component]));
          for (int component = Index(ConstraintComponent::momentum_x);
               component <= Index(ConstraintComponent::momentum_z); ++component)
            error = fmax(error, fabs(result.values[component]));
        } else {
          const Real trace = 0.2 - 0.1 + 0.05;
          const Real k_squared = 0.2 * 0.2 + 0.1 * 0.1 + 0.05 * 0.05;
          const Real expected_hamiltonian = trace * trace - k_squared;
          error = Difference(result.values[Index(ConstraintComponent::hamiltonian)],
                             expected_hamiltonian);
          error = fmax(error, fabs(result.values[Index(ConstraintComponent::momentum_norm)]));
          error = fmax(error, fabs(result.values[Index(ConstraintComponent::z_norm)]));
        }
        errors(test, 0) = error;
        errors(test, 1) = result.values[Index(ConstraintComponent::combined)];
      });
  Kokkos::fence();
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
  Real maximum = 0.0;
  Real combined = 0.0;
  for (int test = 0; test < 3; ++test) {
    maximum = std::max(maximum, host(test, 0));
    combined = std::max(combined, host(test, 1));
  }
  const Real tolerance = 2048.0 * std::numeric_limits<Real>::epsilon();
  if (!(std::isfinite(maximum) && std::isfinite(combined)) || maximum > tolerance) {
    std::cerr << "NR-3 constraint contraction failure: error=" << maximum
              << " combined=" << combined << " tolerance=" << tolerance << '\n';
    return 1;
  }
  std::cout << "NR-3 constraint contraction PASS: error=" << maximum << " combined=" << combined
            << '\n';
  return 0;
}
