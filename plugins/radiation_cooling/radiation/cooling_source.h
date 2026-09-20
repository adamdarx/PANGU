#ifndef PANGU_RADIATION_COOLING_SOURCE_H_
#define PANGU_RADIATION_COOLING_SOURCE_H_

#include <parthenon/parthenon.hpp>

#include "radiation/cooling_model.h"

namespace pangu::radiation {

KOKKOS_INLINE_FUNCTION bool
CoolingCandidateIsRecoverable(const relativity::HydroConservedState& conserved,
                              const parthenon::Real magnetic[3], const eos::RelativisticEOS& eos,
                              const parthenon::Real gamma_max, const parthenon::Real sigma_max,
                              const geometry::MetricPoint& metric) {
  const auto result =
      relativity::SolveGRMHDC2P(conserved, magnetic, eos, gamma_max, sigma_max, metric);
  return result.success && !result.density_floor && !result.pressure_floor &&
         !result.lorentz_ceiling && !result.sigma_ceiling;
}

KOKKOS_INLINE_FUNCTION parthenon::Real
LimitCoolingIncrement(const relativity::HydroConservedState& base,
                      const CoolingIncrement& increment, const parthenon::Real magnetic[3],
                      const eos::RelativisticEOS& eos, const parthenon::Real gamma_max,
                      const parthenon::Real sigma_max, const geometry::MetricPoint& metric,
                      bool& base_valid) {
  const auto full = AddCoolingIncrement(base, increment);
  if (CoolingCandidateIsRecoverable(full, magnetic, eos, gamma_max, sigma_max, metric)) {
    base_valid = true;
    return 1.0;
  }
  base_valid = CoolingCandidateIsRecoverable(base, magnetic, eos, gamma_max, sigma_max, metric);
  if (!base_valid)
    return 0.0;

  parthenon::Real lower = 0.0;
  parthenon::Real upper = 1.0;
  constexpr int bisection_iterations = 12;
  for (int iteration = 0; iteration < bisection_iterations; ++iteration) {
    const parthenon::Real middle = 0.5 * (lower + upper);
    const auto candidate = AddCoolingIncrement(base, increment, middle);
    if (CoolingCandidateIsRecoverable(candidate, magnetic, eos, gamma_max, sigma_max, metric))
      lower = middle;
    else
      upper = middle;
  }
  return lower;
}

parthenon::TaskStatus ApplyCoolingMeshTask(parthenon::MeshData<parthenon::Real>* data,
                                           parthenon::Real time, parthenon::Real dt,
                                           bool record_diagnostics);

} // namespace pangu::radiation

#endif
