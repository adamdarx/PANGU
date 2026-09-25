#include "z4c/puncture/initializer.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "z4c/puncture/bowen_york.h"

namespace pangu::nr::puncture {

bool BuildPunctureADM(const double x, const double y, const double z,
                      const std::vector<Puncture> &punctures,
                      const SpectralInterpolator &interpolator,
                      ADMState &output) {
  FreeData free{};
  if (!EvaluateFreeData(x, y, z, punctures.data(),
                        static_cast<int>(punctures.size()), free))
    return false;
  const double correction = interpolator.RegularCorrection(x, y, z);
  const double psi = free.singular_psi + correction;
  if (!(psi > 0.0) || !std::isfinite(psi))
    return false;
  output = ADMState{};
  const double psi2 = psi * psi;
  const double psi4 = psi2 * psi2;
  output.lapse = 1.0 / psi2;
  for (int axis = 0; axis < 3; ++axis)
    output.metric[SpatialSymmetricComponent(axis, axis)] = psi4;
  for (int component = 0; component < 6; ++component)
    output.extrinsic[component] = free.conformal_extrinsic[component] / psi2;
  return true;
}

std::vector<double>
PunctureEndADMMasses(const std::vector<Puncture> &punctures,
                     const SpectralInterpolator &interpolator) {
  std::vector<double> masses(punctures.size(), 0.0);
  for (std::size_t i = 0; i < punctures.size(); ++i) {
    double regular_part =
        1.0 + interpolator.RegularCorrection(punctures[i].center[0],
                                             punctures[i].center[1],
                                             punctures[i].center[2]);
    for (std::size_t j = 0; j < punctures.size(); ++j) {
      if (i == j)
        continue;
      double distance_squared = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double difference =
            punctures[i].center[axis] - punctures[j].center[axis];
        distance_squared += difference * difference;
      }
      if (!(distance_squared > 0.0)) {
        masses[i] = std::numeric_limits<double>::quiet_NaN();
        break;
      }
      regular_part += punctures[j].mass / (2.0 * std::sqrt(distance_squared));
    }
    if (std::isfinite(masses[i]))
      masses[i] = punctures[i].mass * regular_part;
  }
  return masses;
}

HamiltonianSolution SolveTargetADMMasses(
    std::vector<Puncture> &punctures, const std::vector<double> &target_masses,
    const HamiltonianSolveOptions &solve_options, const double mass_tolerance,
    const int maximum_mass_iterations) {
  HamiltonianSolution solution{};
  if (punctures.empty() || punctures.size() != target_masses.size() ||
      !(mass_tolerance > 0.0) || maximum_mass_iterations < 1)
    return solution;
  for (const double target : target_masses)
    if (!(target > 0.0) || !std::isfinite(target))
      return solution;

  for (int iteration = 0; iteration < maximum_mass_iterations; ++iteration) {
    solution = SolveHamiltonianConstraint(punctures, solve_options);
    if (!solution.converged)
      return solution;
    const SpectralInterpolator interpolator(solution);
    const auto measured = PunctureEndADMMasses(punctures, interpolator);
    double maximum_relative_error = 0.0;
    for (std::size_t i = 0; i < punctures.size(); ++i) {
      if (!(measured[i] > 0.0) || !std::isfinite(measured[i])) {
        solution.converged = false;
        return solution;
      }
      maximum_relative_error = std::max(
          maximum_relative_error,
          std::fabs(measured[i] - target_masses[i]) / target_masses[i]);
    }
    if (maximum_relative_error <= mass_tolerance)
      return solution;
    for (std::size_t i = 0; i < punctures.size(); ++i) {
      const double scaled = punctures[i].mass * target_masses[i] / measured[i];
      punctures[i].mass = 0.25 * punctures[i].mass + 0.75 * scaled;
    }
  }
  solution.converged = false;
  return solution;
}

} // namespace pangu::nr::puncture
