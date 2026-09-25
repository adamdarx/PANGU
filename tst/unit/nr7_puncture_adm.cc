#include <algorithm>
#include <cmath>
#include <iostream>

#include "z4c/puncture/initializer.h"

int main() {
  using namespace pangu::nr;
  using namespace pangu::nr::puncture;
  Puncture body{};
  body.mass = 1.0;
  body.momentum[0] = 0.03;
  body.spin[2] = 0.05;
  HamiltonianSolveOptions options{};
  options.grid = SpectralGridOptions{20, 2.0};
  options.nonlinear_tolerance = 1.0e-8;
  options.maximum_linear_iterations = 800;
  const HamiltonianSolution solution =
      SolveHamiltonianConstraint({body}, options);
  if (!solution.converged) {
    std::cerr << "ADM oracle could not obtain puncture solution\n";
    return 1;
  }
  const SpectralInterpolator interpolator(solution);
  ADMState adm{};
  if (!BuildPunctureADM(1.0, 0.5, -0.25, {body}, interpolator, adm)) {
    std::cerr << "ADM puncture construction failed\n";
    return 1;
  }
  const double inverse_diagonal = 1.0 / adm.metric[0];
  const double trace = inverse_diagonal *
                       (adm.extrinsic[0] + adm.extrinsic[3] + adm.extrinsic[5]);
  const double isotropy = std::max(
      {std::abs(adm.metric[0] - adm.metric[3]),
       std::abs(adm.metric[0] - adm.metric[5]), std::abs(adm.metric[1]),
       std::abs(adm.metric[2]), std::abs(adm.metric[4])});
  if (!(adm.lapse > 0.0) || !std::isfinite(trace) ||
      std::abs(trace) > 2.0e-14 || isotropy > 2.0e-14) {
    std::cerr << "ADM puncture oracle failure: lapse=" << adm.lapse
              << " trace=" << trace << " isotropy=" << isotropy << '\n';
    return 1;
  }
  Puncture first{};
  first.mass = 0.5;
  first.center[0] = 2.0;
  Puncture second = first;
  second.center[0] = -2.0;
  std::vector<Puncture> binary{first, second};
  const std::vector<double> targets{0.55, 0.55};
  const auto tuned =
      SolveTargetADMMasses(binary, targets, options, 1.0e-10, 24);
  if (!tuned.converged) {
    std::cerr << "target ADM mass solve did not converge\n";
    return 1;
  }
  const SpectralInterpolator tuned_interpolator(tuned);
  const auto measured = PunctureEndADMMasses(binary, tuned_interpolator);
  const double mass_error = std::max(std::abs(measured[0] - targets[0]),
                                     std::abs(measured[1] - targets[1]));
  if (mass_error > 1.0e-10) {
    std::cerr << "target ADM mass mismatch: " << mass_error << '\n';
    return 1;
  }
  std::cout << "native puncture ADM PASS: lapse=" << adm.lapse
            << " trace=" << trace << " bare_mass=" << binary[0].mass
            << " target_error=" << mass_error << '\n';
  return 0;
}
