#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "z4c/puncture/spectral_solver.h"

int main() {
  using namespace pangu::nr::puncture;
  HamiltonianSolveOptions options{};
  options.grid = SpectralGridOptions{24, 2.0};
  options.nonlinear_tolerance = 1.0e-8;
  options.linear_tolerance = 1.0e-9;
  options.maximum_newton_iterations = 20;
  options.maximum_linear_iterations = 800;

  Puncture stationary{};
  stationary.mass = 1.0;
  const auto trivial = SolveHamiltonianConstraint({stationary}, options);
  if (!trivial.converged || trivial.final_residual != 0.0 ||
      trivial.newton_iterations != 0) {
    std::cerr << "stationary puncture did not preserve u=0\n";
    return 1;
  }

  Puncture spinning = stationary;
  spinning.spin[2] = 0.05;
  const auto solution = SolveHamiltonianConstraint({spinning}, options);
  const double maximum_correction = *std::max_element(
      solution.regular_correction.begin(), solution.regular_correction.end());
  const double minimum_correction = *std::min_element(
      solution.regular_correction.begin(), solution.regular_correction.end());
  const auto maximum_iterator = std::max_element(
      solution.regular_correction.begin(), solution.regular_correction.end());
  const std::size_t maximum_index =
      maximum_iterator - solution.regular_correction.begin();
  if (!solution.converged || !std::isfinite(solution.final_residual) ||
      solution.final_residual > options.nonlinear_tolerance ||
      maximum_correction <= 0.0 ||
      minimum_correction < -0.01 * maximum_correction) {
    std::cerr << "spinning puncture Hamiltonian solve failure: converged="
              << solution.converged << " initial=" << solution.initial_residual
              << " final=" << solution.final_residual
              << " newton=" << solution.newton_iterations
              << " linear=" << solution.linear_iterations
              << " u_min=" << minimum_correction
              << " u_max=" << maximum_correction
              << " max_index=" << maximum_index << '\n';
    return 1;
  }
  Puncture first{};
  first.mass = 0.483;
  first.center[0] = 3.257;
  first.momentum[1] = -0.133;
  Puncture second = first;
  second.center[0] = -3.257;
  second.momentum[1] = 0.133;
  auto binary_options = options;
  binary_options.grid.points = 30;
  binary_options.nonlinear_tolerance = 1.0e-10;
  binary_options.linear_tolerance = 1.0e-11;
  binary_options.maximum_linear_iterations = 2000;
  const auto binary =
      SolveHamiltonianConstraint({first, second}, binary_options);
  if (!binary.converged || !std::isfinite(binary.final_residual) ||
      binary.final_residual > binary_options.nonlinear_tolerance) {
    std::cerr << "binary puncture Hamiltonian solve failure: converged="
              << binary.converged << " initial=" << binary.initial_residual
              << " final=" << binary.final_residual
              << " newton=" << binary.newton_iterations
              << " linear=" << binary.linear_iterations << '\n';
    return 1;
  }
  std::cout << "native Hamiltonian solve PASS: initial="
            << solution.initial_residual << " final=" << solution.final_residual
            << " newton=" << solution.newton_iterations
            << " linear=" << solution.linear_iterations
            << " u_max=" << maximum_correction << " max_index=" << maximum_index
            << " binary_final=" << binary.final_residual << '\n';
  return 0;
}
