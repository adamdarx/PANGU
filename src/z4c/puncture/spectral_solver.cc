#include "z4c/puncture/spectral_solver.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <algorithm>
#include <numeric>

#include "z4c/puncture/bowen_york.h"

namespace pangu::nr::puncture {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

} // namespace

CompactifiedSpectralGrid::CompactifiedSpectralGrid(const SpectralGridOptions options)
    : points_(options.points), scale_(options.scale) {
  if (points_ < 4) throw std::invalid_argument("puncture spectral grid requires at least 4 points");
  if (!(scale_ > 0.0) || !std::isfinite(scale_))
    throw std::invalid_argument("puncture spectral compactification scale must be positive");

  computational_.resize(points_);
  physical_.resize(points_);
  first_derivative_.assign(static_cast<std::size_t>(points_) * points_, 0.0);
  second_derivative_.assign(static_cast<std::size_t>(points_) * points_, 0.0);
  first_metric_.resize(points_);
  second_metric_.resize(points_);

  for (int i = 0; i < points_; ++i) {
    const double q = std::cos(kPi * static_cast<double>(i) / static_cast<double>(points_ - 1));
    computational_[i] = q;
    const double one_minus_q2 = std::max(0.0, 1.0 - q * q);
    physical_[i] = one_minus_q2 == 0.0
                       ? std::copysign(std::numeric_limits<double>::infinity(), q)
                       : scale_ * q / one_minus_q2;
    const double one_plus_q2 = 1.0 + q * q;
    first_metric_[i] = one_minus_q2 * one_minus_q2 / (scale_ * one_plus_q2);
    second_metric_[i] = -2.0 * q * one_minus_q2 * one_minus_q2 * one_minus_q2 *
                        (3.0 + q * q) /
                        (scale_ * scale_ * one_plus_q2 * one_plus_q2 * one_plus_q2);
  }

  // Chebyshev--Lobatto first-derivative matrix in descending node order.
  for (int i = 0; i < points_; ++i) {
    const double ci = (i == 0 || i == points_ - 1) ? 2.0 : 1.0;
    for (int j = 0; j < points_; ++j) {
      if (i == j) continue;
      const double cj = (j == 0 || j == points_ - 1) ? 2.0 : 1.0;
      const double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
      first_derivative_[static_cast<std::size_t>(i) * points_ + j] =
          sign * ci / (cj * (computational_[i] - computational_[j]));
    }
  }
  // Determine every diagonal from the exact derivative of a constant. This is
  // algebraically equivalent to the closed forms and is less sensitive to a
  // node-ordering convention.
  for (int i = 0; i < points_; ++i) {
    double off_diagonal_sum = 0.0;
    for (int j = 0; j < points_; ++j)
      if (i != j)
        off_diagonal_sum += first_derivative_[static_cast<std::size_t>(i) * points_ + j];
    first_derivative_[static_cast<std::size_t>(i) * points_ + i] = -off_diagonal_sum;
  }

  for (int i = 0; i < points_; ++i)
    for (int j = 0; j < points_; ++j)
      for (int k = 0; k < points_; ++k)
        second_derivative_[static_cast<std::size_t>(i) * points_ + j] +=
            first_derivative_[static_cast<std::size_t>(i) * points_ + k] *
            first_derivative_[static_cast<std::size_t>(k) * points_ + j];
}

std::size_t CompactifiedSpectralGrid::size() const {
  return static_cast<std::size_t>(points_) * points_ * points_;
}

std::size_t CompactifiedSpectralGrid::Index(const int i, const int j, const int k) const {
  return static_cast<std::size_t>(i) +
         static_cast<std::size_t>(points_) *
             (static_cast<std::size_t>(j) + static_cast<std::size_t>(points_) * k);
}

bool CompactifiedSpectralGrid::IsBoundary(const int i, const int j, const int k) const {
  return i == 0 || j == 0 || k == 0 || i == points_ - 1 || j == points_ - 1 ||
         k == points_ - 1;
}

double CompactifiedSpectralGrid::ComputationalCoordinate(const int i) const {
  return computational_.at(i);
}

double CompactifiedSpectralGrid::PhysicalCoordinate(const int i) const {
  return physical_.at(i);
}

void CompactifiedSpectralGrid::LocalLaplacianCoefficients(
    const int i, double &lower, double &diagonal, double &upper) const {
  lower = diagonal = upper = 0.0;
  if (i <= 0 || i >= points_ - 1) return;
  const double qm = computational_[i - 1];
  const double q = computational_[i];
  const double qp = computational_[i + 1];
  const double first_m = (q - qp) / ((qm - q) * (qm - qp));
  const double first_0 = (2.0 * q - qm - qp) / ((q - qm) * (q - qp));
  const double first_p = (q - qm) / ((qp - qm) * (qp - q));
  const double second_m = 2.0 / ((qm - q) * (qm - qp));
  const double second_0 = 2.0 / ((q - qm) * (q - qp));
  const double second_p = 2.0 / ((qp - qm) * (qp - q));
  const double first_metric = first_metric_[i];
  const double second_metric = second_metric_[i];
  lower = first_metric * first_metric * second_m + second_metric * first_m;
  diagonal = first_metric * first_metric * second_0 + second_metric * first_0;
  upper = first_metric * first_metric * second_p + second_metric * first_p;
}

void CompactifiedSpectralGrid::ApplyLaplacian(const std::vector<double>& input,
                                              std::vector<double>& output) const {
  if (input.size() != size()) throw std::invalid_argument("puncture spectral field has wrong size");
  output.assign(size(), 0.0);
  for (int k = 0; k < points_; ++k) {
    for (int j = 0; j < points_; ++j) {
      for (int i = 0; i < points_; ++i) {
        const std::size_t center = Index(i, j, k);
        if (IsBoundary(i, j, k)) {
          output[center] = input[center];
          continue;
        }
        double first[3]{};
        double second[3]{};
        for (int sample = 0; sample < points_; ++sample) {
          first[0] += first_derivative_[static_cast<std::size_t>(i) * points_ + sample] *
                      input[Index(sample, j, k)];
          second[0] += second_derivative_[static_cast<std::size_t>(i) * points_ + sample] *
                       input[Index(sample, j, k)];
          first[1] += first_derivative_[static_cast<std::size_t>(j) * points_ + sample] *
                      input[Index(i, sample, k)];
          second[1] += second_derivative_[static_cast<std::size_t>(j) * points_ + sample] *
                       input[Index(i, sample, k)];
          first[2] += first_derivative_[static_cast<std::size_t>(k) * points_ + sample] *
                      input[Index(i, j, sample)];
          second[2] += second_derivative_[static_cast<std::size_t>(k) * points_ + sample] *
                       input[Index(i, j, sample)];
        }
        output[center] = first_metric_[i] * first_metric_[i] * second[0] +
                         second_metric_[i] * first[0] +
                         first_metric_[j] * first_metric_[j] * second[1] +
                         second_metric_[j] * first[1] +
                         first_metric_[k] * first_metric_[k] * second[2] +
                         second_metric_[k] * first[2];
      }
    }
  }
}

double CompactifiedSpectralGrid::LaplacianDiagonal(const int i, const int j,
                                                    const int k) const {
  if (IsBoundary(i, j, k)) return 1.0;
  const auto diagonal = [&](const int index) {
    return first_metric_[index] * first_metric_[index] *
               second_derivative_[static_cast<std::size_t>(index) * points_ + index] +
           second_metric_[index] *
               first_derivative_[static_cast<std::size_t>(index) * points_ + index];
  };
  return diagonal(i) + diagonal(j) + diagonal(k);
}

namespace {

double InfinityNorm(const std::vector<double>& values) {
  double norm = 0.0;
  for (const double value : values) norm = std::max(norm, std::abs(value));
  return norm;
}

double Dot(const std::vector<double>& first, const std::vector<double>& second) {
  return std::inner_product(first.begin(), first.end(), second.begin(), 0.0);
}

struct HamiltonianSystem {
  const CompactifiedSpectralGrid& grid;
  std::vector<double> singular_psi;
  std::vector<double> extrinsic_squared;
  std::vector<unsigned char> puncture_node;

  void Residual(const std::vector<double>& correction, std::vector<double>& residual) const {
    grid.ApplyLaplacian(correction, residual);
    const int n = grid.points();
    for (int k = 1; k < n - 1; ++k)
      for (int j = 1; j < n - 1; ++j)
        for (int i = 1; i < n - 1; ++i) {
          const std::size_t index = grid.Index(i, j, k);
          if (puncture_node[index]) continue;
          const double psi = singular_psi[index] + correction[index];
          if (!(psi > 0.0) || !std::isfinite(psi)) {
            residual[index] = std::numeric_limits<double>::infinity();
            continue;
          }
          residual[index] += 0.125 * extrinsic_squared[index] * std::pow(psi, -7.0);
        }
  }

  void Jacobian(const std::vector<double>& correction, const std::vector<double>& input,
                std::vector<double>& output) const {
    grid.ApplyLaplacian(input, output);
    const int n = grid.points();
    for (int k = 1; k < n - 1; ++k)
      for (int j = 1; j < n - 1; ++j)
        for (int i = 1; i < n - 1; ++i) {
          const std::size_t index = grid.Index(i, j, k);
          if (puncture_node[index]) continue;
          const double psi = singular_psi[index] + correction[index];
          output[index] -=
              0.875 * extrinsic_squared[index] * std::pow(psi, -8.0) * input[index];
        }
  }

  double JacobianDiagonal(const std::vector<double>& correction, const int i, const int j,
                          const int k) const {
    const std::size_t index = grid.Index(i, j, k);
    double diagonal = grid.LaplacianDiagonal(i, j, k);
    if (!grid.IsBoundary(i, j, k) && !puncture_node[index]) {
      const double psi = singular_psi[index] + correction[index];
      diagonal -= 0.875 * extrinsic_squared[index] * std::pow(psi, -8.0);
    }
    return diagonal;
  }
};

void SolveTridiagonal(std::vector<double> &lower, std::vector<double> &diagonal,
                      std::vector<double> &upper, std::vector<double> &right_hand_side,
                      std::vector<double> &solution) {
  const int size = static_cast<int>(diagonal.size());
  for (int row = 1; row < size; ++row) {
    const double factor = lower[row] / diagonal[row - 1];
    diagonal[row] -= factor * upper[row - 1];
    right_hand_side[row] -= factor * right_hand_side[row - 1];
  }
  solution[size - 1] = right_hand_side[size - 1] / diagonal[size - 1];
  for (int row = size - 2; row >= 0; --row)
    solution[row] =
        (right_hand_side[row] - upper[row] * solution[row + 1]) / diagonal[row];
}

void ApplyLinePreconditioner(const HamiltonianSystem &system,
                             const std::vector<double> &correction,
                             const std::vector<double> &right_hand_side,
                             std::vector<double> &solution) {
  const int n = system.grid.points();
  solution.assign(right_hand_side.size(), 0.0);
  std::vector<double> lower(n), diagonal(n), upper(n), rhs(n), line(n);
  std::vector<double> local_lower(n), local_diagonal(n), local_upper(n);
  for (int index = 0; index < n; ++index)
    system.grid.LocalLaplacianCoefficients(
        index, local_lower[index], local_diagonal[index], local_upper[index]);
  std::vector<double> local_potential(right_hand_side.size(), 0.0);
  for (int k = 1; k < n - 1; ++k)
    for (int j = 1; j < n - 1; ++j)
      for (int i = 1; i < n - 1; ++i) {
        const std::size_t index = system.grid.Index(i, j, k);
        if (!system.puncture_node[index]) {
          const double psi = system.singular_psi[index] + correction[index];
          local_potential[index] =
              -0.875 * system.extrinsic_squared[index] * std::pow(psi, -8.0);
        }
      }

  // Alternating line Gauss--Seidel is deliberately much stronger than point
  // Jacobi on a compactified spectral grid.  TwoPunctures uses the same idea
  // for its finite-difference Jacobian preconditioner.
  constexpr int sweeps = 16;
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    for (int k = 1; k < n - 1; ++k)
      for (int j = 1; j < n - 1; ++j) {
        std::fill(lower.begin(), lower.end(), 0.0);
        std::fill(diagonal.begin(), diagonal.end(), 1.0);
        std::fill(upper.begin(), upper.end(), 0.0);
        rhs[0] = right_hand_side[system.grid.Index(0, j, k)];
        rhs[n - 1] = right_hand_side[system.grid.Index(n - 1, j, k)];
        for (int i = 1; i < n - 1; ++i) {
          const std::size_t index = system.grid.Index(i, j, k);
          lower[i] = local_lower[i];
          upper[i] = local_upper[i];
          diagonal[i] = local_diagonal[i] + local_diagonal[j] + local_diagonal[k] +
                        local_potential[index];
          rhs[i] = right_hand_side[index] -
                   local_lower[j] * solution[system.grid.Index(i, j - 1, k)] -
                   local_upper[j] * solution[system.grid.Index(i, j + 1, k)] -
                   local_lower[k] * solution[system.grid.Index(i, j, k - 1)] -
                   local_upper[k] * solution[system.grid.Index(i, j, k + 1)];
        }
        SolveTridiagonal(lower, diagonal, upper, rhs, line);
        for (int i = 0; i < n; ++i) solution[system.grid.Index(i, j, k)] = line[i];
      }

    for (int k = 1; k < n - 1; ++k)
      for (int i = 1; i < n - 1; ++i) {
        std::fill(lower.begin(), lower.end(), 0.0);
        std::fill(diagonal.begin(), diagonal.end(), 1.0);
        std::fill(upper.begin(), upper.end(), 0.0);
        rhs[0] = right_hand_side[system.grid.Index(i, 0, k)];
        rhs[n - 1] = right_hand_side[system.grid.Index(i, n - 1, k)];
        for (int j = 1; j < n - 1; ++j) {
          const std::size_t index = system.grid.Index(i, j, k);
          lower[j] = local_lower[j];
          upper[j] = local_upper[j];
          diagonal[j] = local_diagonal[i] + local_diagonal[j] + local_diagonal[k] +
                        local_potential[index];
          rhs[j] = right_hand_side[index] -
                   local_lower[i] * solution[system.grid.Index(i - 1, j, k)] -
                   local_upper[i] * solution[system.grid.Index(i + 1, j, k)] -
                   local_lower[k] * solution[system.grid.Index(i, j, k - 1)] -
                   local_upper[k] * solution[system.grid.Index(i, j, k + 1)];
        }
        SolveTridiagonal(lower, diagonal, upper, rhs, line);
        for (int j = 0; j < n; ++j) solution[system.grid.Index(i, j, k)] = line[j];
      }

    for (int j = 1; j < n - 1; ++j)
      for (int i = 1; i < n - 1; ++i) {
        std::fill(lower.begin(), lower.end(), 0.0);
        std::fill(diagonal.begin(), diagonal.end(), 1.0);
        std::fill(upper.begin(), upper.end(), 0.0);
        rhs[0] = right_hand_side[system.grid.Index(i, j, 0)];
        rhs[n - 1] = right_hand_side[system.grid.Index(i, j, n - 1)];
        for (int k = 1; k < n - 1; ++k) {
          const std::size_t index = system.grid.Index(i, j, k);
          lower[k] = local_lower[k];
          upper[k] = local_upper[k];
          diagonal[k] = local_diagonal[i] + local_diagonal[j] + local_diagonal[k] +
                        local_potential[index];
          rhs[k] = right_hand_side[index] -
                   local_lower[i] * solution[system.grid.Index(i - 1, j, k)] -
                   local_upper[i] * solution[system.grid.Index(i + 1, j, k)] -
                   local_lower[j] * solution[system.grid.Index(i, j - 1, k)] -
                   local_upper[j] * solution[system.grid.Index(i, j + 1, k)];
        }
        SolveTridiagonal(lower, diagonal, upper, rhs, line);
        for (int k = 0; k < n; ++k) solution[system.grid.Index(i, j, k)] = line[k];
      }
  }
}

bool SolveLinearized(const HamiltonianSystem& system, const std::vector<double>& correction,
                     const std::vector<double>& right_hand_side, const double tolerance,
                     const int maximum_iterations, std::vector<double>& solution,
                     int& iterations) {
  const std::size_t size = right_hand_side.size();
  solution.assign(size, 0.0);
  std::vector<double> residual = right_hand_side;
  std::vector<double> shadow = residual;
  std::vector<double> direction(size, 0.0), image(size, 0.0), intermediate(size, 0.0);
  std::vector<double> preconditioned(size, 0.0), preconditioned_intermediate(size, 0.0);
  std::vector<double> image_intermediate(size, 0.0);
  const double target = tolerance * std::max(1.0, InfinityNorm(right_hand_side));
  double rho_previous = 1.0;
  double alpha = 1.0;
  double omega = 1.0;
  for (iterations = 0; iterations < maximum_iterations; ++iterations) {
    const double rho = Dot(shadow, residual);
    if (!std::isfinite(rho) || std::abs(rho) <= std::numeric_limits<double>::min()) return false;
    const double beta = (rho / rho_previous) * (alpha / omega);
    for (std::size_t index = 0; index < size; ++index)
      direction[index] = residual[index] + beta * (direction[index] - omega * image[index]);
    ApplyLinePreconditioner(system, correction, direction, preconditioned);
    system.Jacobian(correction, preconditioned, image);
    const double denominator = Dot(shadow, image);
    if (!std::isfinite(denominator) ||
        std::abs(denominator) <= std::numeric_limits<double>::min())
      return false;
    alpha = rho / denominator;
    for (std::size_t index = 0; index < size; ++index)
      intermediate[index] = residual[index] - alpha * image[index];
    if (InfinityNorm(intermediate) <= target) {
      for (std::size_t index = 0; index < size; ++index)
        solution[index] += alpha * preconditioned[index];
      ++iterations;
      return true;
    }
    ApplyLinePreconditioner(system, correction, intermediate,
                            preconditioned_intermediate);
    system.Jacobian(correction, preconditioned_intermediate, image_intermediate);
    const double image_norm = Dot(image_intermediate, image_intermediate);
    if (!std::isfinite(image_norm) || image_norm <= std::numeric_limits<double>::min()) return false;
    omega = Dot(image_intermediate, intermediate) / image_norm;
    if (!std::isfinite(omega) || std::abs(omega) <= std::numeric_limits<double>::min()) return false;
    for (std::size_t index = 0; index < size; ++index) {
      solution[index] +=
          alpha * preconditioned[index] + omega * preconditioned_intermediate[index];
      residual[index] = intermediate[index] - omega * image_intermediate[index];
    }
    if (InfinityNorm(residual) <= target) {
      ++iterations;
      return true;
    }
    rho_previous = rho;
  }
  return false;
}

} // namespace

HamiltonianSolution SolveHamiltonianConstraint(const std::vector<Puncture>& punctures,
                                                const HamiltonianSolveOptions& options) {
  if (punctures.empty()) throw std::invalid_argument("at least one puncture is required");
  if (!(options.nonlinear_tolerance > 0.0) || !(options.linear_tolerance > 0.0) ||
      options.maximum_newton_iterations < 1 || options.maximum_linear_iterations < 1 ||
      options.maximum_line_search_iterations < 1)
    throw std::invalid_argument("invalid puncture Hamiltonian solver options");
  CompactifiedSpectralGrid grid(options.grid);
  HamiltonianSystem system{grid, std::vector<double>(grid.size(), 1.0),
                           std::vector<double>(grid.size(), 0.0),
                           std::vector<unsigned char>(grid.size(), 0)};
  const int n = grid.points();
  for (int k = 1; k < n - 1; ++k)
    for (int j = 1; j < n - 1; ++j)
      for (int i = 1; i < n - 1; ++i) {
        const std::size_t index = grid.Index(i, j, k);
        const double x = grid.PhysicalCoordinate(i);
        const double y = grid.PhysicalCoordinate(j);
        const double z = grid.PhysicalCoordinate(k);
        bool at_puncture = false;
        for (const auto& body : punctures) {
          const double dx = x - body.center[0];
          const double dy = y - body.center[1];
          const double dz = z - body.center[2];
          at_puncture = at_puncture || dx * dx + dy * dy + dz * dz < 1.0e-28;
        }
        if (at_puncture) {
          system.puncture_node[index] = 1;
          continue;
        }
        FreeData free{};
        if (!EvaluateFreeData(x, y, z, punctures.data(), static_cast<int>(punctures.size()), free))
          throw std::runtime_error("invalid Bowen--York free data on spectral grid");
        system.singular_psi[index] = free.singular_psi;
        system.extrinsic_squared[index] = free.conformal_extrinsic_squared;
      }

  HamiltonianSolution solution{};
  solution.grid = options.grid;
  solution.regular_correction.assign(grid.size(), 0.0);
  std::vector<double> residual;
  system.Residual(solution.regular_correction, residual);
  solution.initial_residual = InfinityNorm(residual);
  solution.final_residual = solution.initial_residual;
  if (solution.final_residual <= options.nonlinear_tolerance) {
    solution.converged = true;
    return solution;
  }

  for (int iteration = 0; iteration < options.maximum_newton_iterations; ++iteration) {
    std::vector<double> right_hand_side(residual.size());
    for (std::size_t index = 0; index < residual.size(); ++index)
      right_hand_side[index] = -residual[index];
    std::vector<double> step;
    int linear_iterations = 0;
    // Use an inexact Newton forcing term.  Solving the early Jacobian systems
    // to the final nonlinear tolerance wastes Krylov iterations and can cause
    // stagnation on compactified grids.  Tighten the linear solve naturally as
    // the nonlinear residual falls, while retaining the requested floor.
    const double linear_tolerance =
        std::max(options.linear_tolerance, 1.0e-3 * solution.final_residual);
    const bool linear_converged =
        SolveLinearized(system, solution.regular_correction, right_hand_side,
                        linear_tolerance, options.maximum_linear_iterations, step,
                        linear_iterations);
    solution.linear_iterations += linear_iterations;
    if (!linear_converged)
      break;

    bool accepted = false;
    double damping = 1.0;
    std::vector<double> candidate(solution.regular_correction.size());
    std::vector<double> candidate_residual;
    for (int line_search = 0; line_search < options.maximum_line_search_iterations;
         ++line_search) {
      for (std::size_t index = 0; index < candidate.size(); ++index)
        candidate[index] = solution.regular_correction[index] + damping * step[index];
      system.Residual(candidate, candidate_residual);
      const double candidate_norm = InfinityNorm(candidate_residual);
      if (std::isfinite(candidate_norm) && candidate_norm < solution.final_residual) {
        solution.regular_correction.swap(candidate);
        residual.swap(candidate_residual);
        solution.final_residual = candidate_norm;
        accepted = true;
        break;
      }
      damping *= 0.5;
    }
    solution.newton_iterations = iteration + 1;
    if (!accepted) break;
    if (solution.final_residual <= options.nonlinear_tolerance) {
      solution.converged = true;
      break;
    }
  }
  return solution;
}

} // namespace pangu::nr::puncture
