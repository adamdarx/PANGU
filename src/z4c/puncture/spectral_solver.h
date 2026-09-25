#ifndef PANGU_Z4C_PUNCTURE_SPECTRAL_SOLVER_H_
#define PANGU_Z4C_PUNCTURE_SPECTRAL_SOLVER_H_

#include <cstddef>
#include <vector>

#include "z4c/puncture/puncture_data.h"

namespace pangu::nr::puncture {

struct SpectralGridOptions {
  int points = 24;
  double scale = 2.0;
};

struct HamiltonianSolveOptions {
  SpectralGridOptions grid{};
  double nonlinear_tolerance = 1.0e-10;
  double linear_tolerance = 1.0e-8;
  int maximum_newton_iterations = 12;
  int maximum_linear_iterations = 400;
  int maximum_line_search_iterations = 12;
};

struct HamiltonianSolution {
  SpectralGridOptions grid{};
  std::vector<double> regular_correction;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  int newton_iterations = 0;
  int linear_iterations = 0;
  bool converged = false;
};

// Tensor-product rational Chebyshev grid on R^3.  Each computational
// coordinate q in [-1,1] is mapped by x=L*q/(1-q^2), so the Lobatto end
// points represent spatial infinity and homogeneous asymptotic data are
// imposed there without truncating the physical domain.
class CompactifiedSpectralGrid {
 public:
  explicit CompactifiedSpectralGrid(SpectralGridOptions options);

  int points() const { return points_; }
  double scale() const { return scale_; }
  std::size_t size() const;
  std::size_t Index(int i, int j, int k) const;
  bool IsBoundary(int i, int j, int k) const;

  double ComputationalCoordinate(int i) const;
  double PhysicalCoordinate(int i) const;

  // Apply the flat physical-space Laplacian. At compactified infinity the
  // returned value is the input itself, encoding homogeneous Dirichlet data.
  void ApplyLaplacian(const std::vector<double>& input,
                      std::vector<double>& output) const;

  double LaplacianDiagonal(int i, int j, int k) const;

 private:
  int points_ = 0;
  double scale_ = 0.0;
  std::vector<double> computational_;
  std::vector<double> physical_;
  std::vector<double> first_derivative_;
  std::vector<double> second_derivative_;
  std::vector<double> first_metric_;
  std::vector<double> second_metric_;
};

HamiltonianSolution SolveHamiltonianConstraint(const std::vector<Puncture>& punctures,
                                                const HamiltonianSolveOptions& options);

} // namespace pangu::nr::puncture

#endif
