#include <algorithm>
#include <cmath>
#include <iostream>

#include "z4c/puncture/interpolation.h"

int main() {
  using namespace pangu::nr::puncture;
  HamiltonianSolution solution{};
  solution.grid = SpectralGridOptions{16, 2.0};
  solution.converged = true;
  const CompactifiedSpectralGrid grid(solution.grid);
  solution.regular_correction.resize(grid.size());
  for (int k = 0; k < grid.points(); ++k)
    for (int j = 0; j < grid.points(); ++j)
      for (int i = 0; i < grid.points(); ++i) {
        if (grid.IsBoundary(i, j, k)) continue;
        const double x = grid.PhysicalCoordinate(i);
        const double y = grid.PhysicalCoordinate(j);
        const double z = grid.PhysicalCoordinate(k);
        solution.regular_correction[grid.Index(i, j, k)] =
            0.25 + x + 0.5 * y * y - 0.125 * z * z * z;
      }
  const SpectralInterpolator interpolator(solution);
  const double points[4][3] = {{0.1, -0.2, 0.3}, {-0.7, 0.4, -0.1},
                               {1.0, -0.8, 0.6}, {-1.2, 0.9, -0.5}};
  double maximum_error = 0.0;
  for (const auto& point : points) {
    const double exact =
        0.25 + point[0] + 0.5 * point[1] * point[1] - 0.125 * std::pow(point[2], 3);
    maximum_error = std::max(
        maximum_error,
        std::abs(interpolator.RegularCorrection(point[0], point[1], point[2]) - exact));
  }
  constexpr double tolerance = 2.0e-12;
  if (!std::isfinite(maximum_error) || maximum_error > tolerance) {
    std::cerr << "puncture tricubic interpolation failure: error=" << maximum_error
              << " tolerance=" << tolerance << '\n';
    return 1;
  }
  std::cout << "puncture tricubic interpolation PASS: error=" << maximum_error << '\n';
  return 0;
}
