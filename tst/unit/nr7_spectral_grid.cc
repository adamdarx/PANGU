#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "z4c/puncture/spectral_solver.h"

int main() {
  using pangu::nr::puncture::CompactifiedSpectralGrid;
  using pangu::nr::puncture::SpectralGridOptions;
  const CompactifiedSpectralGrid grid(SpectralGridOptions{32, 2.0});
  std::vector<double> field(grid.size(), 0.0);
  for (int k = 0; k < grid.points(); ++k)
    for (int j = 0; j < grid.points(); ++j)
      for (int i = 0; i < grid.points(); ++i) {
        if (grid.IsBoundary(i, j, k)) continue;
        const double qx = grid.ComputationalCoordinate(i);
        const double qy = grid.ComputationalCoordinate(j);
        const double qz = grid.ComputationalCoordinate(k);
        field[grid.Index(i, j, k)] = std::pow(1.0 - qx * qx, 2) *
                                             std::pow(1.0 - qy * qy, 2) *
                                             std::pow(1.0 - qz * qz, 2);
      }

  std::vector<double> laplacian;
  grid.ApplyLaplacian(field, laplacian);
  double maximum_error = 0.0;
  double boundary_error = 0.0;
  for (int k = 0; k < grid.points(); ++k)
    for (int j = 0; j < grid.points(); ++j)
      for (int i = 0; i < grid.points(); ++i) {
        const std::size_t index = grid.Index(i, j, k);
        if (grid.IsBoundary(i, j, k)) {
          boundary_error = std::max(boundary_error, std::abs(laplacian[index]));
          continue;
        }
        if (std::abs(grid.ComputationalCoordinate(i)) > 0.8 ||
            std::abs(grid.ComputationalCoordinate(j)) > 0.8 ||
            std::abs(grid.ComputationalCoordinate(k)) > 0.8)
          continue;
        const double q[3] = {grid.ComputationalCoordinate(i),
                             grid.ComputationalCoordinate(j),
                             grid.ComputationalCoordinate(k)};
        double exact = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
          const double one_minus = 1.0 - q[axis] * q[axis];
          const double one_plus = 1.0 + q[axis] * q[axis];
          const double metric = one_minus * one_minus / (grid.scale() * one_plus);
          const double metric_derivative =
              -2.0 * q[axis] * one_minus * (3.0 + q[axis] * q[axis]) /
              (grid.scale() * one_plus * one_plus);
          const double first = -4.0 * q[axis] * one_minus;
          const double second = -4.0 + 12.0 * q[axis] * q[axis];
          double transverse = 1.0;
          for (int other = 0; other < 3; ++other)
            if (other != axis)
              transverse *= std::pow(1.0 - q[other] * q[other], 2);
          exact += (metric * metric * second + metric * metric_derivative * first) * transverse;
        }
        maximum_error = std::max(maximum_error, std::abs(laplacian[index] - exact));
      }
  constexpr double tolerance = 2.0e-11;
  if (!std::isfinite(maximum_error) || maximum_error > tolerance || boundary_error != 0.0) {
    std::cerr << "compactified spectral Laplacian failure: error=" << maximum_error
              << " boundary=" << boundary_error << " tolerance=" << tolerance << '\n';
    return 1;
  }
  std::cout << "compactified spectral Laplacian PASS: error=" << maximum_error << '\n';
  return 0;
}
