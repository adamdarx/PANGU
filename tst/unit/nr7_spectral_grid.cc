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
        const double x = grid.PhysicalCoordinate(i);
        const double y = grid.PhysicalCoordinate(j);
        const double z = grid.PhysicalCoordinate(k);
        field[grid.Index(i, j, k)] = std::exp(-(x * x + y * y + z * z));
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
        const double x = grid.PhysicalCoordinate(i);
        const double y = grid.PhysicalCoordinate(j);
        const double z = grid.PhysicalCoordinate(k);
        const double radius_squared = x * x + y * y + z * z;
        const double exact = (4.0 * radius_squared - 6.0) * std::exp(-radius_squared);
        maximum_error = std::max(maximum_error, std::abs(laplacian[index] - exact));
      }
  constexpr double tolerance = 2.0e-5;
  if (!std::isfinite(maximum_error) || maximum_error > tolerance || boundary_error != 0.0) {
    std::cerr << "compactified spectral Laplacian failure: error=" << maximum_error
              << " boundary=" << boundary_error << " tolerance=" << tolerance << '\n';
    return 1;
  }
  std::cout << "compactified spectral Laplacian PASS: error=" << maximum_error << '\n';
  return 0;
}
