#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

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
        const double x = grid.ComputationalCoordinate(i);
        const double y = grid.ComputationalCoordinate(j);
        const double z = grid.ComputationalCoordinate(k);
        solution.regular_correction[grid.Index(i, j, k)] =
            (1-x*x)*(1-y*y)*(1-z*z)*(0.25+0.125*x-0.0625*y+0.03125*z);
      }
  const SpectralInterpolator interpolator(solution);
  const double points[][3] = {{0.1, -0.2, 0.3}, {-0.7, 0.4, -0.1},
      {1.0, -0.8, 0.6}, {-1.2, 0.9, -0.5}, {0.,0.,0.},
      {-214.583,-214.583,-214.583}, {200.,0.,0.}, {400.,400.,400.},
      {1.e6,-1.e6,0.}, {std::numeric_limits<double>::max(),0.,0.}};
  double maximum_error = 0.0;
  for (const auto& point : points) {
    double q[3];
    for (int axis=0; axis<3; ++axis) {
      const long double ratio = static_cast<long double>(point[axis])/grid.scale();
      q[axis] = 2*ratio/(1+std::sqrt(1+4*ratio*ratio));
    }
    const double exact = (1-q[0]*q[0])*(1-q[1]*q[1])*(1-q[2]*q[2])*
        (0.25+0.125*q[0]-0.0625*q[1]+0.03125*q[2]);
    const double actual = interpolator.RegularCorrection(point[0], point[1], point[2]);
    if (!std::isfinite(actual)) {
      std::cerr << "nonfinite far-field interpolation\n";
      return 1;
    }
    maximum_error = std::max(
        maximum_error, std::abs(actual - exact));
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
