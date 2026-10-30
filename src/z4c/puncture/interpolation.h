#ifndef PANGU_Z4C_PUNCTURE_INTERPOLATION_H_
#define PANGU_Z4C_PUNCTURE_INTERPOLATION_H_

#include "z4c/puncture/spectral_solver.h"

namespace pangu::nr::puncture {

// Local polynomial interpolation in compactified coordinates, including
// the homogeneous asymptotic boundary values at q=+/-1. Finite physical
// coordinates always map inside this grid; far-field extrapolation is avoided.
class SpectralInterpolator {
 public:
  explicit SpectralInterpolator(const HamiltonianSolution& solution,
                                int stencil_points = 4);

  double RegularCorrection(double x, double y, double z) const;

 private:
  struct Stencil {
    int start = 0;
    double weight[8]{};
  };

  Stencil BuildStencil(double coordinate) const;

  CompactifiedSpectralGrid grid_;
  std::vector<double> correction_;
  int stencil_points_;
};

} // namespace pangu::nr::puncture

#endif
