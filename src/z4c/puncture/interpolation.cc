#include "z4c/puncture/interpolation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pangu::nr::puncture {

SpectralInterpolator::SpectralInterpolator(const HamiltonianSolution& solution)
    : grid_(solution.grid), correction_(solution.regular_correction) {
  if (!solution.converged) throw std::invalid_argument("cannot interpolate an unconverged puncture solution");
  if (correction_.size() != grid_.size())
    throw std::invalid_argument("puncture solution and spectral grid sizes differ");
}

SpectralInterpolator::Stencil SpectralInterpolator::BuildStencil(const double coordinate) const {
  const int n = grid_.points();
  int upper = 1;
  while (upper < n - 2 && grid_.PhysicalCoordinate(upper + 1) > coordinate) ++upper;
  Stencil stencil{};
  stencil.start = std::clamp(upper - 1, 1, n - 5);
  for (int local = 0; local < 4; ++local) {
    const double node = grid_.PhysicalCoordinate(stencil.start + local);
    double weight = 1.0;
    for (int other = 0; other < 4; ++other) {
      if (other == local) continue;
      const double other_node = grid_.PhysicalCoordinate(stencil.start + other);
      weight *= (coordinate - other_node) / (node - other_node);
    }
    stencil.weight[local] = weight;
  }
  return stencil;
}

double SpectralInterpolator::RegularCorrection(const double x, const double y,
                                               const double z) const {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
    throw std::invalid_argument("puncture interpolation coordinates must be finite");
  const Stencil sx = BuildStencil(x);
  const Stencil sy = BuildStencil(y);
  const Stencil sz = BuildStencil(z);
  double value = 0.0;
  for (int local_k = 0; local_k < 4; ++local_k)
    for (int local_j = 0; local_j < 4; ++local_j)
      for (int local_i = 0; local_i < 4; ++local_i)
        value += sx.weight[local_i] * sy.weight[local_j] * sz.weight[local_k] *
                 correction_[grid_.Index(sx.start + local_i, sy.start + local_j,
                                         sz.start + local_k)];
  return value;
}

} // namespace pangu::nr::puncture
