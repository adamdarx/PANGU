#include "z4c/puncture/interpolation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pangu::nr::puncture {

SpectralInterpolator::SpectralInterpolator(const HamiltonianSolution& solution,
                                         const int stencil_points)
    : grid_(solution.grid), correction_(solution.regular_correction),
      stencil_points_(stencil_points) {
  if (!solution.converged) throw std::invalid_argument("cannot interpolate an unconverged puncture solution");
  if (correction_.size() != grid_.size())
    throw std::invalid_argument("puncture solution and spectral grid sizes differ");
  if ((stencil_points_ != 4 && stencil_points_ != 6 && stencil_points_ != 8) ||
      stencil_points_ > grid_.points())
    throw std::invalid_argument("puncture interpolation requires 4, 6, or 8 points, no more than the spectral grid");
}

SpectralInterpolator::Stencil SpectralInterpolator::BuildStencil(const double coordinate) const {
  const int n = grid_.points();
  // Invert x=L*q/(1-q^2) without cancellation or squaring a large x.
  // Interpolate in the bounded coordinate of the elliptic grid, including
  // u=0 at infinity. Cartesian cubic extrapolation beyond the last finite
  // collocation node can instead grow as x^3 and make psi negative.
  double q;
  if (std::abs(coordinate) <= grid_.scale()) {
    const double ratio = coordinate / grid_.scale();
    q = 2.0 * ratio / (1.0 + std::hypot(1.0, 2.0 * ratio));
  } else {
    const double half_inverse = 0.5 * (grid_.scale() / coordinate);
    q = std::copysign(1.0 / (std::hypot(1.0, half_inverse) +
                              std::abs(half_inverse)), coordinate);
  }
  int upper = 0;
  while (upper < n - 2 && grid_.ComputationalCoordinate(upper + 1) > q) ++upper;
  Stencil stencil{};
  stencil.start = std::clamp(upper - (stencil_points_ / 2 - 1), 0,
                            n - stencil_points_);
  for (int local = 0; local < stencil_points_; ++local) {
    const double node = grid_.ComputationalCoordinate(stencil.start + local);
    double weight = 1.0;
    for (int other = 0; other < stencil_points_; ++other) {
      if (other == local) continue;
      const double other_node = grid_.ComputationalCoordinate(stencil.start + other);
      weight *= (q - other_node) / (node - other_node);
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
  for (int local_k = 0; local_k < stencil_points_; ++local_k)
    for (int local_j = 0; local_j < stencil_points_; ++local_j)
      for (int local_i = 0; local_i < stencil_points_; ++local_i)
        value += sx.weight[local_i] * sy.weight[local_j] * sz.weight[local_k] *
                 correction_[grid_.Index(sx.start + local_i, sy.start + local_j,
                                         sz.start + local_k)];
  return value;
}

} // namespace pangu::nr::puncture
