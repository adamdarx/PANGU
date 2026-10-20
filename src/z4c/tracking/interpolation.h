#pragma once

namespace pangu::nr::tracking {

// Host expressions from AthenaK CellCenterX/LagrangeInterpolator. Algebraic
// rewrites and device FMA contraction change tiny stationary-tracker motion,
// which can change strict tracker AMR at tangent blocks.
template <class Real>
Real InterpolationCellCenter(int index, int cells, Real lower, Real upper) {
  const Real x = (static_cast<Real>(index) + 0.5) / static_cast<Real>(cells);
  return (x * upper - x * lower) - (0.5 * upper - 0.5 * lower) +
         (0.5 * lower + 0.5 * upper);
}

template <class Real, class Weights, class Samples>
Real InterpolateHost(int points, int weight_stride, const Weights &weights,
                     const Samples &samples) {
  Real value = 0.;
  for (int i = 0; i < points; ++i)
    for (int j = 0; j < points; ++j)
      for (int k = 0; k < points; ++k) {
        const Real weight = weights[i] * weights[weight_stride + j] *
                            weights[2 * weight_stride + k];
        value += weight * samples((i * points + j) * points + k);
      }
  return value;
}

} // namespace pangu::nr::tracking
