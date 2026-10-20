#pragma once

#include <cmath>
#include <limits>

#include <Kokkos_Core.hpp>

namespace pangu::nr::amr {

constexpr int kDerefine = -1;
constexpr int kSame = 0;
constexpr int kRefine = 1;

template <class Real, class Bounds, class Centers, class Radii, class Levels>
KOKKOS_INLINE_FUNCTION int TrackerTag(const Bounds& bounds, const int level,
                                      const int tracker_count, const Centers& centers,
                                      const Radii& radii, const Levels& target_levels) {
  int flag = kDerefine;
  for (int tracker = 0; tracker < tracker_count; ++tracker) {
    // Keep the operation order and strict comparison of AthenaK RefineTracker.
    // In particular, a roundoff-sized displacement must not be suppressed here.
    const Real x = centers[3 * tracker];
    const Real y = centers[3 * tracker + 1];
    const Real z = centers[3 * tracker + 2];
    const Real cx = fmax(bounds[0], fmin(x, bounds[1]));
    const Real cy = fmax(bounds[2], fmin(y, bounds[3]));
    const Real cz = fmax(bounds[4], fmin(z, bounds[5]));
    const Real distance_squared =
        (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
    const bool contained =
        centers[3 * tracker] >= bounds[0] && centers[3 * tracker] <= bounds[1] &&
        centers[3 * tracker + 1] >= bounds[2] && centers[3 * tracker + 1] <= bounds[3] &&
        centers[3 * tracker + 2] >= bounds[4] && centers[3 * tracker + 2] <= bounds[5];
    int tracker_flag = kDerefine;
    const Real radius_squared = radii[tracker] * radii[tracker];
    const bool intersects = distance_squared < radius_squared;
    // Keep the containment safety net, including the radius-zero case.
    if (intersects || contained) {
      if (target_levels[tracker] < 0 || level < target_levels[tracker])
        tracker_flag = kRefine;
      else if (level == target_levels[tracker])
        tracker_flag = kSame;
    }
    flag = tracker_flag > flag ? tracker_flag : flag;
  }
  return flag;
}

template <class Real> KOKKOS_INLINE_FUNCTION int ChiTag(const Real minimum, const Real threshold) {
  if (minimum < threshold)
    return kRefine;
  if (minimum > 1.25 * threshold)
    return kDerefine;
  return kSame;
}

template <class Real>
KOKKOS_INLINE_FUNCTION Real DchiIndicator(const Real dx, const Real dy, const Real dz,
                                          const Real chi, const bool normalized) {
  Real indicator = sqrt(dx * dx + dy * dy + dz * dz);
  if (normalized)
    indicator /= 2.0 * fmax(fabs(chi), static_cast<Real>(1.0e-30));
  return indicator;
}

template <class Real>
KOKKOS_INLINE_FUNCTION int DchiTag(const Real maximum, const Real upper, const Real lower) {
  if (maximum > upper)
    return kRefine;
  if (maximum < lower)
    return kDerefine;
  return kSame;
}

template <class Real, class Bounds, class Radii, class Levels>
KOKKOS_INLINE_FUNCTION int ApplyFixedRadii(int flag, const Bounds& bounds, const int level,
                                           const int region_count, const Radii& radii,
                                           const Levels& target_levels) {
  Real minimum_corner_radius_squared = std::numeric_limits<Real>::max();
  for (int corner = 0; corner < 8; ++corner) {
    const Real x = bounds[(corner & 1) ? 1 : 0];
    const Real y = bounds[(corner & 2) ? 3 : 2];
    const Real z = bounds[(corner & 4) ? 5 : 4];
    minimum_corner_radius_squared = fmin(minimum_corner_radius_squared, x * x + y * y + z * z);
  }
  for (int region = 0; region < region_count; ++region) {
    if (minimum_corner_radius_squared < radii[region] * radii[region]) {
      if (level < target_levels[region])
        flag = kRefine;
      else if (level == target_levels[region] && flag == kDerefine)
        flag = kSame;
    }
  }
  return flag;
}

} // namespace pangu::nr::amr
