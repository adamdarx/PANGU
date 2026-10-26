#ifndef PANGU_Z4C_HORIZON_CONVERGENCE_H_
#define PANGU_Z4C_HORIZON_CONVERGENCE_H_

#include <cmath>

namespace pangu::nr {

// Mass stationarity is necessary but not sufficient: near a minimal surface
// area is stationary even if the expansion is nonzero. The residual is made
// dimensionless with the input compact-object mass, not the measured area.
inline bool HorizonConverged(const double previous_mass, const double mass,
                             const double rms_expansion, const double mass_scale,
                             const int iteration, const double mass_tolerance,
                             const double expansion_tolerance, const bool require_expansion) {
  if (iteration < 1 || !std::isfinite(previous_mass) || !std::isfinite(mass) ||
      !std::isfinite(rms_expansion) || !std::isfinite(mass_scale) ||
      !std::isfinite(mass_tolerance) || !std::isfinite(expansion_tolerance) ||
      !(mass > 0.0) || !(previous_mass > 0.0) || !(mass_scale > 0.0) ||
      !(rms_expansion >= 0.0) || !(mass_tolerance > 0.0) || !(expansion_tolerance > 0.0))
    return false;
  return std::abs(previous_mass - mass) < mass_tolerance &&
         (!require_expansion || mass_scale * rms_expansion <= expansion_tolerance);
}

} // namespace pangu::nr

#endif
