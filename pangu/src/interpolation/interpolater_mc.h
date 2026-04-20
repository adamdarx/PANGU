// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/interpolation module defines interpolater_mc.h responsibilities for
// the Pangu runtime. It centers on basic_types to express core data flow, keep interfaces
// readable, and preserve predictable behavior across task coordination, recovery paths, and
// performance-sensitive execution.

#ifndef PANGU_SRC_INTERPOLATION_INTERPOLATERMC_H
#define PANGU_SRC_INTERPOLATION_INTERPOLATERMC_H
#include "basic_types.hpp"

KOKKOS_INLINE_FUNCTION
parthenon::Real InterpolateMc(const parthenon::Real left,
                              const parthenon::Real center,
                              const parthenon::Real right) {
  const parthenon::Real forward_diff = 2. * (center - left);
  const parthenon::Real backward_diff = 2. * (right - center);
  const parthenon::Real center_diff = 0.5 * (right - left);
  const parthenon::Real sign = forward_diff * backward_diff;

  if (sign <= 0.)
    return 0.;
  else {
    if (Kokkos::abs(forward_diff) < Kokkos::abs(backward_diff) &&
        Kokkos::abs(forward_diff) < Kokkos::abs(center_diff))
      return (forward_diff);
    else if (Kokkos::abs(backward_diff) < Kokkos::abs(center_diff))
      return (backward_diff);
    else
      return (center_diff);
  }
}

#endif  // PANGU_SRC_INTERPOLATION_INTERPOLATERMC_H
