#pragma once

#include <cmath>

#include <basic_types.hpp>
#include <Kokkos_Core.hpp>

namespace Minkowski {

KOKKOS_INLINE_FUNCTION
void CalculatePhysicalCoordinates(const parthenon::Real x[4],
                                  parthenon::Real y[4]) {
    y[0] = x[0];
    y[1] = x[1];
    y[2] = x[2];
    y[3] = x[3];
}

KOKKOS_INLINE_FUNCTION
void CalculateMetricTensor(const parthenon::Real x[4],
                           parthenon::Real gcov[4][4]) {
    (void)x;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            gcov[i][j] = (i == j) * (1 - 2 * (i == 0));
        }
    }
}

} // namespace Minkowski
