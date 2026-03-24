#pragma once

#include <cmath>

#include <basic_types.hpp>
#include <Kokkos_Core.hpp>

namespace Kerr {

KOKKOS_INLINE_FUNCTION
void CalculatePhysicalCoordinates(const parthenon::Real x[4],
                                  parthenon::Real y[4],
                                  const parthenon::Real h,
                                  const parthenon::Real a) {
    (void)a;
    y[0] = x[0];
    y[1] = Kokkos::exp(x[1]);
    y[2] = M_PI_2 * (x[2] + 1.0) +
           0.5 * h * Kokkos::sin(M_PI * (x[2] + 1.0));
    y[3] = x[3];
}

KOKKOS_INLINE_FUNCTION
void CalculateMetricTensor(const parthenon::Real x[4],
                           parthenon::Real gcov[4][4],
                           const parthenon::Real h,
                           const parthenon::Real a) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            gcov[i][j] = 0.0;
        }
    }

    parthenon::Real y[4];
    CalculatePhysicalCoordinates(x, y, h, a);

    const parthenon::Real r = y[1];
    const parthenon::Real theta = y[2];
    const parthenon::Real cth = Kokkos::cos(theta);
    const parthenon::Real sth = Kokkos::sin(theta);
    const parthenon::Real sigma = r * r + a * a * cth * cth;
    const parthenon::Real two_r_over_sigma = 2.0 * r / sigma;

    gcov[0][0] = -1.0 + two_r_over_sigma;
    gcov[0][1] = two_r_over_sigma;
    gcov[0][3] = -two_r_over_sigma * a * sth * sth;

    gcov[1][0] = gcov[0][1];
    gcov[1][1] = 1.0 + two_r_over_sigma;
    gcov[1][3] = -a * (1.0 + two_r_over_sigma) * sth * sth;

    gcov[2][2] = sigma;

    gcov[3][0] = gcov[0][3];
    gcov[3][1] = gcov[1][3];
    gcov[3][3] = sth * sth * (sigma + a * a * sth * sth * (1.0 + two_r_over_sigma));
}

} // namespace Kerr
