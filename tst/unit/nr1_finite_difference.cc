#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>
#include <parthenon/parthenon.hpp>

#include "z4c/finite_difference.h"

namespace {

using Real = parthenon::Real;

template <class View> struct Accessor {
  View values;
  int k;
  int j;
  int i;

  KOKKOS_INLINE_FUNCTION Real operator()(const int dk, const int dj, const int di) const {
    return values(k + dk, j + dj, i + di);
  }
};

template <int Order, class View>
Real CheckOrder(const View& field, const int center, const Real inverse_x, const Real inverse_y,
                const Real inverse_z) {
  constexpr int kFirst = 0;
  constexpr int kSecond = 3;
  constexpr int kAdvectiveNegative = 6;
  constexpr int kAdvectivePositive = 9;
  constexpr int kMixed = 12;
  constexpr int kDissipation = 15;
  constexpr int kErrors = 18;
  const Real inverse_spacing[3]{inverse_x, inverse_y, inverse_z};
  const Real first_exact[3]{2.0, -3.0, 4.0};
  const Real second_exact[3]{1.0, -1.5, 2.5};
  const Real mixed_exact[3]{1.5, -2.0, 0.625};
  Kokkos::View<Real*> errors("NR-1 finite-difference errors", kErrors);
  Kokkos::parallel_for(
      "NR-1 polynomial operator checks", 1, KOKKOS_LAMBDA(const int) {
        const Accessor<View> value{field, center, center, center};
        for (int direction = 0; direction < 3; ++direction) {
          errors(kFirst + direction) =
              fabs(pangu::nr::fd::First<Order>(direction, inverse_spacing[direction], value) -
                   first_exact[direction]);
          errors(kSecond + direction) =
              fabs(pangu::nr::fd::Second<Order>(direction, inverse_spacing[direction], value) -
                   second_exact[direction]);
          errors(kAdvectiveNegative + direction) =
              fabs(pangu::nr::fd::AdvectiveFirst<Order>(direction, inverse_spacing[direction], -0.7,
                                                        value) +
                   0.7 * first_exact[direction]);
          errors(kAdvectivePositive + direction) =
              fabs(pangu::nr::fd::AdvectiveFirst<Order>(direction, inverse_spacing[direction], 0.9,
                                                        value) -
                   0.9 * first_exact[direction]);
          errors(kDissipation + direction) = fabs(
              pangu::nr::fd::KreissOliger<Order>(direction, inverse_spacing[direction], value));
        }
        errors(kMixed) = fabs(pangu::nr::fd::MixedSecond<Order>(0, 1, inverse_x, inverse_y, value) -
                              mixed_exact[0]);
        errors(kMixed + 1) = fabs(
            pangu::nr::fd::MixedSecond<Order>(0, 2, inverse_x, inverse_z, value) - mixed_exact[1]);
        errors(kMixed + 2) = fabs(
            pangu::nr::fd::MixedSecond<Order>(1, 2, inverse_y, inverse_z, value) - mixed_exact[2]);
      });
  Kokkos::fence();
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
  Real maximum = 0.0;
  for (int error = 0; error < kErrors; ++error)
    maximum = std::max(maximum, host(error));
  return maximum;
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int return_code = 0;
  {
    constexpr int kCells = 9;
    constexpr int kCenter = kCells / 2;
    constexpr Real kDx = 0.07;
    constexpr Real kDy = 0.11;
    constexpr Real kDz = 0.13;
    Kokkos::View<Real***> field("NR-1 polynomial field", kCells, kCells, kCells);
    auto host = Kokkos::create_mirror_view(field);
    for (int k = 0; k < kCells; ++k) {
      const Real z = (k - kCenter) * kDz;
      for (int j = 0; j < kCells; ++j) {
        const Real y = (j - kCenter) * kDy;
        for (int i = 0; i < kCells; ++i) {
          const Real x = (i - kCenter) * kDx;
          host(k, j, i) = 1.0 + 2.0 * x - 3.0 * y + 4.0 * z + 0.5 * x * x - 0.75 * y * y +
                          1.25 * z * z + 1.5 * x * y - 2.0 * x * z + 0.625 * y * z;
        }
      }
    }
    Kokkos::deep_copy(field, host);
    const Real second_order = CheckOrder<2>(field, kCenter, 1.0 / kDx, 1.0 / kDy, 1.0 / kDz);
    const Real fourth_order = CheckOrder<4>(field, kCenter, 1.0 / kDx, 1.0 / kDy, 1.0 / kDz);
    const Real sixth_order = CheckOrder<6>(field, kCenter, 1.0 / kDx, 1.0 / kDy, 1.0 / kDz);
    const Real maximum = std::max({second_order, fourth_order, sixth_order});
    const Real tolerance = 4096.0 * std::numeric_limits<Real>::epsilon();
    if (!std::isfinite(maximum) || maximum > tolerance) {
      std::cerr << "NR-1 finite-difference polynomial failure: order2=" << second_order
                << " order4=" << fourth_order << " order6=" << sixth_order
                << " tolerance=" << tolerance << '\n';
      return_code = 1;
    } else {
      std::cout << "NR-1 finite-difference polynomial PASS: order2=" << second_order
                << " order4=" << fourth_order << " order6=" << sixth_order << '\n';
    }
  }
  Kokkos::finalize();
  return return_code;
}
