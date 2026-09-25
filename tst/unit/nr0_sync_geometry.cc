#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include <Kokkos_Core.hpp>
#include <parthenon/parthenon.hpp>

#include "geometry/metric/z4c.h"
#include "geometry/mode/sync.h"
#include "z4c/core/component_indices.h"

namespace {

using Real = parthenon::Real;

struct FieldPack {
  Kokkos::View<Real*****> values;

  KOKKOS_INLINE_FUNCTION Real& operator()(const int block, const int component, const int k,
                                          const int j, const int i) const {
    return values(block, component, k, j, i);
  }
};

struct Coordinates {};

template <class View>
void SetSymmetric(View& field, const int cell, const Real xx, const Real xy, const Real xz,
                  const Real yy, const Real yz, const Real zz) {
  field(0, pangu::nr::Index(pangu::nr::ADMComponent::gxx), 0, 0, cell) = xx;
  field(0, pangu::nr::Index(pangu::nr::ADMComponent::gxy), 0, 0, cell) = xy;
  field(0, pangu::nr::Index(pangu::nr::ADMComponent::gxz), 0, 0, cell) = xz;
  field(0, pangu::nr::Index(pangu::nr::ADMComponent::gyy), 0, 0, cell) = yy;
  field(0, pangu::nr::Index(pangu::nr::ADMComponent::gyz), 0, 0, cell) = yz;
  field(0, pangu::nr::Index(pangu::nr::ADMComponent::gzz), 0, 0, cell) = zz;
}

template <class View>
void SetZ4cSymmetric(View& field, const int cell, const Real xx, const Real xy,
                     const Real xz, const Real yy, const Real yz, const Real zz) {
  field(0, pangu::nr::Index(pangu::nr::Z4cComponent::chi), 0, 0, cell) = 1.0;
  field(0, pangu::nr::Index(pangu::nr::Z4cComponent::gxx), 0, 0, cell) = xx;
  field(0, pangu::nr::Index(pangu::nr::Z4cComponent::gxy), 0, 0, cell) = xy;
  field(0, pangu::nr::Index(pangu::nr::Z4cComponent::gxz), 0, 0, cell) = xz;
  field(0, pangu::nr::Index(pangu::nr::Z4cComponent::gyy), 0, 0, cell) = yy;
  field(0, pangu::nr::Index(pangu::nr::Z4cComponent::gyz), 0, 0, cell) = yz;
  field(0, pangu::nr::Index(pangu::nr::Z4cComponent::gzz), 0, 0, cell) = zz;
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int return_code = 0;
  {
    constexpr int kSpecifiedCells = 2;
    constexpr int kRandomCells = 64;
    constexpr int kCells = kSpecifiedCells + kRandomCells;
    FieldPack z4c{
        Kokkos::View<Real*****>("NR-0 test Z4c", 1, pangu::nr::kZ4cComponents, 1, 1, kCells)};
    FieldPack adm{
        Kokkos::View<Real*****>("NR-0 test ADM", 1, pangu::nr::kADMComponents, 1, 1, kCells)};
    auto z4c_host = Kokkos::create_mirror_view(z4c.values);
    auto adm_host = Kokkos::create_mirror_view(adm.values);
    Kokkos::deep_copy(z4c_host, 0.0);
    Kokkos::deep_copy(adm_host, 0.0);

    z4c_host(0, pangu::nr::Index(pangu::nr::Z4cComponent::alpha), 0, 0, 0) = 1.0;
    z4c_host(0, pangu::nr::Index(pangu::nr::Z4cComponent::alpha), 0, 0, 1) = 3.0;
    z4c_host(0, pangu::nr::Index(pangu::nr::Z4cComponent::betax), 0, 0, 0) = 0.0;
    z4c_host(0, pangu::nr::Index(pangu::nr::Z4cComponent::betax), 0, 0, 1) = 0.2;
    z4c_host(0, pangu::nr::Index(pangu::nr::Z4cComponent::betay), 0, 0, 0) = 0.1;
    z4c_host(0, pangu::nr::Index(pangu::nr::Z4cComponent::betay), 0, 0, 1) = 0.3;
    z4c_host(0, pangu::nr::Index(pangu::nr::Z4cComponent::betaz), 0, 0, 0) = -0.2;
    z4c_host(0, pangu::nr::Index(pangu::nr::Z4cComponent::betaz), 0, 0, 1) = 0.0;
    SetSymmetric(adm_host, 0, 2.0, 0.1, 0.2, 3.0, 0.3, 4.0);
    SetSymmetric(adm_host, 1, 4.0, 0.3, 0.4, 5.0, 0.5, 8.0);
    SetZ4cSymmetric(z4c_host, 0, 2.0, 0.1, 0.2, 3.0, 0.3, 4.0);
    SetZ4cSymmetric(z4c_host, 1, 4.0, 0.3, 0.4, 5.0, 0.5, 8.0);

    std::mt19937_64 generator(0x50414e47554e5230ULL);
    std::uniform_real_distribution<Real> diagonal(0.75, 2.0);
    std::uniform_real_distribution<Real> off_diagonal(-0.25, 0.25);
    std::uniform_real_distribution<Real> lapse(0.5, 2.0);
    std::uniform_real_distribution<Real> shift(-0.4, 0.4);
    for (int cell = kSpecifiedCells; cell < kCells; ++cell) {
      const Real l00 = diagonal(generator);
      const Real l10 = off_diagonal(generator);
      const Real l11 = diagonal(generator);
      const Real l20 = off_diagonal(generator);
      const Real l21 = off_diagonal(generator);
      const Real l22 = diagonal(generator);
      const Real xx = l00 * l00;
      const Real xy = l00 * l10;
      const Real xz = l00 * l20;
      const Real yy = l10 * l10 + l11 * l11;
      const Real yz = l10 * l20 + l11 * l21;
      const Real zz = l20 * l20 + l21 * l21 + l22 * l22;
      SetSymmetric(adm_host, cell, xx, xy, xz, yy, yz, zz);
      SetZ4cSymmetric(z4c_host, cell, xx, xy, xz, yy, yz, zz);
      z4c_host(0, pangu::nr::Index(pangu::nr::Z4cComponent::alpha), 0, 0, cell) = lapse(generator);
      z4c_host(0, pangu::nr::Index(pangu::nr::Z4cComponent::betax), 0, 0, cell) = shift(generator);
      z4c_host(0, pangu::nr::Index(pangu::nr::Z4cComponent::betay), 0, 0, cell) = shift(generator);
      z4c_host(0, pangu::nr::Index(pangu::nr::Z4cComponent::betaz), 0, 0, cell) = shift(generator);
    }
    Kokkos::deep_copy(z4c.values, z4c_host);
    Kokkos::deep_copy(adm.values, adm_host);

    using Geometry =
        pangu::geometry::Geometry<pangu::geometry::metric::Z4c, pangu::geometry::SyncMode>;
    const Geometry geometry(pangu::geometry::metric::Z4c::Parameters{});
    constexpr int kSpecifiedErrors = 31;
    Kokkos::View<Real*> errors("NR-0 geometry errors", kSpecifiedErrors + kRandomCells);
    Kokkos::parallel_for(
        "NR-0 synchronized geometry checks", 1, KOKKOS_LAMBDA(const int) {
          const Coordinates coordinates{};
          const auto cell = geometry.MetricAt(pangu::geometry::Location::cell_center, 0, 0, 0, 1,
                                              coordinates, z4c, adm);
          const auto face = geometry.MetricAt(pangu::geometry::Location::face1, 0, 0, 0, 1,
                                              coordinates, z4c, adm);
          errors(0) = fabs(cell.lapse - 3.0);
          errors(1) = fabs(cell.lower[1][1] - 4.0);
          errors(2) = fabs(cell.lower[1][2] - 0.3);
          errors(3) = fabs(face.lapse - 2.0);
          errors(4) = fabs(face.shift[0] - 0.1);
          errors(5) = fabs(face.shift[1] - 0.2);
          errors(6) = fabs(face.shift[2] + 0.1);
          errors(7) = fabs(face.lower[1][1] - 3.0);
          errors(8) = fabs(face.lower[1][2] - 0.2);
          errors(9) = fabs(face.lower[1][3] - 0.3);
          errors(10) = fabs(face.lower[2][2] - 4.0);
          errors(11) = fabs(face.lower[2][3] - 0.4);
          errors(12) = fabs(face.lower[3][3] - 6.0);
          const Real expected_det = 3.0 * (4.0 * 6.0 - 0.4 * 0.4) - 0.2 * (0.2 * 6.0 - 0.3 * 0.4) +
                                    0.3 * (0.2 * 0.4 - 0.3 * 4.0);
          errors(13) = fabs(face.spatial_det - expected_det);
          errors(14) = fabs(face.gdet - 2.0 * sqrt(expected_det));
          int error = 15;
          for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
              Real identity = 0.0;
              for (int contracted = 0; contracted < 4; ++contracted)
                identity += face.lower[row][contracted] * face.upper[contracted][column];
              errors(error++) = fabs(identity - (row == column ? 1.0 : 0.0));
            }
          }
        });
    Kokkos::parallel_for(
        "NR-0 random positive-definite geometry checks", kRandomCells,
        KOKKOS_LAMBDA(const int sample) {
          const int i = kSpecifiedCells + sample;
          const Coordinates coordinates{};
          const auto cell = geometry.MetricAt(pangu::geometry::Location::cell_center, 0, 0, 0, i,
                                              coordinates, z4c, adm);
          const auto face = geometry.MetricAt(pangu::geometry::Location::face1, 0, 0, 0, i,
                                              coordinates, z4c, adm);
          Real maximum = 0.0;
          for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
              Real identity = 0.0;
              for (int contracted = 0; contracted < 4; ++contracted)
                identity += cell.lower[row][contracted] * cell.upper[contracted][column];
              maximum = fmax(maximum, fabs(identity - (row == column ? 1.0 : 0.0)));
            }
          }
          maximum = fmax(
              maximum,
              fabs(face.lapse -
                   0.5 * (z4c(0, pangu::nr::Index(pangu::nr::Z4cComponent::alpha), 0, 0, i) +
                          z4c(0, pangu::nr::Index(pangu::nr::Z4cComponent::alpha), 0, 0, i - 1))));
          for (int axis = 0; axis < 3; ++axis) {
            const int shift_component = pangu::nr::Index(pangu::nr::Z4cComponent::betax) + axis;
            maximum = fmax(maximum,
                           fabs(face.shift[axis] - 0.5 * (z4c(0, shift_component, 0, 0, i) +
                                                          z4c(0, shift_component, 0, 0, i - 1))));
            for (int second = axis; second < 3; ++second) {
              const int component = pangu::nr::SpatialSymmetricComponent(axis, second);
              maximum =
                  fmax(maximum,
                       fabs(face.lower[axis + 1][second + 1] -
                            0.5 * (adm(0, component, 0, 0, i) + adm(0, component, 0, 0, i - 1))));
            }
          }
          errors(kSpecifiedErrors + sample) = maximum;
        });
    Kokkos::fence();
    const auto errors_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
    Real maximum = 0.0;
    for (std::size_t index = 0; index < errors_host.extent(0); ++index)
      maximum = std::max(maximum, errors_host(index));
    const Real tolerance = 256.0 * std::numeric_limits<Real>::epsilon();
    if (!std::isfinite(maximum) || maximum > tolerance) {
      std::cerr << "NR-0 synchronized Geometry failure: max error=" << maximum
                << " tolerance=" << tolerance << '\n';
      return_code = 1;
    } else {
      std::cout << "NR-0 synchronized Geometry PASS: max error=" << maximum << '\n';
    }
  }
  Kokkos::finalize();
  return return_code;
}
