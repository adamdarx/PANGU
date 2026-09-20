#ifndef PANGU_Z4C_WEYL_H_
#define PANGU_Z4C_WEYL_H_

#include <cmath>

#include <parthenon/parthenon.hpp>

#include "z4c/component_indices.h"
#include "z4c/finite_difference.h"
#include "z4c/z4c_rhs.h"

namespace pangu::nr {

struct WeylScalars {
  parthenon::Real real = 0.0;
  parthenon::Real imag = 0.0;
};

template <int Order, class ADMPack>
KOKKOS_INLINE_FUNCTION WeylScalars ComputeWeylScalars(const ADMPack& adm, const int block,
                                                      const int k, const int j, const int i,
                                                      const parthenon::Real inverse_spacing[3],
                                                      const parthenon::Real x,
                                                      const parthenon::Real y,
                                                      const parthenon::Real z) {
  using Real = parthenon::Real;
  static_assert(fd::kSupportedOrder<Order>);

  const auto metric_accessor = [&](const int first, const int second) {
    return rhs::ComponentAccessor<ADMPack>{
        adm, block, Index(ADMComponent::gxx) + SpatialSymmetricComponent(first, second), k, j, i};
  };
  const auto extrinsic_accessor = [&](const int first, const int second) {
    return rhs::ComponentAccessor<ADMPack>{
        adm, block, Index(ADMComponent::kxx) + SpatialSymmetricComponent(first, second), k, j, i};
  };

  Real metric[3][3]{};
  Real extrinsic[3][3]{};
  Real metric_symmetric[6]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      metric[first][second] =
          adm(block, Index(ADMComponent::gxx) + SpatialSymmetricComponent(first, second), k, j, i);
      extrinsic[first][second] =
          adm(block, Index(ADMComponent::kxx) + SpatialSymmetricComponent(first, second), k, j, i);
      if (second >= first)
        metric_symmetric[SpatialSymmetricComponent(first, second)] = metric[first][second];
    }
  }
  const Real determinant = rhs::SpatialDeterminant(metric_symmetric);
  if (!(determinant > 0.0))
    return {};
  Real inverse_symmetric[6]{};
  rhs::SpatialInverse(1.0 / determinant, metric_symmetric, inverse_symmetric);
  Real inverse[3][3]{};
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      inverse[first][second] = rhs::SymmetricValue(inverse_symmetric, first, second);

  Real dmetric[3][3][3]{};
  Real dextrinsic[3][3][3]{};
  Real ddmetric[3][3][3][3]{};
  for (int derivative = 0; derivative < 3; ++derivative) {
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
        dmetric[derivative][first][second] = fd::First<Order>(
            derivative, inverse_spacing[derivative], metric_accessor(first, second));
        dextrinsic[derivative][first][second] = fd::First<Order>(
            derivative, inverse_spacing[derivative], extrinsic_accessor(first, second));
      }
    }
  }
  for (int first_derivative = 0; first_derivative < 3; ++first_derivative) {
    for (int second_derivative = 0; second_derivative < 3; ++second_derivative) {
      for (int first = 0; first < 3; ++first) {
        for (int second = 0; second < 3; ++second) {
          if (first_derivative == second_derivative) {
            ddmetric[first_derivative][second_derivative][first][second] =
                fd::Second<Order>(first_derivative, inverse_spacing[first_derivative],
                                  metric_accessor(first, second));
          } else {
            ddmetric[first_derivative][second_derivative][first][second] = fd::MixedSecond<Order>(
                first_derivative, second_derivative, inverse_spacing[first_derivative],
                inverse_spacing[second_derivative], metric_accessor(first, second));
          }
        }
      }
    }
  }

  Real christoffel_lower[3][3][3]{};
  Real christoffel_upper[3][3][3]{};
  for (int upper_or_lower = 0; upper_or_lower < 3; ++upper_or_lower) {
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
        christoffel_lower[upper_or_lower][first][second] =
            0.5 * (dmetric[first][second][upper_or_lower] + dmetric[second][first][upper_or_lower] -
                   dmetric[upper_or_lower][first][second]);
      }
    }
  }
  for (int upper = 0; upper < 3; ++upper) {
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
        for (int lower = 0; lower < 3; ++lower)
          christoffel_upper[upper][first][second] +=
              inverse[upper][lower] * christoffel_lower[lower][first][second];
      }
    }
  }

  Real ricci[3][3]{};
  Real ricci_scalar = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      for (int contracted_first = 0; contracted_first < 3; ++contracted_first) {
        for (int contracted_second = 0; contracted_second < 3; ++contracted_second) {
          for (int christoffel = 0; christoffel < 3; ++christoffel) {
            ricci[first][second] +=
                inverse[contracted_first][contracted_second] *
                (christoffel_upper[christoffel][first][contracted_first] *
                     christoffel_lower[christoffel][second][contracted_second] -
                 christoffel_upper[christoffel][first][second] *
                     christoffel_lower[christoffel][contracted_first][contracted_second]);
          }
          ricci[first][second] += 0.5 * inverse[contracted_first][contracted_second] *
                                  (-ddmetric[contracted_first][contracted_second][first][second] -
                                   ddmetric[first][second][contracted_first][contracted_second] +
                                   ddmetric[first][contracted_first][second][contracted_second] +
                                   ddmetric[second][contracted_first][first][contracted_second]);
        }
      }
      ricci_scalar += inverse[first][second] * ricci[first][second];
    }
  }

  Real extrinsic_mixed[3][3]{};
  Real trace_extrinsic = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      for (int contracted = 0; contracted < 3; ++contracted)
        extrinsic_mixed[first][second] +=
            inverse[first][contracted] * extrinsic[contracted][second];
    }
    trace_extrinsic += extrinsic_mixed[first][first];
  }
  Real covariant_dextrinsic[3][3][3]{};
  for (int derivative = 0; derivative < 3; ++derivative) {
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
        covariant_dextrinsic[derivative][first][second] = dextrinsic[derivative][first][second];
        for (int contracted = 0; contracted < 3; ++contracted) {
          covariant_dextrinsic[derivative][first][second] -=
              christoffel_upper[contracted][derivative][first] * extrinsic[contracted][second] +
              christoffel_upper[contracted][derivative][second] * extrinsic[first][contracted];
        }
      }
    }
  }

  Real radial[3] = {x, y, z};
  Real polar[3] = {x * z, y * z, -(x * x + y * y)};
  Real azimuthal[3] = {-y, x, 0.0};
  if (x * x + y * y < 1.0e-10) {
    const Real shifted_x = x + 1.0e-8;
    radial[0] = shifted_x;
    polar[0] = shifted_x * z;
    polar[2] = -(shifted_x * shifted_x + y * y);
    azimuthal[1] = shifted_x;
  }
  const auto dot = [&](const Real first[3], const Real second[3]) {
    Real value = 0.0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        value += metric[a][b] * first[a] * second[b];
    return value;
  };
  Real norm = sqrt(dot(azimuthal, azimuthal));
  if (!(norm > 0.0))
    return {};
  for (int axis = 0; axis < 3; ++axis)
    azimuthal[axis] /= norm;
  Real projection = dot(azimuthal, radial);
  for (int axis = 0; axis < 3; ++axis)
    radial[axis] -= projection * azimuthal[axis];
  norm = sqrt(dot(radial, radial));
  if (!(norm > 0.0))
    return {};
  for (int axis = 0; axis < 3; ++axis)
    radial[axis] /= norm;
  const Real azimuthal_projection = dot(azimuthal, polar);
  const Real radial_projection = dot(radial, polar);
  for (int axis = 0; axis < 3; ++axis)
    polar[axis] -= azimuthal_projection * azimuthal[axis] + radial_projection * radial[axis];
  norm = sqrt(dot(polar, polar));
  if (!(norm > 0.0))
    return {};
  for (int axis = 0; axis < 3; ++axis)
    polar[axis] /= norm;

  Real riemann3[3][3][3][3]{};
  Real riemann4_spatial[3][3][3][3]{};
  Real riemann4_one_normal[3][3][3]{};
  Real riemann4_two_normal[3][3]{};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      riemann4_two_normal[a][b] = ricci[a][b] + trace_extrinsic * extrinsic[a][b];
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d)
          riemann4_two_normal[a][b] -= inverse[c][d] * extrinsic[a][c] * extrinsic[d][b];
        riemann4_one_normal[a][b][c] =
            -(covariant_dextrinsic[c][a][b] - covariant_dextrinsic[b][a][c]);
        for (int d = 0; d < 3; ++d) {
          riemann3[a][b][c][d] = metric[a][c] * ricci[b][d] + metric[b][d] * ricci[a][c] -
                                 metric[a][d] * ricci[b][c] - metric[b][c] * ricci[a][d] -
                                 0.5 * ricci_scalar * metric[a][c] * metric[b][d] +
                                 0.5 * ricci_scalar * metric[a][d] * metric[b][c];
          riemann4_spatial[a][b][c][d] = riemann3[a][b][c][d] + extrinsic[a][c] * extrinsic[b][d] -
                                         extrinsic[a][d] * extrinsic[b][c];
        }
      }
    }
  }

  WeylScalars result{};
  constexpr Real quarter = 0.25;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const Real real_tetrad = polar[a] * polar[b] - azimuthal[a] * azimuthal[b];
      const Real imag_tetrad = -polar[a] * azimuthal[b] - azimuthal[a] * polar[b];
      result.real -= quarter * riemann4_two_normal[a][b] * real_tetrad;
      result.imag -= quarter * riemann4_two_normal[a][b] * imag_tetrad;
      for (int c = 0; c < 3; ++c) {
        result.real += 0.5 * riemann4_one_normal[a][c][b] * radial[c] * real_tetrad;
        result.imag += 0.5 * riemann4_one_normal[a][c][b] * radial[c] * imag_tetrad;
        for (int d = 0; d < 3; ++d) {
          result.real -=
              quarter * riemann4_spatial[d][a][c][b] * radial[d] * radial[c] * real_tetrad;
          result.imag -=
              quarter * riemann4_spatial[d][a][c][b] * radial[d] * radial[c] * imag_tetrad;
        }
      }
    }
  }
  const Real radius = sqrt(x * x + y * y + z * z);
  result.real *= radius;
  result.imag *= radius;
  return result;
}

} // namespace pangu::nr

#endif
