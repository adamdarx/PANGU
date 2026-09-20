#ifndef PANGU_GEOMETRY_METRIC_MKS_H_
#define PANGU_GEOMETRY_METRIC_MKS_H_

#include <cmath>

#include "geometry/geometry.h"
#include "utils/error_checking.hpp"

namespace pangu::geometry::metric {

// Modified Kerr-Schild coordinates following KHARMA's SphKSCoords composed
// with ModifyTransform. Native coordinates are (t, log(r), x2, phi).
struct MKS {
  struct Parameters {
    parthenon::Real spin = 0.0;
    parthenon::Real hslope = 0.3;
  };

  static constexpr Symmetry symmetry = Symmetry::axisymmetric_2d;
  static constexpr bool supports_excision = false;
  static constexpr bool unit_determinant = false;
  static constexpr bool unit_coordinate_light_bound = false;
  static constexpr parthenon::Real pi = 3.141592653589793238462643383279502884;

  static Parameters Configure(const Background background, const parthenon::Real spin,
                              parthenon::ParameterInput* pin) {
    PARTHENON_REQUIRE(background == Background::kerr_schild,
                      "METRIC=mks requires geometry/background=mks");
    const parthenon::Real hslope = pin->GetOrAddReal("geometry", "hslope", 0.3);
    PARTHENON_REQUIRE(hslope > 0.0 && hslope < 2.0,
                      "geometry/hslope must be in (0,2) for monotonic MKS coordinates");
    return {spin, hslope};
  }

  KOKKOS_INLINE_FUNCTION static parthenon::Real Radius(const parthenon::Real x1) { return exp(x1); }

  KOKKOS_INLINE_FUNCTION static parthenon::Real PolarAngle(const parthenon::Real x2,
                                                           const Parameters& parameters) {
    return pi * x2 + 0.5 * (1.0 - parameters.hslope) * sin(2.0 * pi * x2);
  }

  KOKKOS_INLINE_FUNCTION static parthenon::Real PolarJacobian(const parthenon::Real x2,
                                                              const Parameters& parameters) {
    return pi * (1.0 + (1.0 - parameters.hslope) * cos(2.0 * pi * x2));
  }

  KOKKOS_INLINE_FUNCTION static parthenon::Real
  PolarSecondDerivative(const parthenon::Real x2, const Parameters& parameters) {
    return -2.0 * pi * pi * (1.0 - parameters.hslope) * sin(2.0 * pi * x2);
  }

  KOKKOS_INLINE_FUNCTION static void NativeToEmbedding(const parthenon::Real native[4],
                                                       parthenon::Real embedding[4],
                                                       const Parameters& parameters) {
    embedding[0] = native[0];
    embedding[1] = Radius(native[1]);
    embedding[2] = RegularizePolarAxis(PolarAngle(native[2], parameters));
    embedding[3] = native[3];
  }

  KOKKOS_INLINE_FUNCTION static void EmbeddingToNative(const parthenon::Real embedding[4],
                                                       parthenon::Real native[4],
                                                       const Parameters& parameters) {
    native[0] = embedding[0];
    native[1] = log(embedding[1]);
    native[3] = embedding[3];
    parthenon::Real lower = 0.0;
    parthenon::Real upper = 1.0;
    for (int iteration = 0; iteration < 64; ++iteration) {
      const parthenon::Real middle = 0.5 * (lower + upper);
      if (PolarAngle(middle, parameters) < embedding[2])
        lower = middle;
      else
        upper = middle;
    }
    native[2] = 0.5 * (lower + upper);
  }

  KOKKOS_INLINE_FUNCTION static SphericalKerrSchildPoint
  SphericalCoordinates(const parthenon::Real x1, const parthenon::Real x2, const parthenon::Real x3,
                       const Parameters& parameters) {
    return {Radius(x1), RegularizePolarAxis(PolarAngle(x2, parameters)), x3};
  }

  KOKKOS_INLINE_FUNCTION static void
  BoyerLindquistSpatialToNative(const parthenon::Real x1, const parthenon::Real x2,
                                const parthenon::Real, const parthenon::Real boyer_lindquist[3],
                                parthenon::Real native[3], const Parameters& parameters) {
    const parthenon::Real radius = Radius(x1);
    const parthenon::Real delta =
        radius * radius - 2.0 * radius + parameters.spin * parameters.spin;
    native[0] = boyer_lindquist[0] / radius;
    native[1] = boyer_lindquist[1] / PolarJacobian(x2, parameters);
    native[2] = boyer_lindquist[2] + parameters.spin / delta * boyer_lindquist[0];
  }

  KOKKOS_INLINE_FUNCTION static void
  AzimuthalCovectorToNative(const parthenon::Real, const parthenon::Real, const parthenon::Real,
                            const parthenon::Real azimuthal, parthenon::Real native[3],
                            const Parameters&) {
    native[0] = 0.0;
    native[1] = 0.0;
    native[2] = azimuthal;
  }

  KOKKOS_INLINE_FUNCTION static MetricPoint Calculate(const parthenon::Real x1,
                                                      const parthenon::Real x2,
                                                      const parthenon::Real, const Location,
                                                      const Parameters& parameters) {
    const parthenon::Real r = Radius(x1);
    const parthenon::Real theta = RegularizePolarAxis(PolarAngle(x2, parameters));
    const parthenon::Real polar_jacobian = PolarJacobian(x2, parameters);
    // Preserve the exact equatorial reflection symmetry.  Evaluating
    // cos(pi/2) and sin(pi) through libm leaves O(eps) remnants; in a
    // magnetically dominated one-dimensional Bondi state those remnants act
    // as a spurious polar geometric source and seed a transverse CT field.
    const bool equatorial = fabs(x2 - 0.5) < 1.0e-14;
    const parthenon::Real cosine = equatorial ? 0.0 : cos(theta);
    const parthenon::Real sine = equatorial ? 1.0 : sin(theta);
    const parthenon::Real sine2 = sine * sine;
    const parthenon::Real spin2 = parameters.spin * parameters.spin;
    const parthenon::Real rho2 = r * r + spin2 * cosine * cosine;
    const parthenon::Real factor = 2.0 * r / rho2;
    const parthenon::Real one_plus_factor = 1.0 + factor;

    MetricPoint metric{};
    metric.lower[0][0] = -1.0 + factor;
    metric.lower[0][1] = factor * r;
    metric.lower[0][3] = -parameters.spin * factor * sine2;
    metric.lower[1][0] = metric.lower[0][1];
    metric.lower[1][1] = one_plus_factor * r * r;
    metric.lower[1][3] = -parameters.spin * sine2 * one_plus_factor * r;
    metric.lower[2][2] = rho2 * polar_jacobian * polar_jacobian;
    metric.lower[3][0] = metric.lower[0][3];
    metric.lower[3][1] = metric.lower[1][3];
    metric.lower[3][3] = sine2 * (rho2 + spin2 * sine2 * one_plus_factor);

    const parthenon::Real delta = r * r - 2.0 * r + spin2;
    metric.upper[0][0] = -one_plus_factor;
    metric.upper[0][1] = factor / r;
    metric.upper[1][0] = metric.upper[0][1];
    metric.upper[1][1] = delta / (rho2 * r * r);
    metric.upper[1][3] = parameters.spin / (rho2 * r);
    metric.upper[2][2] = 1.0 / (rho2 * polar_jacobian * polar_jacobian);
    metric.upper[3][1] = metric.upper[1][3];
    metric.upper[3][3] = 1.0 / (rho2 * sine2);

    metric.lapse = sqrt(-1.0 / metric.upper[0][0]);
    for (int axis = 0; axis < 3; ++axis)
      metric.shift[axis] = metric.lapse * metric.lapse * metric.upper[0][axis + 1];
    metric.gdet = fabs(rho2 * sine * r * polar_jacobian);
    const parthenon::Real determinant_ratio = metric.gdet / metric.lapse;
    metric.spatial_det = determinant_ratio * determinant_ratio;
    return metric;
  }

  KOKKOS_INLINE_FUNCTION static MetricDerivatives
  CalculateDerivatives(const parthenon::Real x1, const parthenon::Real x2, const parthenon::Real,
                       const Parameters& parameters) {
    const parthenon::Real r = Radius(x1);
    const parthenon::Real theta = RegularizePolarAxis(PolarAngle(x2, parameters));
    const parthenon::Real polar_jacobian = PolarJacobian(x2, parameters);
    const bool equatorial = fabs(x2 - 0.5) < 1.0e-14;
    const parthenon::Real polar_second =
        equatorial ? 0.0 : PolarSecondDerivative(x2, parameters);
    const parthenon::Real cosine = equatorial ? 0.0 : cos(theta);
    const parthenon::Real sine = equatorial ? 1.0 : sin(theta);
    const parthenon::Real sine2 = sine * sine;
    const parthenon::Real sine2_theta = 2.0 * sine * cosine;
    const parthenon::Real spin = parameters.spin;
    const parthenon::Real spin2 = spin * spin;
    const parthenon::Real rho2 = r * r + spin2 * cosine * cosine;
    const parthenon::Real rho2_r = 2.0 * r;
    const parthenon::Real rho2_theta = -2.0 * spin2 * sine * cosine;
    const parthenon::Real inverse_rho4 = 1.0 / (rho2 * rho2);
    const parthenon::Real factor = 2.0 * r / rho2;
    const parthenon::Real factor_r = 2.0 / rho2 - 4.0 * r * r * inverse_rho4;
    const parthenon::Real factor_theta = -2.0 * r * rho2_theta * inverse_rho4;
    const parthenon::Real one_plus_factor = 1.0 + factor;

    parthenon::Real spherical[4][4]{};
    parthenon::Real spherical_r[4][4]{};
    parthenon::Real spherical_theta[4][4]{};
    spherical[0][0] = -1.0 + factor;
    spherical[0][1] = factor;
    spherical[0][3] = -spin * factor * sine2;
    spherical[1][1] = one_plus_factor;
    spherical[1][3] = -spin * sine2 * one_plus_factor;
    spherical[2][2] = rho2;
    const parthenon::Real azimuthal_inner = rho2 + spin2 * sine2 * one_plus_factor;
    spherical[3][3] = sine2 * azimuthal_inner;

    spherical_r[0][0] = factor_r;
    spherical_r[0][1] = factor_r;
    spherical_r[0][3] = -spin * sine2 * factor_r;
    spherical_r[1][1] = factor_r;
    spherical_r[1][3] = -spin * sine2 * factor_r;
    spherical_r[2][2] = rho2_r;
    spherical_r[3][3] = sine2 * (rho2_r + spin2 * sine2 * factor_r);

    spherical_theta[0][0] = factor_theta;
    spherical_theta[0][1] = factor_theta;
    spherical_theta[0][3] = -spin * (sine2_theta * factor + sine2 * factor_theta);
    spherical_theta[1][1] = factor_theta;
    spherical_theta[1][3] = -spin * (sine2_theta * one_plus_factor + sine2 * factor_theta);
    spherical_theta[2][2] = rho2_theta;
    const parthenon::Real azimuthal_inner_theta =
        rho2_theta + spin2 * (sine2_theta * one_plus_factor + sine2 * factor_theta);
    spherical_theta[3][3] = sine2_theta * azimuthal_inner + sine2 * azimuthal_inner_theta;

    MirrorSymmetric(spherical);
    MirrorSymmetric(spherical_r);
    MirrorSymmetric(spherical_theta);

    const parthenon::Real scale[4]{1.0, r, polar_jacobian, 1.0};
    const parthenon::Real scale_x1[4]{0.0, r, 0.0, 0.0};
    const parthenon::Real scale_x2[4]{0.0, 0.0, polar_second, 0.0};
    MetricDerivatives derivatives{};
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        derivatives.lower[0][mu][nu] =
            spherical_r[mu][nu] * r * scale[mu] * scale[nu] +
            spherical[mu][nu] * (scale_x1[mu] * scale[nu] + scale[mu] * scale_x1[nu]);
        derivatives.lower[1][mu][nu] =
            spherical_theta[mu][nu] * polar_jacobian * scale[mu] * scale[nu] +
            spherical[mu][nu] * (scale_x2[mu] * scale[nu] + scale[mu] * scale_x2[nu]);
      }
    }
    return derivatives;
  }

private:
  KOKKOS_INLINE_FUNCTION static parthenon::Real RegularizeNear(const parthenon::Real value,
                                                               const parthenon::Real center,
                                                               const parthenon::Real range) {
    return fabs(value - center) > range ? value
                                        : (value > center ? center + range : center - range);
  }

  KOKKOS_INLINE_FUNCTION static parthenon::Real RegularizePolarAxis(const parthenon::Real theta) {
    constexpr parthenon::Real epsilon = 1.0e-20;
    return RegularizeNear(RegularizeNear(theta, 0.0, epsilon), pi, epsilon);
  }

  KOKKOS_INLINE_FUNCTION static void MirrorSymmetric(parthenon::Real tensor[4][4]) {
    tensor[1][0] = tensor[0][1];
    tensor[2][0] = tensor[0][2];
    tensor[2][1] = tensor[1][2];
    tensor[3][0] = tensor[0][3];
    tensor[3][1] = tensor[1][3];
    tensor[3][2] = tensor[2][3];
  }
};

} // namespace pangu::geometry::metric

#endif
