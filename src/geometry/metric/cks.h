#ifndef PANGU_GEOMETRY_METRIC_CKS_H_
#define PANGU_GEOMETRY_METRIC_CKS_H_

#include "geometry/geometry.h"

namespace pangu::geometry::metric {

struct CKS {
private:
  KOKKOS_INLINE_FUNCTION static void
  SetDerivative(MetricDerivatives& result, const int axis, const parthenon::Real df,
                const parthenon::Real dl0, const parthenon::Real dl1, const parthenon::Real dl2,
                const parthenon::Real dl3, const parthenon::Real f, const parthenon::Real l[4]) {
    result.lower[axis][0][0] = df * l[0] * l[0] + f * dl0 * l[0] + f * l[0] * dl0;
    result.lower[axis][0][1] = df * l[0] * l[1] + f * dl0 * l[1] + f * l[0] * dl1;
    result.lower[axis][0][2] = df * l[0] * l[2] + f * dl0 * l[2] + f * l[0] * dl2;
    result.lower[axis][0][3] = df * l[0] * l[3] + f * dl0 * l[3] + f * l[0] * dl3;
    result.lower[axis][1][0] = result.lower[axis][0][1];
    result.lower[axis][1][1] = df * l[1] * l[1] + f * dl1 * l[1] + f * l[1] * dl1;
    result.lower[axis][1][2] = df * l[1] * l[2] + f * dl1 * l[2] + f * l[1] * dl2;
    result.lower[axis][1][3] = df * l[1] * l[3] + f * dl1 * l[3] + f * l[1] * dl3;
    result.lower[axis][2][0] = result.lower[axis][0][2];
    result.lower[axis][2][1] = result.lower[axis][1][2];
    result.lower[axis][2][2] = df * l[2] * l[2] + f * dl2 * l[2] + f * l[2] * dl2;
    result.lower[axis][2][3] = df * l[2] * l[3] + f * dl2 * l[3] + f * l[2] * dl3;
    result.lower[axis][3][0] = result.lower[axis][0][3];
    result.lower[axis][3][1] = result.lower[axis][1][3];
    result.lower[axis][3][2] = result.lower[axis][2][3];
    result.lower[axis][3][3] = df * l[3] * l[3] + f * dl3 * l[3] + f * l[3] * dl3;
  }

public:
  struct Parameters {
    parthenon::Real spin = 0.0;
    bool flat = false;
  };

  static constexpr Symmetry symmetry = Symmetry::general_3d;
  static constexpr bool supports_excision = true;
  static constexpr bool unit_determinant = true;
  static constexpr bool unit_coordinate_light_bound = true;

  static Parameters Configure(const Background background, const parthenon::Real spin,
                              parthenon::ParameterInput*) {
    return Parameters{spin, background == Background::minkowski};
  }

  static ExcisionParameters ConfigureExcision(parthenon::ParameterInput* pin) {
    return {pin->GetOrAddBoolean("geometry", "excision", false),
            pin->GetOrAddReal("geometry", "excision_radius", 1.0),
            pin->GetOrAddReal("geometry", "dexcise", -1.0),
            pin->GetOrAddReal("geometry", "pexcise", -1.0)};
  }

  KOKKOS_INLINE_FUNCTION static SphericalKerrSchildPoint
  SphericalCoordinates(const parthenon::Real x, const parthenon::Real y, const parthenon::Real z,
                       const Parameters& parameters) {
    const parthenon::Real spherical = sqrt(x * x + y * y + z * z);
    const parthenon::Real spin2 = parameters.spin * parameters.spin;
    const parthenon::Real radius =
        sqrt(spherical * spherical - spin2 +
             sqrt((spherical * spherical - spin2) * (spherical * spherical - spin2) +
                  4.0 * spin2 * z * z)) /
        sqrt(2.0);
    const parthenon::Real cosine = z / radius;
    const parthenon::Real theta = acos(fabs(cosine) > 1.0 ? copysign(1.0, cosine) : cosine);
    const parthenon::Real phi =
        atan2(radius * y - parameters.spin * x, parameters.spin * y + radius * x);
    return {radius, theta, phi};
  }

  KOKKOS_INLINE_FUNCTION static void
  BoyerLindquistSpatialToNative(const parthenon::Real x, const parthenon::Real y,
                                const parthenon::Real z, const parthenon::Real boyer_lindquist[3],
                                parthenon::Real native[3], const Parameters& parameters) {
    const auto spherical = SphericalCoordinates(x, y, z, parameters);
    const parthenon::Real radius = spherical.radius;
    const parthenon::Real spin = parameters.spin;
    const parthenon::Real delta = radius * radius - 2.0 * radius + spin * spin;
    const parthenon::Real radial = boyer_lindquist[0];
    const parthenon::Real polar = boyer_lindquist[1];
    const parthenon::Real azimuthal = boyer_lindquist[2];
    const parthenon::Real cotangent = cos(spherical.theta) / sin(spherical.theta);
    native[0] =
        radial * ((radius * x + spin * y) / (radius * radius + spin * spin) - spin * y / delta) +
        polar * cotangent * x - azimuthal * y;
    native[1] =
        radial * ((radius * y - spin * x) / (radius * radius + spin * spin) + spin * x / delta) +
        polar * cotangent * y + azimuthal * x;
    native[2] = radial * z / radius - polar * radius * sin(spherical.theta);
  }

  KOKKOS_INLINE_FUNCTION static void
  AzimuthalCovectorToNative(const parthenon::Real x, const parthenon::Real y, parthenon::Real z,
                            const parthenon::Real azimuthal, parthenon::Real native[3],
                            const Parameters& parameters) {
    parthenon::Real spherical_radius = sqrt(x * x + y * y + z * z);
    if (spherical_radius < 1.0 && fabs(z) < 1.0e-5)
      z = 1.0e-5;
    spherical_radius = sqrt(x * x + y * y + z * z);
    const auto spherical = SphericalCoordinates(x, y, z, parameters);
    const parthenon::Real radius = spherical.radius;
    const parthenon::Real spin = parameters.spin;
    const parthenon::Real root =
        2.0 * radius * radius - spherical_radius * spherical_radius + spin * spin;
    native[0] = azimuthal * (-y / (x * x + y * y) +
                             spin * x * radius / ((spin * spin + radius * radius) * root));
    native[1] = azimuthal * (x / (x * x + y * y) +
                             spin * y * radius / ((spin * spin + radius * radius) * root));
    native[2] = azimuthal * spin * z / (radius * root);
  }

  KOKKOS_INLINE_FUNCTION static parthenon::Real ExcisionRadius(const parthenon::Real x,
                                                               const parthenon::Real y,
                                                               const parthenon::Real z,
                                                               const Parameters& parameters) {
    const parthenon::Real radius2 = x * x + y * y + z * z;
    const parthenon::Real spin2 = parameters.spin * parameters.spin;
    return sqrt(0.5 * (radius2 - spin2 +
                       sqrt((radius2 - spin2) * (radius2 - spin2) + 4.0 * spin2 * z * z)));
  }

  template <class Coordinates>
  KOKKOS_INLINE_FUNCTION static bool
  NeedsFluxExcision(const Coordinates& coordinates, const int k, const int j, const int i,
                    const Parameters& parameters, const parthenon::Real radius) {
    const parthenon::Real x1v = coordinates.template Xc<1>(i);
    const parthenon::Real x1vm1 = coordinates.template Xc<1>(i - 1);
    const parthenon::Real x1vp1 = coordinates.template Xc<1>(i + 1);
    const parthenon::Real x1f = coordinates.template Xf<1>(i);
    const parthenon::Real x1fm1 = coordinates.template Xf<1>(i - 1);
    const parthenon::Real x1fp1 = coordinates.template Xf<1>(i + 1);
    const parthenon::Real x1fp2 = coordinates.template Xf<1>(i + 2);
    const parthenon::Real x2v = coordinates.template Xc<2>(j);
    const parthenon::Real x2vm1 = coordinates.template Xc<2>(j - 1);
    const parthenon::Real x2vp1 = coordinates.template Xc<2>(j + 1);
    const parthenon::Real x2f = coordinates.template Xf<2>(j);
    const parthenon::Real x2fm1 = coordinates.template Xf<2>(j - 1);
    const parthenon::Real x2fp1 = coordinates.template Xf<2>(j + 1);
    const parthenon::Real x2fp2 = coordinates.template Xf<2>(j + 2);
    const parthenon::Real x3v = coordinates.template Xc<3>(k);
    const parthenon::Real x3vm1 = coordinates.template Xc<3>(k - 1);
    const parthenon::Real x3vp1 = coordinates.template Xc<3>(k + 1);
    const parthenon::Real x3f = coordinates.template Xf<3>(k);
    const parthenon::Real x3fm1 = coordinates.template Xf<3>(k - 1);
    const parthenon::Real x3fp1 = coordinates.template Xf<3>(k + 1);
    const parthenon::Real x3fp2 = coordinates.template Xf<3>(k + 2);

    parthenon::Real x1 = CloserToOrigin(
        CloserToOrigin(CloserToOrigin(CloserToOrigin(x1v, x1vm1), x1f), x1fm1), x1fp1);
    parthenon::Real x2 = CloserToOrigin(CloserToOrigin(x2v, x2f), x2fp1);
    parthenon::Real x3 = CloserToOrigin(CloserToOrigin(x3v, x3f), x3fp1);
    if (ExcisionRadius(x1, x2, x3, parameters) <= radius)
      return true;

    x1 = CloserToOrigin(CloserToOrigin(CloserToOrigin(CloserToOrigin(x1vp1, x1v), x1fp1), x1f),
                        x1fp2);
    if (ExcisionRadius(x1, x2, x3, parameters) <= radius)
      return true;

    x1 = CloserToOrigin(CloserToOrigin(x1v, x1f), x1fp1);
    x2 = CloserToOrigin(CloserToOrigin(CloserToOrigin(CloserToOrigin(x2v, x2vm1), x2f), x2fm1),
                        x2fp1);
    x3 = CloserToOrigin(CloserToOrigin(x3v, x3f), x3fp1);
    if (ExcisionRadius(x1, x2, x3, parameters) <= radius)
      return true;

    x2 = CloserToOrigin(CloserToOrigin(CloserToOrigin(CloserToOrigin(x2vp1, x2v), x2fp1), x2f),
                        x2fp2);
    if (ExcisionRadius(x1, x2, x3, parameters) <= radius)
      return true;

    x1 = CloserToOrigin(CloserToOrigin(x1v, x1f), x1fp1);
    x2 = CloserToOrigin(CloserToOrigin(x2v, x2f), x2fp1);
    x3 = CloserToOrigin(CloserToOrigin(CloserToOrigin(CloserToOrigin(x3v, x3vm1), x3f), x3fm1),
                        x3fp1);
    if (ExcisionRadius(x1, x2, x3, parameters) <= radius)
      return true;

    x3 = CloserToOrigin(CloserToOrigin(CloserToOrigin(CloserToOrigin(x3vp1, x3v), x3fp1), x3f),
                        x3fp2);
    return ExcisionRadius(x1, x2, x3, parameters) <= radius;
  }

  KOKKOS_INLINE_FUNCTION static MetricPoint Calculate(const parthenon::Real x,
                                                      const parthenon::Real y,
                                                      const parthenon::Real z, const Location,
                                                      const Parameters& parameters) {
    const parthenon::Real spin = parameters.spin;
    MetricPoint metric{};
    // Preserve the established operation ordering. Algebraically equivalent
    // radius-squared/loop forms lose ulps close to the excision surface.
    const parthenon::Real radius = sqrt(x * x + y * y + z * z);
    parthenon::Real r =
        sqrt(((radius * radius) - (spin * spin) +
              sqrt(((radius * radius) - (spin * spin)) * ((radius * radius) - (spin * spin)) +
                   4.0 * (spin * spin) * (z * z))) /
             2.0);
    constexpr parthenon::Real epsilon = 1.0e-6;
    if (r < epsilon)
      r = 0.5 * (epsilon + r * r / epsilon);
    parthenon::Real lower_null[4]{1.0, (r * x + spin * y) / ((r * r) + (spin * spin)),
                                  (r * y - spin * x) / ((r * r) + (spin * spin)), z / r};
    parthenon::Real factor = 2.0 * (r * r) * r / (((r * r) * (r * r)) + (spin * spin) * (z * z));
    if (parameters.flat)
      factor = 0.0;
    metric.lower[0][0] = factor * lower_null[0] * lower_null[0] - 1.0;
    metric.lower[0][1] = factor * lower_null[0] * lower_null[1];
    metric.lower[0][2] = factor * lower_null[0] * lower_null[2];
    metric.lower[0][3] = factor * lower_null[0] * lower_null[3];
    metric.lower[1][0] = metric.lower[0][1];
    metric.lower[1][1] = factor * lower_null[1] * lower_null[1] + 1.0;
    metric.lower[1][2] = factor * lower_null[1] * lower_null[2];
    metric.lower[1][3] = factor * lower_null[1] * lower_null[3];
    metric.lower[2][0] = metric.lower[0][2];
    metric.lower[2][1] = metric.lower[1][2];
    metric.lower[2][2] = factor * lower_null[2] * lower_null[2] + 1.0;
    metric.lower[2][3] = factor * lower_null[2] * lower_null[3];
    metric.lower[3][0] = metric.lower[0][3];
    metric.lower[3][1] = metric.lower[1][3];
    metric.lower[3][2] = metric.lower[2][3];
    metric.lower[3][3] = factor * lower_null[3] * lower_null[3] + 1.0;

    const parthenon::Real upper_null[4]{-1.0, lower_null[1], lower_null[2], lower_null[3]};
    metric.upper[0][0] = -factor * upper_null[0] * upper_null[0] - 1.0;
    metric.upper[0][1] = -factor * upper_null[0] * upper_null[1];
    metric.upper[0][2] = -factor * upper_null[0] * upper_null[2];
    metric.upper[0][3] = -factor * upper_null[0] * upper_null[3];
    metric.upper[1][0] = metric.upper[0][1];
    metric.upper[1][1] = -factor * upper_null[1] * upper_null[1] + 1.0;
    metric.upper[1][2] = -factor * upper_null[1] * upper_null[2];
    metric.upper[1][3] = -factor * upper_null[1] * upper_null[3];
    metric.upper[2][0] = metric.upper[0][2];
    metric.upper[2][1] = metric.upper[1][2];
    metric.upper[2][2] = -factor * upper_null[2] * upper_null[2] + 1.0;
    metric.upper[2][3] = -factor * upper_null[2] * upper_null[3];
    metric.upper[3][0] = metric.upper[0][3];
    metric.upper[3][1] = metric.upper[1][3];
    metric.upper[3][2] = metric.upper[2][3];
    metric.upper[3][3] = -factor * upper_null[3] * upper_null[3] + 1.0;
    metric.lapse = sqrt(-1.0 / metric.upper[0][0]);
    for (int axis = 0; axis < 3; ++axis)
      metric.shift[axis] = metric.lapse * metric.lapse * metric.upper[0][axis + 1];
    metric.spatial_det = 1.0 / (metric.lapse * metric.lapse);
    metric.gdet = 1.0;
    return metric;
  }

  KOKKOS_INLINE_FUNCTION static MetricDerivatives
  CalculateDerivatives(const parthenon::Real x, const parthenon::Real y, const parthenon::Real z,
                       const Parameters& parameters) {
    MetricDerivatives result{};
    if (parameters.flat)
      return result;
    const parthenon::Real spin = parameters.spin;
    const parthenon::Real radius = sqrt(x * x + y * y + z * z);
    parthenon::Real r =
        sqrt(((radius * radius) - (spin * spin) +
              sqrt(((radius * radius) - (spin * spin)) * ((radius * radius) - (spin * spin)) +
                   4.0 * (spin * spin) * (z * z))) /
             2.0);
    constexpr parthenon::Real epsilon = 1.0e-6;
    if (r < epsilon)
      r = 0.5 * (epsilon + r * r / epsilon);
    const parthenon::Real l[4]{1.0, (r * x + spin * y) / ((r * r) + (spin * spin)),
                               (r * y - spin * x) / ((r * r) + (spin * spin)), z / r};
    const parthenon::Real qa = 2.0 * (r * r) - (radius * radius) + (spin * spin);
    const parthenon::Real qb = (r * r) + (spin * spin);
    const parthenon::Real qc = 3.0 * ((spin * z) * (spin * z)) - (r * r) * (r * r);
    const parthenon::Real f = 2.0 * (r * r) * r / (((r * r) * (r * r)) + (spin * spin) * (z * z));
    const parthenon::Real df_dx1 = f * f * x / (2.0 * pow(r, 3)) * qc / qa;
    const parthenon::Real df_dx2 = f * f * y / (2.0 * pow(r, 3)) * qc / qa;
    const parthenon::Real df_dx3 =
        f * f * z / (2.0 * pow(r, 5)) * (qc * qb / qa - 2.0 * (spin * r) * (spin * r));
    const parthenon::Real dl1_dx1 =
        x * r * ((spin * spin) * x - 2.0 * spin * r * y - (r * r) * x) / ((qb * qb) * qa) + r / qb;
    const parthenon::Real dl1_dx2 =
        y * r * ((spin * spin) * x - 2.0 * spin * r * y - (r * r) * x) / ((qb * qb) * qa) +
        spin / qb;
    const parthenon::Real dl1_dx3 =
        z / r * ((spin * spin) * x - 2.0 * spin * r * y - (r * r) * x) / (qb * qa);
    const parthenon::Real dl2_dx1 =
        x * r * ((spin * spin) * y + 2.0 * spin * r * x - (r * r) * y) / ((qb * qb) * qa) -
        spin / qb;
    const parthenon::Real dl2_dx2 =
        y * r * ((spin * spin) * y + 2.0 * spin * r * x - (r * r) * y) / ((qb * qb) * qa) + r / qb;
    const parthenon::Real dl2_dx3 =
        z / r * ((spin * spin) * y + 2.0 * spin * r * x - (r * r) * y) / (qb * qa);
    const parthenon::Real dl3_dx1 = -x * z / (r * qa);
    const parthenon::Real dl3_dx2 = -y * z / (r * qa);
    const parthenon::Real dl3_dx3 = -(z * z) / ((r * r) * r) * qb / qa + 1.0 / r;
    const parthenon::Real dl0_dx1 = 0.0, dl0_dx2 = 0.0, dl0_dx3 = 0.0;

    SetDerivative(result, 0, df_dx1, dl0_dx1, dl1_dx1, dl2_dx1, dl3_dx1, f, l);
    SetDerivative(result, 1, df_dx2, dl0_dx2, dl1_dx2, dl2_dx2, dl3_dx2, f, l);
    SetDerivative(result, 2, df_dx3, dl0_dx3, dl1_dx3, dl2_dx3, dl3_dx3, f, l);
    return result;
  }

private:
  KOKKOS_INLINE_FUNCTION static parthenon::Real CloserToOrigin(const parthenon::Real current,
                                                               const parthenon::Real candidate) {
    return fabs(current) < fabs(candidate) ? current : candidate;
  }
};

} // namespace pangu::geometry::metric

#endif
