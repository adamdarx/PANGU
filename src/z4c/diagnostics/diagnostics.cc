#include "z4c/diagnostics/diagnostics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "globals.hpp"
#include "z4c/core/component_indices.h"
#include "z4c/core/finite_difference.h"
#include "z4c/evolution/package.h"
#include "utils/error_checking.hpp"

namespace pangu::nr {
using namespace parthenon::package::prelude;

namespace {

constexpr Real kPi = 3.141592653589793238462643383279502884;

struct SurfacePoint {
  Real theta;
  Real phi;
  Real weight;
  Real x;
  Real y;
  Real z;
};

struct HorizonSurfacePoint {
  Real theta;
  Real phi;
  Real weight;
  Real x;
  Real y;
  Real z;
  Real radius;
  Real radius_theta;
  Real radius_phi;
  Real radius_theta_theta;
  Real radius_theta_phi;
  Real radius_phi_phi;
};

struct RealHarmonic {
  Real value = 0.0;
  Real theta = 0.0;
  Real phi = 0.0;
  Real theta_theta = 0.0;
  Real theta_phi = 0.0;
  Real phi_phi = 0.0;
};

struct HorizonEvaluation {
  Real area = std::numeric_limits<Real>::quiet_NaN();
  Real coordinate_area = std::numeric_limits<Real>::quiet_NaN();
  Real mean = std::numeric_limits<Real>::quiet_NaN();
  Real rms = std::numeric_limits<Real>::quiet_NaN();
  Real integrated_expansion = std::numeric_limits<Real>::quiet_NaN();
  Real spin_x = std::numeric_limits<Real>::quiet_NaN();
  Real spin_y = std::numeric_limits<Real>::quiet_NaN();
  Real spin_z = std::numeric_limits<Real>::quiet_NaN();
  std::vector<Real> expansion;
  std::vector<Real> flow;
};

Real Factorial(int value) {
  Real result = 1.0;
  for (int factor = 2; factor <= value; ++factor)
    result *= factor;
  return result;
}

ComplexValue ScalarSphericalHarmonic(const int l, const int m, const Real theta, const Real phi) {
  Real wigner_d = 0.0;
  const int first = std::max(0, m);
  const int last = std::min(l + m, l);
  for (int k = first; k <= last; ++k) {
    const Real sign = (k % 2 == 0) ? 1.0 : -1.0;
    wigner_d += sign * pow(cos(0.5 * theta), 2 * l + m - 2 * k) * pow(sin(0.5 * theta), 2 * k - m) /
                (Factorial(l + m - k) * Factorial(l - k) * Factorial(k) * Factorial(k - m));
  }
  wigner_d *= sqrt((2.0 * l + 1.0) / (4.0 * kPi)) * Factorial(l) *
              sqrt(Factorial(l + m) * Factorial(l - m));
  return {wigner_d * cos(m * phi), wigner_d * sin(m * phi)};
}

// AthenaK's spin-weight -2 spherical harmonic convention from
// z4c_wave_extr.cpp.  Keeping the summation limits and operation order here is
// important: all l=2..8 waveform columns are compared directly with AthenaK.
ComplexValue SpinMinusTwoHarmonic(const int l, const int m, const Real theta, const Real phi) {
  Real wigner_d = 0.0;
  const int first = std::max(m - 2, 0);
  const int last = std::min(l + m, l - 2);
  for (int k = first; k <= last; ++k) {
    wigner_d += pow(-1.0, k) *
                sqrt(Factorial(l + m) * Factorial(l - m) * Factorial(l + 2) * Factorial(l - 2)) *
                pow(cos(theta / 2.0), 2 * l + m - 2 - 2 * k) *
                pow(sin(theta / 2.0), 2 * k + 2 - m) /
                (Factorial(l + m - k) * Factorial(l - 2 - k) * Factorial(k) * Factorial(k + 2 - m));
  }
  const Real normalization = sqrt((2.0 * l + 1.0) / (4.0 * kPi));
  return {normalization * wigner_d * cos(m * phi), normalization * wigner_d * sin(m * phi)};
}

RealHarmonic ComplexHarmonicDerivatives(const int l, const int m, const Real theta, const Real phi,
                                        const bool imaginary) {
  const auto harmonic = ScalarSphericalHarmonic(l, m, theta, phi);
  Real derivative_theta_real = 0.0;
  Real derivative_theta_imag = 0.0;
  if (m < l) {
    const auto upper = ScalarSphericalHarmonic(l, m + 1, theta, phi);
    const Real coefficient = 0.5 * sqrt(static_cast<Real>((l - m) * (l + m + 1)));
    derivative_theta_real += coefficient * (upper.real * cos(phi) + upper.imag * sin(phi));
    derivative_theta_imag += coefficient * (-upper.real * sin(phi) + upper.imag * cos(phi));
  }
  if (m > -l) {
    const auto lower = ScalarSphericalHarmonic(l, m - 1, theta, phi);
    const Real coefficient = 0.5 * sqrt(static_cast<Real>((l + m) * (l - m + 1)));
    derivative_theta_real -= coefficient * (lower.real * cos(phi) - lower.imag * sin(phi));
    derivative_theta_imag -= coefficient * (lower.real * sin(phi) + lower.imag * cos(phi));
  }
  const Real value = imaginary ? harmonic.imag : harmonic.real;
  const Real derivative_theta = imaginary ? derivative_theta_imag : derivative_theta_real;
  const Real derivative_phi = imaginary ? m * harmonic.real : -m * harmonic.imag;
  const Real derivative_theta_phi =
      imaginary ? m * derivative_theta_real : -m * derivative_theta_imag;
  const Real inverse_sin_squared = 1.0 / (sin(theta) * sin(theta));
  const Real derivative_theta_theta = -l * (l + 1.0) * value -
                                      cos(theta) / sin(theta) * derivative_theta +
                                      m * m * inverse_sin_squared * value;
  return {
      value,         derivative_theta, derivative_phi, derivative_theta_theta, derivative_theta_phi,
      -m * m * value};
}

int HarmonicModeCount(const int lmax) { return (lmax + 1) * (lmax + 1); }

std::vector<int> HarmonicDegrees(const int lmax) {
  std::vector<int> degrees;
  degrees.reserve(HarmonicModeCount(lmax));
  for (int l = 0; l <= lmax; ++l) {
    degrees.push_back(l);
    for (int m = 1; m <= l; ++m) {
      degrees.push_back(l);
      degrees.push_back(l);
    }
  }
  return degrees;
}

std::vector<RealHarmonic> BuildRealHarmonics(const std::vector<SurfacePoint>& sphere,
                                             const int lmax) {
  constexpr Real sqrt_two = 1.414213562373095048801688724209698079;
  const int modes = HarmonicModeCount(lmax);
  std::vector<RealHarmonic> harmonics(sphere.size() * modes);
  for (std::size_t point = 0; point < sphere.size(); ++point) {
    int mode = 0;
    for (int l = 0; l <= lmax; ++l) {
      harmonics[point * modes + mode++] =
          ComplexHarmonicDerivatives(l, 0, sphere[point].theta, sphere[point].phi, false);
      for (int m = 1; m <= l; ++m) {
        auto cosine =
            ComplexHarmonicDerivatives(l, m, sphere[point].theta, sphere[point].phi, false);
        auto sine = ComplexHarmonicDerivatives(l, m, sphere[point].theta, sphere[point].phi, true);
        cosine.value *= sqrt_two;
        cosine.theta *= sqrt_two;
        cosine.phi *= sqrt_two;
        cosine.theta_theta *= sqrt_two;
        cosine.theta_phi *= sqrt_two;
        cosine.phi_phi *= sqrt_two;
        sine.value *= sqrt_two;
        sine.theta *= sqrt_two;
        sine.phi *= sqrt_two;
        sine.theta_theta *= sqrt_two;
        sine.theta_phi *= sqrt_two;
        sine.phi_phi *= sqrt_two;
        harmonics[point * modes + mode++] = cosine;
        harmonics[point * modes + mode++] = sine;
      }
    }
  }
  return harmonics;
}

double LegendrePolynomial(const int degree, const double x) {
  if (degree == 0)
    return 1.0;
  if (degree == 1)
    return x;
  double previous = 1.0;
  double current = x;
  for (int order = 2; order <= degree; ++order) {
    const double next =
        ((2.0 * order - 1.0) * x * current - (order - 1.0) * previous) / static_cast<double>(order);
    previous = current;
    current = next;
  }
  return current;
}

double LegendreDerivative(const int degree, const double x) {
  if (degree == 0)
    return 0.0;
  const double denominator = 1.0 - x * x;
  if (fabs(denominator) < 1.0e-30)
    return 0.0;
  return degree * (LegendrePolynomial(degree - 1, x) - x * LegendrePolynomial(degree, x)) /
         denominator;
}

// Preserve AthenaK's all-root Newton iteration and final sort.  Computing only
// half the roots and reflecting them changes low bits and the subsequent
// reduction order, which is visible in non-axisymmetric fast-flow surfaces.
std::vector<std::pair<Real, Real>> GaussLegendre(const int count) {
  std::vector<std::pair<Real, Real>> result(count);
  for (int index = 0; index < count; ++index) {
    double root = cos(kPi * (index + 0.75) / (count + 0.5));
    for (int iteration = 0; iteration < 20; ++iteration) {
      const double derivative = LegendreDerivative(count, root);
      if (fabs(derivative) < 1.0e-30)
        break;
      const double update = LegendrePolynomial(count, root) / derivative;
      root -= update;
      if (fabs(update) < std::numeric_limits<double>::epsilon())
        break;
      PARTHENON_REQUIRE(iteration != 19, "failed to initialize Gauss-Legendre grid");
    }
    const double derivative = LegendreDerivative(count, root);
    result[index] = {root, 2.0 / ((1.0 - root * root) * derivative * derivative)};
  }
  std::sort(result.begin(), result.end(),
            [](const auto& left, const auto& right) { return left.first < right.first; });
  return result;
}

std::vector<SurfacePoint> BuildSphere(const int ntheta, const Real radius,
                                      const std::array<Real, 3>& center) {
  const int nphi = 2 * ntheta;
  const Real dphi = 2.0 * kPi / nphi;
  const auto polar = GaussLegendre(ntheta);
  std::vector<SurfacePoint> points;
  points.reserve(ntheta * nphi);
  // AthenaK stores theta as the fast index and phi as the slow index, with the
  // first azimuth at exactly zero.
  for (int azimuth = 0; azimuth < nphi; ++azimuth) {
    const Real phi = azimuth * dphi;
    for (const auto& [mu, polar_weight] : polar) {
      const Real theta = acos(mu);
      const Real sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));
      points.push_back({theta, phi, polar_weight * dphi, center[0] + radius * sin_theta * cos(phi),
                        center[1] + radius * sin_theta * sin(phi), center[2] + radius * mu});
    }
  }
  return points;
}

struct HorizonAngularData {
  std::vector<SurfacePoint> unit_sphere;
  std::vector<RealHarmonic> harmonics;
  std::vector<int> degrees;
};

const HorizonAngularData& GetHorizonAngularData(const int ntheta, const int lmax) {
  // AthenaK constructs the Gauss-Legendre grid and its complete real-harmonic
  // table once in FastFlow's constructor.  These arrays depend only on
  // (ntheta,lmax), not on the evolving horizon, so retain the same immutable
  // data across completed RK steps and across multiple horizon finders.
  static std::map<std::pair<int, int>, HorizonAngularData> cache;
  const auto key = std::make_pair(ntheta, lmax);
  const auto found = cache.find(key);
  if (found != cache.end())
    return found->second;
  const std::array<Real, 3> origin{0.0, 0.0, 0.0};
  HorizonAngularData angular;
  angular.unit_sphere = BuildSphere(ntheta, 1.0, origin);
  angular.harmonics = BuildRealHarmonics(angular.unit_sphere, lmax);
  angular.degrees = HarmonicDegrees(lmax);
  return cache.emplace(key, std::move(angular)).first->second;
}

// Host-side, diagnostic-only port of AthenaK's GeodesicGrid.  Wave extraction
// uses this rotated icosahedral grid; the Gauss-Legendre grid above is reserved
// for fast-flow horizons.  Indexing and arithmetic intentionally follow the
// upstream implementation so that angular quadrature has the same point order
// and polygonal solid-angle weights.
class GeodesicGrid {
public:
  explicit GeodesicGrid(const int level) : level_(level) {
    PARTHENON_REQUIRE(level_ > 0, "waveform geodesic level must be positive");
    angles_ = 10 * level_ * level_ + 2;
    const int rows = level_ + 2;
    const int columns = 2 * level_ + 2;
    normals_.assign(5 * rows * columns * 3, 0.0);
    pole_normals_.assign(2 * 3, 0.0);
    indices_.assign(5 * rows * columns, 0);
    pole_indices_.assign(2, 0);
    weights_.assign(angles_, 0.0);

    const Real sin_angle = 2.0 / sqrt(5.0);
    const Real cos_angle = 1.0 / sqrt(5.0);
    const Real p1[3] = {0.0, 0.0, 1.0};
    const Real p2[3] = {sin_angle, 0.0, cos_angle};
    const Real p3[3] = {sin_angle * cos(0.2 * kPi), sin_angle * sin(0.2 * kPi), -cos_angle};
    const Real p4[3] = {sin_angle * cos(-0.4 * kPi), sin_angle * sin(-0.4 * kPi), cos_angle};
    const Real p5[3] = {sin_angle * cos(-0.2 * kPi), sin_angle * sin(-0.2 * kPi), -cos_angle};
    const Real p6[3] = {0.0, 0.0, -1.0};
    Pole(0, 0) = 0.0;
    Pole(0, 1) = 0.0;
    Pole(0, 2) = 1.0;
    Pole(1, 0) = 0.0;
    Pole(1, 1) = 0.0;
    Pole(1, 2) = -1.0;

    int row = 1;
    for (int l = 0; l < level_; ++l) {
      int column = 1;
      for (int m = l; m < level_; ++m) {
        SetNormalizedCombination(row, column++, m - l + 1, p2, level_ - m - 1, p1, l, p4);
      }
      for (int m = level_ - l; m < level_; ++m) {
        SetNormalizedCombination(row, column++, level_ - l, p2, m - level_ + l + 1, p5,
                                 level_ - m - 1, p4);
      }
      for (int m = l; m < level_; ++m) {
        SetNormalizedCombination(row, column++, m - l + 1, p3, level_ - m - 1, p2, l, p5);
      }
      for (int m = level_ - l; m < level_; ++m) {
        SetNormalizedCombination(row, column++, level_ - l, p3, m - level_ + l + 1, p6,
                                 level_ - m - 1, p5);
      }
      ++row;
    }

    for (int patch = 1; patch < 5; ++patch) {
      for (int l = 1; l < 1 + level_; ++l) {
        for (int m = 1; m < 1 + 2 * level_; ++m) {
          const Real x = Normal(0, l, m, 0);
          const Real y = Normal(0, l, m, 1);
          const Real z = Normal(0, l, m, 2);
          Normal(patch, l, m, 0) = x * cos(patch * 0.4 * kPi) + y * sin(patch * 0.4 * kPi);
          Normal(patch, l, m, 1) = y * cos(patch * 0.4 * kPi) - x * sin(patch * 0.4 * kPi);
          Normal(patch, l, m, 2) = z;
        }
      }
    }
    FillNormalGhosts();

    PoleIndex(0) = 10 * level_ * level_;
    PoleIndex(1) = 10 * level_ * level_ + 1;
    for (int patch = 0; patch < 5; ++patch)
      for (int l = 0; l < level_; ++l)
        for (int m = 0; m < 2 * level_; ++m)
          IndexAt(patch, l + 1, m + 1) = patch * 2 * level_ * level_ + l * 2 * level_ + m;
    FillIndexGhosts();

    for (int point = 0; point < angles_; ++point) {
      Real lengths[6]{};
      SolidAngleAndArcLengths(point, weights_[point], lengths);
    }
    Real angles[2]{0.0, 0.0};
    OptimalAngles(angles);
    RotateGrid(angles[0], angles[1]);
  }

  std::vector<SurfacePoint> Surface(const Real radius, const std::array<Real, 3>& center) const {
    std::vector<SurfacePoint> points;
    points.reserve(angles_);
    for (int point = 0; point < angles_; ++point) {
      Real x, y, z;
      GridCartPosition(point, x, y, z);
      points.push_back({acos(z), atan2(y, x), weights_[point], center[0] + radius * x,
                        center[1] + radius * y, center[2] + radius * z});
    }
    return points;
  }

private:
  int level_ = 0;
  int angles_ = 0;
  std::vector<Real> normals_;
  std::vector<Real> pole_normals_;
  std::vector<int> indices_;
  std::vector<int> pole_indices_;
  std::vector<Real> weights_;

  Real& Normal(const int patch, const int row, const int column, const int component) {
    return normals_[(((patch * (level_ + 2) + row) * (2 * level_ + 2) + column) * 3) + component];
  }
  const Real& Normal(const int patch, const int row, const int column, const int component) const {
    return normals_[(((patch * (level_ + 2) + row) * (2 * level_ + 2) + column) * 3) + component];
  }
  Real& Pole(const int pole, const int component) { return pole_normals_[3 * pole + component]; }
  const Real& Pole(const int pole, const int component) const {
    return pole_normals_[3 * pole + component];
  }
  int& IndexAt(const int patch, const int row, const int column) {
    return indices_[(patch * (level_ + 2) + row) * (2 * level_ + 2) + column];
  }
  const int& IndexAt(const int patch, const int row, const int column) const {
    return indices_[(patch * (level_ + 2) + row) * (2 * level_ + 2) + column];
  }
  int& PoleIndex(const int pole) { return pole_indices_[pole]; }

  void SetNormalizedCombination(const int row, const int column, const int a, const Real first[3],
                                const int b, const Real second[3], const int c,
                                const Real third[3]) {
    Real vector[3]{};
    Real norm_squared = 0.0;
    for (int component = 0; component < 3; ++component) {
      vector[component] =
          (a * first[component] + b * second[component] + c * third[component]) / level_;
      norm_squared += vector[component] * vector[component];
    }
    const Real norm = sqrt(norm_squared);
    for (int component = 0; component < 3; ++component)
      Normal(0, row, column, component) = vector[component] / norm;
  }

  void FillNormalGhosts() {
    for (int component = 0; component < 3; ++component) {
      for (int patch = 0; patch < 5; ++patch) {
        for (int k = 0; k < level_; ++k) {
          Normal(patch, 0, k + 1, component) = Normal((patch + 4) % 5, k + 1, 1, component);
          Normal(patch, 0, k + level_ + 1, component) =
              Normal((patch + 4) % 5, level_, k + 1, component);
          Normal(patch, k + 1, 2 * level_ + 1, component) =
              Normal((patch + 4) % 5, level_, k + level_ + 1, component);
          Normal(patch, k + 2, 0, component) = Normal((patch + 1) % 5, 1, k + 1, component);
          Normal(patch, level_ + 1, k + 1, component) =
              Normal((patch + 1) % 5, 1, k + level_ + 1, component);
          Normal(patch, level_ + 1, k + level_ + 1, component) =
              Normal((patch + 1) % 5, k + 2, 2 * level_, component);
        }
        Normal(patch, 1, 0, component) = Pole(0, component);
        Normal(patch, level_ + 1, 2 * level_, component) = Pole(1, component);
        Normal(patch, 0, 2 * level_ + 1, component) = Normal(patch, 0, 2 * level_, component);
      }
    }
  }

  void FillIndexGhosts() {
    for (int patch = 0; patch < 5; ++patch) {
      for (int k = 0; k < level_; ++k) {
        IndexAt(patch, 0, k + 1) = IndexAt((patch + 4) % 5, k + 1, 1);
        IndexAt(patch, 0, k + level_ + 1) = IndexAt((patch + 4) % 5, level_, k + 1);
        IndexAt(patch, k + 1, 2 * level_ + 1) = IndexAt((patch + 4) % 5, level_, k + level_ + 1);
        IndexAt(patch, k + 2, 0) = IndexAt((patch + 1) % 5, 1, k + 1);
        IndexAt(patch, level_ + 1, k + 1) = IndexAt((patch + 1) % 5, 1, k + level_ + 1);
        IndexAt(patch, level_ + 1, k + level_ + 1) = IndexAt((patch + 1) % 5, k + 2, 2 * level_);
      }
      IndexAt(patch, 1, 0) = PoleIndex(0);
      IndexAt(patch, level_ + 1, 2 * level_) = PoleIndex(1);
      IndexAt(patch, 0, 2 * level_ + 1) = IndexAt(patch, 0, 2 * level_);
    }
  }

  void GridCartPosition(const int point, Real& x, Real& y, Real& z) const {
    const int patch = point / (2 * level_ * level_);
    const int row = (point % (2 * level_ * level_)) / (2 * level_);
    const int column = (point % (2 * level_ * level_)) % (2 * level_);
    if (patch == 5) {
      x = Pole(column, 0);
      y = Pole(column, 1);
      z = Pole(column, 2);
    } else {
      x = Normal(patch, row + 1, column + 1, 0);
      y = Normal(patch, row + 1, column + 1, 1);
      z = Normal(patch, row + 1, column + 1, 2);
    }
  }

  void Neighbors(const int point, int& count, int neighbors[6]) const {
    if (point == 10 * level_ * level_) {
      for (int patch = 0; patch < 5; ++patch)
        neighbors[patch] = IndexAt(patch, 1, 1);
      count = 5;
    } else if (point == 10 * level_ * level_ + 1) {
      for (int patch = 0; patch < 5; ++patch)
        neighbors[patch] = IndexAt(patch, level_, 2 * level_);
      count = 5;
    } else {
      const int patch = point / (2 * level_ * level_);
      const int row = (point % (2 * level_ * level_)) / (2 * level_);
      const int column = (point % (2 * level_ * level_)) % (2 * level_);
      neighbors[0] = IndexAt(patch, row + 1, column + 2);
      neighbors[1] = IndexAt(patch, row + 2, column + 1);
      neighbors[2] = IndexAt(patch, row + 2, column);
      neighbors[3] = IndexAt(patch, row + 1, column);
      neighbors[4] = IndexAt(patch, row, column + 1);
      if (point % (2 * level_ * level_) == level_ - 1 ||
          point % (2 * level_ * level_) == 2 * level_ - 1) {
        count = 5;
      } else {
        neighbors[5] = IndexAt(patch, row, column + 2);
        count = 6;
      }
    }
  }

  static void CircumcenterNormalized(const Real x1, const Real x2, const Real x3, const Real y1,
                                     const Real y2, const Real y3, const Real z1, const Real z2,
                                     const Real z3, Real& x, Real& y, Real& z) {
    const Real a = sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2) + (z3 - z2) * (z3 - z2));
    const Real b = sqrt((x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3) + (z1 - z3) * (z1 - z3));
    const Real c = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1));
    const Real denominator = 1.0 / ((a + c + b) * (a + c - b) * (a + b - c) * (b + c - a));
    const Real xc =
        (x1 * (a * a * (b * b + c * c - a * a)) + x2 * (b * b * (c * c + a * a - b * b)) +
         x3 * (c * c * (a * a + b * b - c * c))) *
        denominator;
    const Real yc =
        (y1 * (a * a * (b * b + c * c - a * a)) + y2 * (b * b * (c * c + a * a - b * b)) +
         y3 * (c * c * (a * a + b * b - c * c))) *
        denominator;
    const Real zc =
        (z1 * (a * a * (b * b + c * c - a * a)) + z2 * (b * b * (c * c + a * a - b * b)) +
         z3 * (c * c * (a * a + b * b - c * c))) *
        denominator;
    const Real norm = sqrt(xc * xc + yc * yc + zc * zc);
    x = xc / norm;
    y = yc / norm;
    z = zc / norm;
  }

  void SolidAngleAndArcLengths(const int point, Real& weight, Real lengths[6]) const {
    int count;
    int neighbors[6]{};
    Neighbors(point, count, neighbors);
    Real x0, y0, z0;
    GridCartPosition(point, x0, y0, z0);
    weight = 0.0;
    for (int neighbor = 0; neighbor < count; ++neighbor) {
      Real x1, y1, z1, x2, y2, z2, x3, y3, z3;
      GridCartPosition(neighbors[(neighbor + count - 1) % count], x1, y1, z1);
      GridCartPosition(neighbors[neighbor], x2, y2, z2);
      GridCartPosition(neighbors[(neighbor + 1) % count], x3, y3, z3);
      Real cx1, cy1, cz1, cx2, cy2, cz2;
      CircumcenterNormalized(x0, x1, x2, y0, y1, y2, z0, z1, z2, cx1, cy1, cz1);
      CircumcenterNormalized(x0, x2, x3, y0, y2, y3, z0, z2, z3, cx2, cy2, cz2);
      const Real dot01 = x0 * cx1 + y0 * cy1 + z0 * cz1;
      const Real dot02 = x0 * cx2 + y0 * cy2 + z0 * cz2;
      const Real dot12 = cx1 * cx2 + cy1 * cy2 + cz1 * cz2;
      const Real numerator = fabs(x0 * (cy1 * cz2 - cz1 * cy2) + y0 * (cz1 * cx2 - cx1 * cz2) +
                                  z0 * (cx1 * cy2 - cy1 * cx2));
      weight += 2.0 * atan(numerator / (1.0 + dot01 + dot02 + dot12));
      lengths[neighbor] = acos(dot12);
    }
  }

  Real ArcLength(const int first, const int second) const {
    Real x1, y1, z1, x2, y2, z2;
    GridCartPosition(first, x1, y1, z1);
    GridCartPosition(second, x2, y2, z2);
    return acos(x1 * x2 + y1 * y2 + z1 * z2);
  }

  void OptimalAngles(Real angles[2]) const {
    constexpr int nzeta = 200;
    constexpr int npsi = 200;
    const Real delta_zeta = ArcLength(0, 1) / nzeta;
    const Real delta_psi = kPi / npsi;
    Real best = 0.0;
    for (int l = 0; l < nzeta; ++l) {
      const Real zeta = (l + 1) * delta_zeta;
      for (int k = 0; k < npsi; ++k) {
        const Real psi = (k + 1) * delta_psi;
        const Real kx = -sin(psi);
        const Real ky = cos(psi);
        Real current = 1.0;
        for (int point = 0; point < angles_; ++point) {
          Real x, y, z;
          GridCartPosition(point, x, y, z);
          const Real rx =
              x * cos(zeta) + ky * z * sin(zeta) + kx * (kx * x + ky * y) * (1.0 - cos(zeta));
          const Real ry =
              y * cos(zeta) - kx * z * sin(zeta) + ky * (kx * x + ky * y) * (1.0 - cos(zeta));
          const Real rz = z * cos(zeta) + (kx * y - ky * x) * sin(zeta);
          current = fmin(current, fabs(rx));
          current = fmin(current, fabs(ry));
          current = fmin(current, fabs(rz));
        }
        if (current > best) {
          best = current;
          angles[0] = zeta;
          angles[1] = psi;
        }
      }
    }
  }

  void RotateGrid(const Real zeta, const Real psi) {
    const Real kx = -sin(psi);
    const Real ky = cos(psi);
    for (int patch = 0; patch < 5; ++patch) {
      for (int l = 0; l < level_; ++l) {
        for (int m = 0; m < 2 * level_; ++m) {
          const Real x = Normal(patch, l + 1, m + 1, 0);
          const Real y = Normal(patch, l + 1, m + 1, 1);
          const Real z = Normal(patch, l + 1, m + 1, 2);
          Normal(patch, l + 1, m + 1, 0) =
              x * cos(zeta) + ky * z * sin(zeta) + kx * (kx * x + ky * y) * (1.0 - cos(zeta));
          Normal(patch, l + 1, m + 1, 1) =
              y * cos(zeta) - kx * z * sin(zeta) + ky * (kx * x + ky * y) * (1.0 - cos(zeta));
          Normal(patch, l + 1, m + 1, 2) = z * cos(zeta) + (kx * y - ky * x) * sin(zeta);
        }
      }
    }
    for (int pole = 0; pole < 2; ++pole) {
      const Real x = Pole(pole, 0);
      const Real y = Pole(pole, 1);
      const Real z = Pole(pole, 2);
      Pole(pole, 0) =
          x * cos(zeta) + ky * z * sin(zeta) + kx * (kx * x + ky * y) * (1.0 - cos(zeta));
      Pole(pole, 1) =
          y * cos(zeta) - kx * z * sin(zeta) + ky * (kx * x + ky * y) * (1.0 - cos(zeta));
      Pole(pole, 2) = z * cos(zeta) + (kx * y - ky * x) * sin(zeta);
    }
    FillNormalGhosts();
  }
};

std::vector<HorizonSurfacePoint> BuildHorizonSurface(const std::vector<SurfacePoint>& unit_sphere,
                                                     const std::vector<RealHarmonic>& harmonics,
                                                     const std::vector<Real>& coefficients,
                                                     const std::array<Real, 3>& center) {
  const int modes = coefficients.size();
  PARTHENON_REQUIRE(harmonics.size() == unit_sphere.size() * modes,
                    "invalid apparent-horizon harmonic table");
  std::vector<HorizonSurfacePoint> surface;
  surface.reserve(unit_sphere.size());
  for (std::size_t point = 0; point < unit_sphere.size(); ++point) {
    Real radius = 0.0;
    Real radius_theta = 0.0;
    Real radius_phi = 0.0;
    Real radius_theta_theta = 0.0;
    Real radius_theta_phi = 0.0;
    Real radius_phi_phi = 0.0;
    for (int mode = 0; mode < modes; ++mode) {
      const auto& harmonic = harmonics[point * modes + mode];
      radius += coefficients[mode] * harmonic.value;
      radius_theta += coefficients[mode] * harmonic.theta;
      radius_phi += coefficients[mode] * harmonic.phi;
      radius_theta_theta += coefficients[mode] * harmonic.theta_theta;
      radius_theta_phi += coefficients[mode] * harmonic.theta_phi;
      radius_phi_phi += coefficients[mode] * harmonic.phi_phi;
    }
    const auto& angle = unit_sphere[point];
    const Real sin_theta = sin(angle.theta);
    surface.push_back(
        {angle.theta, angle.phi, angle.weight, center[0] + radius * sin_theta * cos(angle.phi),
         center[1] + radius * sin_theta * sin(angle.phi), center[2] + radius * cos(angle.theta),
         radius, radius_theta, radius_phi, radius_theta_theta, radius_theta_phi, radius_phi_phi});
  }
  return surface;
}

template <typename Point>
bool Contains(const Mesh* mesh, const MeshBlock* block, const Point& point) {
  const auto& size = block->block_size;
  const Real coordinate[3] = {point.x, point.y, point.z};
  const parthenon::CoordinateDirection directions[3] = {X1DIR, X2DIR, X3DIR};
  for (int direction = 0; direction < 3; ++direction) {
    const Real lower = size.xmin(directions[direction]);
    const Real upper = size.xmax(directions[direction]);
    const Real global_upper = mesh->mesh_size.xmax(directions[direction]);
    if (coordinate[direction] < lower ||
        (coordinate[direction] >= upper &&
         !(coordinate[direction] == global_upper && upper == global_upper)))
      return false;
  }
  return true;
}

template <typename Point>
std::vector<int> OwnedPoints(const Mesh* mesh, const MeshBlock* block,
                             const std::vector<Point>& points) {
  std::vector<int> owned;
  for (std::size_t point = 0; point < points.size(); ++point)
    if (Contains(mesh, block, points[point]))
      owned.push_back(static_cast<int>(point));
  return owned;
}

// Exact Parthenon adapter of AthenaK's IndicesAndWeights<NGHOST>.  The
// interpolation stencil has 2*NGHOST cell-centered nodes in each direction.
template <int NGHOST>
KOKKOS_INLINE_FUNCTION void
LagrangeInterpolation(const Real coordinate, const Real lower, const Real spacing,
                             const int interior_start, int* first, Real weights[2 * NGHOST]) {
  const int anchor = static_cast<int>(floor((coordinate - (lower - 0.5 * spacing)) / spacing));
  const int logical_first = anchor - NGHOST;
  *first = interior_start + logical_first;
  for (int node = 0; node < 2 * NGHOST; ++node) {
    const Real node_coordinate = lower + (logical_first + node + 0.5) * spacing;
    weights[node] = 1.0;
    for (int other = 0; other < 2 * NGHOST; ++other) {
      if (other == node)
        continue;
      const Real other_coordinate = lower + (logical_first + other + 0.5) * spacing;
      weights[node] *= (coordinate - other_coordinate) / (node_coordinate - other_coordinate);
    }
  }
}

template <int NPOINTS>
KOKKOS_INLINE_FUNCTION void
SphericalInterpolation(const Real coordinate, const Real lower, const Real spacing,
                              const int interior_start, int* first, Real weights[NPOINTS]) {
  const Real offset = (NPOINTS % 2 == 0) ? -0.5 : 0.0;
  const int anchor = static_cast<int>(floor((coordinate - (lower + offset * spacing)) / spacing));
  const int logical_first = anchor - NPOINTS / 2;
  *first = interior_start + logical_first;
  for (int node = 0; node < NPOINTS; ++node) {
    const Real node_coordinate = lower + (logical_first + node + 0.5) * spacing;
    weights[node] = 1.0;
    for (int other = 0; other < NPOINTS; ++other) {
      if (other == node)
        continue;
      const Real other_coordinate = lower + (logical_first + other + 0.5) * spacing;
      weights[node] *= (coordinate - other_coordinate) / (node_coordinate - other_coordinate);
    }
  }
}

template <int Order> void ComputeADMMetricDerivatives(Mesh* mesh) {
  const auto partitions = mesh->GetDefaultBlockPartitions();
  for (const auto& partition : partitions) {
    auto& data = mesh->mesh_data.Add("base", partition);
    const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
    const auto derivatives = data->PackVariables(std::vector<std::string>{"nr.adm_derivatives"});
    const auto first_block = data->GetBlockData(0)->GetBlockPointer();
    const auto ib = first_block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = first_block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = first_block->cellbounds.GetBoundsK(IndexDomain::interior);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU fast-flow ADM metric derivatives",
        parthenon::DevExecSpace(), 0, adm.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
          const auto& coordinates = adm.GetCoords(block);
          const Real inverse_spacing[3] = {1.0 / coordinates.Dxc<X1DIR>(k, j, i),
                                           1.0 / coordinates.Dxc<X2DIR>(k, j, i),
                                           1.0 / coordinates.Dxc<X3DIR>(k, j, i)};
          for (int derivative = 0; derivative < 3; ++derivative) {
            for (int component = 0; component < 6; ++component) {
              const rhs::ComponentAccessor<decltype(adm)> accessor{
                  adm, block, Index(ADMComponent::gxx) + component, k, j, i};
              derivatives(block, 6 * derivative + component, k, j, i) =
                  fd::First<Order>(derivative, inverse_spacing[derivative], accessor);
            }
          }
        });
  }
  Kokkos::fence();
}

void ComputeADMMetricDerivativesDispatch(Mesh* mesh, const int order) {
  if (order == 2) {
    ComputeADMMetricDerivatives<2>(mesh);
  } else if (order == 4) {
    ComputeADMMetricDerivatives<4>(mesh);
  } else {
    ComputeADMMetricDerivatives<6>(mesh);
  }
}

template <int NPOINTS>
std::vector<Real> InterpolateWeylFixed(Mesh* mesh, const std::vector<SurfacePoint>& points) {
  parthenon::ParArray2DRaw<Real> values("NR waveform interpolation", points.size(), 3);
  Kokkos::deep_copy(values, 0.0);
  const auto partitions = mesh->GetDefaultBlockPartitions();
  for (const auto& partition : partitions) {
    auto& data = mesh->mesh_data.Add("base", partition);
    const auto weyl = data->PackVariables(std::vector<std::string>{"nr.weyl"});
    const auto first_block = data->GetBlockData(0)->GetBlockPointer();
    const auto ib = first_block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = first_block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = first_block->cellbounds.GetBoundsK(IndexDomain::interior);
    std::vector<std::pair<int, int>> owned;
    for (int block = 0; block < data->NumBlocks(); ++block) {
      const auto pointer = data->GetBlockData(block)->GetBlockPointer();
      const auto block_points = OwnedPoints(mesh, pointer, points);
      for (const int point : block_points)
        owned.emplace_back(point, block);
    }
    if (owned.empty())
      continue;
    // point, block, x/y/z, three block lower bounds and three spacings.
    parthenon::ParArray2DRaw<Real> query("NR waveform MeshData queries", owned.size(), 11);
    auto host_query = Kokkos::create_mirror_view(query);
    for (std::size_t local = 0; local < owned.size(); ++local) {
      const int point_index = owned[local].first;
      const int block_index = owned[local].second;
      const auto& point = points[point_index];
      const auto pointer = data->GetBlockData(block_index)->GetBlockPointer();
      const auto& size = pointer->block_size;
      host_query(local, 0) = point_index;
      host_query(local, 1) = block_index;
      host_query(local, 2) = point.x;
      host_query(local, 3) = point.y;
      host_query(local, 4) = point.z;
      for (int direction = 0; direction < 3; ++direction) {
        const auto coordinate_direction =
            static_cast<parthenon::CoordinateDirection>(direction + 1);
        const Real lower = size.xmin(coordinate_direction);
        host_query(local, 5 + direction) = lower;
        host_query(local, 8 + direction) =
            (size.xmax(coordinate_direction) - lower) / size.nx(coordinate_direction);
      }
    }
    Kokkos::deep_copy(query, host_query);
    Kokkos::parallel_for(
        "PANGU waveform MeshData tensor interpolation",
        Kokkos::RangePolicy<parthenon::DevExecSpace>(0, owned.size()),
        KOKKOS_LAMBDA(const int local) {
          const int point = static_cast<int>(query(local, 0));
          const int block = static_cast<int>(query(local, 1));
          int first[3]{};
          Real weights[3][NPOINTS]{};
          const int starts[3] = {ib.s, jb.s, kb.s};
          for (int direction = 0; direction < 3; ++direction)
            SphericalInterpolation<NPOINTS>(query(local, 2 + direction),
                                                   query(local, 5 + direction),
                                                   query(local, 8 + direction), starts[direction],
                                                   &first[direction], weights[direction]);
          for (int component = 0; component < 2; ++component) {
            Real interpolated = 0.0;
            for (int dk = 0; dk < NPOINTS; ++dk)
              for (int dj = 0; dj < NPOINTS; ++dj)
                for (int di = 0; di < NPOINTS; ++di) {
                  const Real weight = weights[0][di] * weights[1][dj] * weights[2][dk];
                  interpolated +=
                      weight * weyl(block, component, first[2] + dk, first[1] + dj, first[0] + di);
                }
            values(point, component) = interpolated;
          }
          values(point, 2) = 1.0;
        });
    Kokkos::fence();
  }
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  std::vector<Real> global(3 * points.size(), 0.0);
  std::vector<Real> local(3 * points.size(), 0.0);
  for (std::size_t point = 0; point < points.size(); ++point)
    for (int component = 0; component < 3; ++component)
      local[3 * point + component] = host(point, component);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(local.data(), global.data(), global.size(), MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#else
  global = local;
#endif
  for (std::size_t point = 0; point < points.size(); ++point)
    PARTHENON_REQUIRE(fabs(global[3 * point + 2] - 1.0) < 0.5,
                      "waveform extraction point has zero or multiple leaf owners");
  return global;
}

std::vector<Real> InterpolateWeyl(Mesh* mesh, const std::vector<SurfacePoint>& points,
                                  const int interpolation_points) {
  switch (interpolation_points) {
  case 2:
    return InterpolateWeylFixed<2>(mesh, points);
  case 3:
    return InterpolateWeylFixed<3>(mesh, points);
  case 4:
    return InterpolateWeylFixed<4>(mesh, points);
  case 5:
    return InterpolateWeylFixed<5>(mesh, points);
  case 6:
    return InterpolateWeylFixed<6>(mesh, points);
  case 7:
    return InterpolateWeylFixed<7>(mesh, points);
  case 8:
    return InterpolateWeylFixed<8>(mesh, points);
  case 9:
    return InterpolateWeylFixed<9>(mesh, points);
  default:
    PARTHENON_FAIL("unsupported waveform interpolation width");
  }
}

template <int NGHOST>
HorizonEvaluation EvaluateStarShapedSurface(Mesh* mesh,
                                            const std::vector<HorizonSurfacePoint>& points,
                                            const int flow_flag) {
  // expansion, area Jacobian/sin(theta), standard flow, Sx/Sy/Sz
  // integrands, and ownership count.
  parthenon::ParArray2DRaw<Real> values("NR horizon surface", points.size(), 7);
  Kokkos::deep_copy(values, 0.0);
  const auto partitions = mesh->GetDefaultBlockPartitions();
  const Real global_upper_x1 = mesh->mesh_size.xmax(X1DIR);
  const Real global_upper_x2 = mesh->mesh_size.xmax(X2DIR);
  const Real global_upper_x3 = mesh->mesh_size.xmax(X3DIR);
  for (const auto& partition : partitions) {
    auto& data = mesh->mesh_data.Add("base", partition);
    const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
    const auto derivatives = data->PackVariables(std::vector<std::string>{"nr.adm_derivatives"});
    const auto first_block = data->GetBlockData(0)->GetBlockPointer();
    const auto ib = first_block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = first_block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = first_block->cellbounds.GetBoundsK(IndexDomain::interior);
    // AthenaK's FastFlow assigns surface points to packed leaf blocks inside
    // its device interpolation kernel.  Do the same with the MeshData packs:
    // each GPU thread owns one angular point and searches only the local packed
    // block bounds before applying the unchanged tensor Lagrange stencil.
    parthenon::ParArray2DRaw<Real> query("NR horizon MeshData queries", points.size(), 11);
    auto host_query = Kokkos::create_mirror_view(query);
    for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
      const auto& point = points[point_index];
      host_query(point_index, 0) = point.x;
      host_query(point_index, 1) = point.y;
      host_query(point_index, 2) = point.z;
      host_query(point_index, 3) = point.theta;
      host_query(point_index, 4) = point.phi;
      host_query(point_index, 5) = point.radius;
      host_query(point_index, 6) = point.radius_theta;
      host_query(point_index, 7) = point.radius_phi;
      host_query(point_index, 8) = point.radius_theta_theta;
      host_query(point_index, 9) = point.radius_theta_phi;
      host_query(point_index, 10) = point.radius_phi_phi;
    }
    Kokkos::deep_copy(query, host_query);
    Kokkos::parallel_for(
        "PANGU fast-flow packed surface interpolation",
        Kokkos::RangePolicy<parthenon::DevExecSpace>(0, points.size()),
        KOKKOS_LAMBDA(const int point) {
          int block = -1;
          Real lower[3]{};
          Real spacing[3]{};
          for (int candidate = 0; candidate < adm.GetDim(5); ++candidate) {
            const auto& coordinates = adm.GetCoords(candidate);
            const Real candidate_lower[3] = {coordinates.Xf<X1DIR>(kb.s, jb.s, ib.s),
                                             coordinates.Xf<X2DIR>(kb.s, jb.s, ib.s),
                                             coordinates.Xf<X3DIR>(kb.s, jb.s, ib.s)};
            const Real candidate_upper[3] = {coordinates.Xf<X1DIR>(kb.s, jb.s, ib.e + 1),
                                             coordinates.Xf<X2DIR>(kb.s, jb.e + 1, ib.s),
                                             coordinates.Xf<X3DIR>(kb.e + 1, jb.s, ib.s)};
            const Real global_upper[3] = {global_upper_x1, global_upper_x2, global_upper_x3};
            bool contains = true;
            for (int direction = 0; direction < 3; ++direction) {
              const Real coordinate = query(point, direction);
              contains = contains && coordinate >= candidate_lower[direction] &&
                         (coordinate < candidate_upper[direction] ||
                          (coordinate == global_upper[direction] &&
                           candidate_upper[direction] == global_upper[direction]));
            }
            if (!contains)
              continue;
            block = candidate;
            lower[0] = candidate_lower[0];
            lower[1] = candidate_lower[1];
            lower[2] = candidate_lower[2];
            spacing[0] = coordinates.Dxc<X1DIR>(kb.s, jb.s, ib.s);
            spacing[1] = coordinates.Dxc<X2DIR>(kb.s, jb.s, ib.s);
            spacing[2] = coordinates.Dxc<X3DIR>(kb.s, jb.s, ib.s);
            break;
          }
          if (block < 0)
            return;
          int first[3]{};
          Real weights[3][2 * NGHOST]{};
          const int starts[3] = {ib.s, jb.s, kb.s};
          for (int direction = 0; direction < 3; ++direction)
            LagrangeInterpolation<NGHOST>(query(point, direction), lower[direction],
                                                 spacing[direction], starts[direction],
                                                 &first[direction], weights[direction]);
          Real metric[3][3]{};
          Real extrinsic[3][3]{};
          Real metric_derivative[3][3][3]{};
          for (int a = 0; a < 3; ++a) {
            for (int b = a; b < 3; ++b) {
              const int metric_component =
                  Index(ADMComponent::gxx) + SpatialSymmetricComponent(a, b);
              const int extrinsic_component =
                  Index(ADMComponent::kxx) + SpatialSymmetricComponent(a, b);
              for (int dk = 0; dk < 2 * NGHOST; ++dk)
                for (int dj = 0; dj < 2 * NGHOST; ++dj)
                  for (int di = 0; di < 2 * NGHOST; ++di) {
                    const int i = first[0] + di;
                    const int j = first[1] + dj;
                    const int k = first[2] + dk;
                    const Real weight = weights[0][di] * weights[1][dj] * weights[2][dk];
                    metric[a][b] += weight * adm(block, metric_component, k, j, i);
                    extrinsic[a][b] += weight * adm(block, extrinsic_component, k, j, i);
                    for (int derivative = 0; derivative < 3; ++derivative) {
                      metric_derivative[derivative][a][b] +=
                          weight * derivatives(block,
                                               6 * derivative + SpatialSymmetricComponent(a, b), k,
                                               j, i);
                    }
                  }
              metric[b][a] = metric[a][b];
              extrinsic[b][a] = extrinsic[a][b];
              for (int derivative = 0; derivative < 3; ++derivative)
                metric_derivative[derivative][b][a] = metric_derivative[derivative][a][b];
            }
          }
          const auto horizon = StarShapedSurfaceExpansion(
              metric, metric_derivative, extrinsic, query(point, 5), query(point, 6),
              query(point, 7), query(point, 8), query(point, 9), query(point, 10), query(point, 3),
              query(point, 4));
          const Real sin_theta = sin(query(point, 3));
          values(point, 0) = horizon.expansion;
          values(point, 1) = sin_theta > 0.0 ? horizon.area_jacobian / sin_theta : 0.0;
          values(point, 2) = flow_flag == 1   ? horizon.expansion_flow
                             : flow_flag == 3 ? horizon.shear_flow
                                              : horizon.flow;
          const Real x = query(point, 5) * sin(query(point, 3)) * cos(query(point, 4));
          const Real y = query(point, 5) * sin(query(point, 3)) * sin(query(point, 4));
          const Real z = query(point, 5) * cos(query(point, 3));
          const Real rotation_x[3] = {0.0, -z, y};
          const Real rotation_y[3] = {z, 0.0, -x};
          const Real rotation_z[3] = {-y, x, 0.0};
          const Real normal[3] = {horizon.normal_x, horizon.normal_y, horizon.normal_z};
          Real spin_x = 0.0;
          Real spin_y = 0.0;
          Real spin_z = 0.0;
          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
              spin_x += rotation_x[a] * normal[b] * extrinsic[a][b];
              spin_y += rotation_y[a] * normal[b] * extrinsic[a][b];
              spin_z += rotation_z[a] * normal[b] * extrinsic[a][b];
            }
          }
          values(point, 3) = spin_x;
          values(point, 4) = spin_y;
          values(point, 5) = spin_z;
          values(point, 6) = 1.0;
        });
    Kokkos::fence();
  }
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  std::vector<Real> local(7 * points.size(), 0.0);
  for (std::size_t point = 0; point < points.size(); ++point) {
    for (int component = 0; component < 7; ++component)
      local[7 * point + component] = host(point, component);
  }
  std::vector<Real> global(local.size(), 0.0);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(local.data(), global.data(), global.size(), MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#else
  global = local;
#endif
  for (std::size_t point = 0; point < points.size(); ++point) {
    const Real owners = global[7 * point + 6];
    if (fabs(owners - 1.0) >= 0.5) {
      if (parthenon::Globals::my_rank == 0) {
        std::cerr << "PANGU apparent-horizon search rejected surface point " << point
                  << ": owners=" << owners << " x=" << points[point].x
                  << " y=" << points[point].y << " z=" << points[point].z
                  << " radius=" << points[point].radius << '\n';
      }
      // A fast-flow trial surface can temporarily leave the leaf mesh when
      // the nonlinear search diverges. This is a failed diagnostic iteration,
      // not a failure of the spacetime evolution. Return the default NaN
      // evaluation so AppendHorizon records found=0 and keeps evolving.
      return HorizonEvaluation{};
    }
  }
  HorizonEvaluation result;
  result.expansion.resize(points.size());
  result.flow.resize(points.size());
  Real expansion_integral = 0.0;
  Real expansion_squared_integral = 0.0;
  result.area = 0.0;
  result.coordinate_area = 0.0;
  result.spin_x = 0.0;
  result.spin_y = 0.0;
  result.spin_z = 0.0;
  for (std::size_t point = 0; point < points.size(); ++point) {
    result.expansion[point] = global[7 * point];
    result.flow[point] = global[7 * point + 2];
    const Real area = global[7 * point + 1] * points[point].weight;
    result.area += area;
    result.coordinate_area += points[point].weight * points[point].radius * points[point].radius;
    expansion_integral += area * result.expansion[point];
    expansion_squared_integral += area * result.expansion[point] * result.expansion[point];
    result.spin_x += area * global[7 * point + 3];
    result.spin_y += area * global[7 * point + 4];
    result.spin_z += area * global[7 * point + 5];
  }
  if (result.area > 0.0) {
    result.mean = expansion_integral / result.area;
    result.rms = sqrt(expansion_squared_integral / result.area);
  }
  result.spin_x /= 8.0 * kPi;
  result.spin_y /= 8.0 * kPi;
  result.spin_z /= 8.0 * kPi;
  result.integrated_expansion = expansion_integral;
  return result;
}

HorizonEvaluation EvaluateStarShapedSurfaceDispatch(Mesh* mesh,
                                                    const std::vector<HorizonSurfacePoint>& points,
                                                    const int order, const int flow_flag) {
  if (order == 2)
    return EvaluateStarShapedSurface<2>(mesh, points, flow_flag);
  if (order == 4)
    return EvaluateStarShapedSurface<3>(mesh, points, flow_flag);
  return EvaluateStarShapedSurface<4>(mesh, points, flow_flag);
}

void EnsureDirectory(const std::filesystem::path& directory) {
  if (parthenon::Globals::my_rank == 0)
    std::filesystem::create_directories(directory);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
#endif
}

void RefreshDiagnosticFields(Mesh* mesh, const bool synchronize_weyl,
                             const bool compute_horizon_derivatives) {
  const auto partitions = mesh->GetDefaultBlockPartitions();
  for (const auto& partition : partitions) {
    auto& base = mesh->mesh_data.Add("base", partition);
    // The final RK stage already converts the canonical `base` Z4c state to
    // ADM before timestep estimation.  Repeating that full-grid conversion
    // here both wastes a launch and risks obscuring the actual diagnostic
    // cost; consume the final-stage ADM state exactly as AthenaK FastFlow does.
    if (synchronize_weyl)
      ComputeWeylMeshTask(base.get());
  }
  Kokkos::fence();
  if (compute_horizon_derivatives) {
    const int order =
        mesh->packages.Get("numerical_relativity")->Param<int>("finite_difference_order");
    ComputeADMMetricDerivativesDispatch(mesh, order);
  }
  if (!synchronize_weyl)
    return;

  parthenon::TaskCollection collection;
  auto& region = collection.AddRegion(partitions.size());
  for (std::size_t partition = 0; partition < partitions.size(); ++partition) {
    auto& base = mesh->mesh_data.Add("base", partitions[partition]);
    auto& weyl =
        mesh->mesh_data.AddShallow("nr_weyl_diagnostic", base, std::vector<std::string>{"nr.weyl"});
    auto& tasks = region[partition];
    const auto receive = tasks.AddTask(
        parthenon::TaskID(0), parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, weyl);
    parthenon::AddBoundaryExchangeTasks(receive, tasks, weyl, mesh->multilevel,
                                        [](parthenon::TaskID dependency, parthenon::TaskList*,
                                           std::shared_ptr<MeshData<Real>>,
                                           bool) { return dependency; });
  }
  PARTHENON_REQUIRE(collection.Execute() == parthenon::TaskListStatus::complete,
                    "NR diagnostic Weyl boundary exchange failed");
}

void AppendWaveforms(Mesh* mesh, const std::shared_ptr<StateDescriptor>& package, const Real time,
                     const int cycle, const std::filesystem::path& directory) {
  const int geodesic_level = package->Param<int>("waveform_geodesic_level");
  const int interpolation_points = package->Param<int>("waveform_interpolation_points");
  const auto radii = package->Param<std::vector<Real>>("waveform_radii");
  const std::array<Real, 3> center{0.0, 0.0, 0.0};
  const GeodesicGrid angular_grid(geodesic_level);
  const auto path = directory / "nr_waveforms.csv";
  const bool needs_header = parthenon::Globals::my_rank == 0 && !std::filesystem::exists(path);
  std::ofstream stream;
  if (parthenon::Globals::my_rank == 0) {
    stream.open(path, std::ios::app);
    PARTHENON_REQUIRE(stream.good(), "could not open NR waveform output");
    if (needs_header) {
      stream << "# quantity=r*Psi4; tetrad=radial-polar-azimuthal Gram-Schmidt; "
                "spin_weight=-2; projection=integral(Psi4*conj(Y_lm)dOmega); "
                "angular_grid=rotated-geodesic; geodesic_level="
             << geodesic_level
             << "; interpolation=MeshData-native-block-tensor-Lagrange; interpolation_points="
             << interpolation_points << "; retarded_time=t-r\n";
      stream << "time,retarded_time,cycle,radius";
      for (int l = 2; l <= 8; ++l)
        for (int m = -l; m <= l; ++m)
          stream << ",re_l" << l << "_m" << m << ",im_l" << l << "_m" << m;
      stream << '\n';
    }
  }
  const auto waveform_directory = directory / "waveforms";
  EnsureDirectory(waveform_directory);
  for (const Real radius : radii) {
    const auto points = angular_grid.Surface(radius, center);
    const auto values = InterpolateWeyl(mesh, points, interpolation_points);
    std::vector<Real> modes(2 * 77, 0.0);
    for (std::size_t point = 0; point < points.size(); ++point) {
      const Real psi_real = values[3 * point];
      const Real psi_imag = values[3 * point + 1];
      int mode = 0;
      for (int l = 2; l <= 8; ++l) {
        for (int m = -l; m <= l; ++m, ++mode) {
          const auto harmonic = SpinMinusTwoHarmonic(l, m, points[point].theta, points[point].phi);
          modes[2 * mode] +=
              points[point].weight * (psi_real * harmonic.real + psi_imag * harmonic.imag);
          modes[2 * mode + 1] +=
              points[point].weight * (psi_imag * harmonic.real - psi_real * harmonic.imag);
        }
      }
    }
    if (parthenon::Globals::my_rank == 0) {
      stream << std::setprecision(17) << time << ',' << time - radius << ',' << cycle << ','
             << radius;
      for (const Real value : modes)
        stream << ',' << value;
      stream << '\n';

      std::ostringstream label;
      label << std::setfill('0') << std::setw(4) << radius;
      const auto real_path = waveform_directory / ("rpsi4_real_" + label.str() + ".txt");
      const auto imag_path = waveform_directory / ("rpsi4_imag_" + label.str() + ".txt");
      const bool real_header = !std::filesystem::exists(real_path);
      const bool imag_header = !std::filesystem::exists(imag_path);
      std::ofstream real_stream(real_path, std::ios::app);
      std::ofstream imag_stream(imag_path, std::ios::app);
      PARTHENON_REQUIRE(real_stream.good() && imag_stream.good(),
                        "could not open waveform output");
      if (real_header || imag_header) {
        std::ostringstream header;
        header << "# 1:time\t";
        int column = 2;
        for (int l = 2; l <= 8; ++l)
          for (int m = -l; m <= l; ++m)
            header << column++ << ':' << l << m << '\t';
        header << '\n';
        if (real_header)
          real_stream << header.str();
        if (imag_header)
          imag_stream << header.str();
      }
      real_stream << std::setprecision(15) << time << '\t';
      imag_stream << std::setprecision(15) << time << '\t';
      for (int mode = 0; mode < 77; ++mode) {
        real_stream << std::setprecision(15) << modes[2 * mode] << '\t';
        imag_stream << std::setprecision(15) << modes[2 * mode + 1] << '\t';
      }
      real_stream << '\n';
      imag_stream << '\n';
    }
  }
}

void AppendHorizon(Mesh* mesh, const std::shared_ptr<StateDescriptor>& package, const int horizon,
                   const Real time, const int cycle, const std::filesystem::path& directory) {
  const auto tracker = package->Param<std::vector<Real>>("tracker_position");
  const auto tracker_mass = package->Param<std::vector<Real>>("tracker_mass");
  const auto tracker_indices = package->Param<std::vector<int>>("horizon_tracker_indices");
  const auto stored_centers = package->Param<std::vector<Real>>("horizon_centers");
  const auto start_times = package->Param<std::vector<Real>>("horizon_start_times");
  const auto stop_times = package->Param<std::vector<Real>>("horizon_stop_times");
  const auto wait_for_punctures = package->Param<std::vector<int>>("horizon_wait_for_punctures");
  const auto mass_weighted_center =
      package->Param<std::vector<int>>("horizon_mass_weighted_center");
  const int horizon_count = package->Param<int>("horizon_count");
  PARTHENON_REQUIRE(horizon >= 0 && horizon < horizon_count &&
                        tracker_indices.size() == static_cast<std::size_t>(horizon_count) &&
                        stored_centers.size() == static_cast<std::size_t>(3 * horizon_count) &&
                        start_times.size() == static_cast<std::size_t>(horizon_count) &&
                        stop_times.size() == static_cast<std::size_t>(horizon_count) &&
                        wait_for_punctures.size() == static_cast<std::size_t>(horizon_count) &&
                        mass_weighted_center.size() == static_cast<std::size_t>(horizon_count),
                    "invalid apparent-horizon index or tracker map");
  if (time < start_times[horizon] || time > stop_times[horizon])
    return;
  if (wait_for_punctures[horizon] != 0) {
    Real maximum_distance = 0.0;
    Real total_mass = 0.0;
    for (std::size_t first = 0; first < tracker_mass.size(); ++first) {
      total_mass += tracker_mass[first];
      for (std::size_t second = first + 1; second < tracker_mass.size(); ++second) {
        Real distance_squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
          const Real delta = tracker[3 * first + axis] - tracker[3 * second + axis];
          distance_squared += delta * delta;
        }
        maximum_distance = fmax(maximum_distance, sqrt(distance_squared));
      }
    }
    if (!(maximum_distance < package->Param<Real>("horizon_merger_distance") * total_mass))
      return;
  }
  const int tracker_index = tracker_indices[horizon];
  std::array<Real, 3> center{stored_centers[3 * horizon], stored_centers[3 * horizon + 1],
                             stored_centers[3 * horizon + 2]};
  if (tracker_index >= 0) {
    PARTHENON_REQUIRE(3 * tracker_index + 2 < static_cast<int>(tracker.size()),
                      "apparent horizon references a missing compact-object tracker");
    for (int axis = 0; axis < 3; ++axis)
      center[axis] = tracker[3 * tracker_index + axis];
  } else if (mass_weighted_center[horizon] != 0) {
    Real total_mass = 0.0;
    center = {0.0, 0.0, 0.0};
    for (std::size_t object = 0; object < tracker_mass.size(); ++object) {
      total_mass += tracker_mass[object];
      for (int axis = 0; axis < 3; ++axis)
        center[axis] += tracker_mass[object] * tracker[3 * object + axis];
    }
    PARTHENON_REQUIRE(total_mass > 0.0, "cannot form a zero-mass horizon center");
    for (Real& coordinate : center)
      coordinate /= total_mass;
  }
  const int ntheta = package->Param<int>("horizon_ntheta");
  const int lmax = package->Param<int>("horizon_lmax");
  const int maximum_iterations = package->Param<int>("horizon_iterations");
  const Real gain = package->Param<Real>("horizon_flow_gain");
  const Real mass_tolerance = package->Param<Real>("horizon_mass_tolerance");
  const Real hmean_limit = package->Param<Real>("horizon_hmean_limit");
  const Real expand_guess = package->Param<Real>("horizon_expand_guess");
  const int flow_flag = package->Param<int>("horizon_flow_flag");
  const bool output_grid = package->Param<bool>("horizon_output_grid");
  const int order = package->Param<int>("finite_difference_order");
  auto* stored_radii = package->MutableParam<std::vector<Real>>("horizon_radii");
  auto* stored_found = package->MutableParam<std::vector<int>>("horizon_found");
  const int modes = HarmonicModeCount(lmax);
  PARTHENON_REQUIRE(stored_radii->size() == static_cast<std::size_t>(horizon_count) &&
                        stored_found->size() == stored_radii->size(),
                    "invalid fast-flow restart state");
  // AthenaK resets every l>0 coefficient at each search and retains only the
  // previous a_00 as the next initial guess.
  std::vector<Real> coefficients(modes, 0.0);
  Real initial_radius = (*stored_radii)[horizon];
  if ((*stored_found)[horizon] != 0) {
    initial_radius *= expand_guess;
  } else if (tracker_index >= 0) {
    PARTHENON_REQUIRE(tracker_mass.size() * 3 == tracker.size(),
                      "invalid compact-object tracker mass state");
    Real maximum_distance = 0.0;
    for (int other = 0; other < static_cast<int>(tracker_mass.size()); ++other) {
      Real distance_squared = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const Real delta = tracker[3 * tracker_index + axis] - tracker[3 * other + axis];
        distance_squared += delta * delta;
      }
      maximum_distance = fmax(maximum_distance, sqrt(distance_squared));
    }
    const Real mass = tracker_mass[tracker_index];
    initial_radius = fmax(0.5 * mass, fmin(mass, 0.5 * maximum_distance));
  }
  coefficients[0] = initial_radius * sqrt(4.0 * kPi);
  const auto& angular = GetHorizonAngularData(ntheta, lmax);
  const auto& degrees = angular.degrees;
  const auto& unit_sphere = angular.unit_sphere;
  const auto& harmonics = angular.harmonics;
  HorizonEvaluation result;
  std::vector<HorizonSurfacePoint> points;
  bool found = false;
  bool valid = true;
  int iterations = 0;
  Real mass = 0.0;
  Real previous_mass = 0.0;
  std::vector<std::array<Real, 9>> iteration_trace;
  iteration_trace.reserve(maximum_iterations);
  for (; iterations < maximum_iterations; ++iterations) {
    points = BuildHorizonSurface(unit_sphere, harmonics, coefficients, center);
    const auto radius_range =
        std::minmax_element(points.begin(), points.end(),
                            [](const HorizonSurfacePoint& left, const HorizonSurfacePoint& right) {
                              return left.radius < right.radius;
                            });
    if (radius_range.first == points.end() || !(radius_range.first->radius > 1.0e-6) ||
        !std::isfinite(radius_range.first->radius) || !std::isfinite(radius_range.second->radius)) {
      valid = false;
      break;
    }
    result = EvaluateStarShapedSurfaceDispatch(mesh, points, order, flow_flag);
    if (!std::isfinite(result.area) || !std::isfinite(result.mean) || !std::isfinite(result.rms)) {
      valid = false;
      break;
    }
    previous_mass = mass;
    mass = result.area > 0.0 ? sqrt(result.area / (16.0 * kPi)) : 0.0;
    const Real mean_radius = coefficients[0] / sqrt(4.0 * kPi);
    iteration_trace.push_back({static_cast<Real>(iterations), result.area, mass, mean_radius,
                               radius_range.first->radius, radius_range.second->radius,
                               result.integrated_expansion, result.rms * result.rms, result.rms});
    if (!std::isfinite(result.integrated_expansion) ||
        fabs(result.integrated_expansion) > hmean_limit || mean_radius < 0.0 || mass < 1.0e-10) {
      valid = false;
      break;
    }
    // AthenaK's native fast-flow convergence criterion is the change in
    // irreducible mass between consecutive surface iterations.
    if (fabs(previous_mass - mass) < mass_tolerance) {
      found = true;
      ++iterations;
      break;
    }
    const Real alpha = gain;
    const Real beta = 0.5 * gain;
    const Real flow_a = alpha / (lmax * (lmax + 1.0)) + beta;
    const Real flow_b = beta / alpha;
    for (int mode = 0; mode < modes; ++mode) {
      Real projection = 0.0;
      for (std::size_t point = 0; point < points.size(); ++point)
        projection +=
            points[point].weight * result.flow[point] * harmonics[point * modes + mode].value;
      coefficients[mode] -=
          flow_a / (1.0 + flow_b * degrees[mode] * (degrees[mode] + 1.0)) * projection;
    }
  }
  if (valid && found) {
    (*stored_radii)[horizon] = coefficients[0] / sqrt(4.0 * kPi);
    (*stored_found)[horizon] = 1;
  }
  Real minimum_radius = std::numeric_limits<Real>::quiet_NaN();
  Real maximum_radius = std::numeric_limits<Real>::quiet_NaN();
  if (!points.empty()) {
    const auto radius_range =
        std::minmax_element(points.begin(), points.end(),
                            [](const HorizonSurfacePoint& left, const HorizonSurfacePoint& right) {
                              return left.radius < right.radius;
                            });
    minimum_radius = radius_range.first->radius;
    maximum_radius = radius_range.second->radius;
  }
  const Real mean_radius = coefficients[0] / sqrt(4.0 * kPi);
  const Real irreducible_mass =
      result.area > 0.0 ? sqrt(result.area / (16.0 * kPi)) : std::numeric_limits<Real>::quiet_NaN();
  const Real spin = sqrt(result.spin_x * result.spin_x + result.spin_y * result.spin_y +
                         result.spin_z * result.spin_z);
  const Real christodoulou_mass =
      irreducible_mass > 0.0 ? sqrt(irreducible_mass * irreducible_mass +
                                    0.25 * (spin / irreducible_mass) * (spin / irreducible_mass))
                             : std::numeric_limits<Real>::quiet_NaN();
  if (parthenon::Globals::my_rank == 0) {
    const std::string suffix = horizon_count == 1 ? "" : "_" + std::to_string(horizon);
    const auto path = directory / ("nr_horizon" + suffix + ".csv");
    const bool needs_header = !std::filesystem::exists(path);
    std::ofstream stream(path, std::ios::app);
    PARTHENON_REQUIRE(stream.good(), "could not open NR horizon output");
    if (needs_header) {
      stream << "# finder=real-spherical-harmonic-standard-fast-flow; "
                "implementation=fast-flow; "
                "interpolation=native-block-2nghost-tensor-Lagrange; "
                "metric_derivatives=grid-centered-finite-difference-then-interpolate; "
                "convergence=successive-irreducible-mass-difference; "
                "surface=F(r,theta,phi)=r-h(theta,phi); "
                "flow=expansion*norm(grad(F)); "
                "expansion=D_i(s^i)+K_ij*s^i*s^j-K\n";
      stream << "time,cycle,found,iterations,center_x,center_y,center_z,radius,area,"
                "coordinate_area,irreducible_mass,christodoulou_mass,spin_x,spin_y,spin_z,spin,"
                "hmean_integral,hrms_mean_square,mean_expansion,rms_expansion,min_radius,"
                "max_radius,lmax\n";
    }
    stream << std::setprecision(17) << time << ',' << cycle << ',' << (found ? 1 : 0) << ','
           << iterations << ',' << center[0] << ',' << center[1] << ',' << center[2] << ','
           << mean_radius << ',' << result.area << ',' << result.coordinate_area << ','
           << irreducible_mass << ',' << christodoulou_mass << ',' << result.spin_x << ','
           << result.spin_y << ',' << result.spin_z << ',' << spin << ','
           << result.integrated_expansion << ',' << result.rms * result.rms << ',' << result.mean
           << ',' << result.rms << ',' << minimum_radius << ',' << maximum_radius << ',' << lmax
           << '\n';

    const auto iteration_path = directory / ("nr_horizon_iterations" + suffix + ".csv");
    const bool iterations_need_header = !std::filesystem::exists(iteration_path);
    std::ofstream iteration_stream(iteration_path, std::ios::app);
    PARTHENON_REQUIRE(iteration_stream.good(), "could not open NR horizon-iteration output");
    if (iterations_need_header)
      iteration_stream << "time,cycle,iteration,area,irreducible_mass,mean_radius,min_radius,"
                          "max_radius,hmean_integral,hrms_mean_square,rms_expansion\n";
    for (const auto& row : iteration_trace)
      iteration_stream << std::setprecision(17) << time << ',' << cycle << ','
                       << static_cast<int>(row[0]) << ',' << row[1] << ',' << row[2] << ','
                       << row[3] << ',' << row[4] << ',' << row[5] << ',' << row[6] << ',' << row[7]
                       << ',' << row[8] << '\n';

    if (found) {
      const auto coefficient_path = directory / ("nr_horizon_coefficients" + suffix + ".csv");
      const bool coefficients_need_header = !std::filesystem::exists(coefficient_path);
      std::ofstream coefficient_stream(coefficient_path, std::ios::app);
      PARTHENON_REQUIRE(coefficient_stream.good(), "could not open NR horizon-coefficient output");
      if (coefficients_need_header)
        coefficient_stream << "time,cycle,l,m,basis,coefficient\n";
      int mode = 0;
      for (int l = 0; l <= lmax; ++l) {
        coefficient_stream << std::setprecision(17) << time << ',' << cycle << ',' << l << ",0,m0,"
                           << coefficients[mode++] << '\n';
        for (int m = 1; m <= l; ++m) {
          coefficient_stream << std::setprecision(17) << time << ',' << cycle << ',' << l << ','
                             << m << ",cos," << coefficients[mode++] << '\n';
          coefficient_stream << std::setprecision(17) << time << ',' << cycle << ',' << l << ','
                             << m << ",sin," << coefficients[mode++] << '\n';
        }
      }

      if (output_grid) {
        const auto surface_path = directory / ("nr_horizon_surface" + suffix + ".csv");
        const bool surface_needs_header = !std::filesystem::exists(surface_path);
        std::ofstream surface_stream(surface_path, std::ios::app);
        PARTHENON_REQUIRE(surface_stream.good(), "could not open NR horizon-surface output");
        if (surface_needs_header)
          surface_stream << "time,cycle,theta,phi,x,y,z,radius,expansion\n";
        for (std::size_t point = 0; point < points.size(); ++point)
          surface_stream << std::setprecision(17) << time << ',' << cycle << ','
                         << points[point].theta << ',' << points[point].phi << ','
                         << points[point].x << ',' << points[point].y << ',' << points[point].z
                         << ',' << points[point].radius << ',' << result.expansion[point] << '\n';
      }
    }
  }
}

} // namespace

void RunPostStepDiagnostics(Mesh* mesh, const Real time, const int cycle) {
  const auto package = mesh->packages.Get("numerical_relativity");
  const bool waveform_enabled = package->Param<bool>("waveform_enabled");
  const bool horizon_enabled = package->Param<bool>("horizon_enabled");
  if (!waveform_enabled && !horizon_enabled)
    return;
  const std::filesystem::path directory(package->Param<std::string>("diagnostic_directory"));
  EnsureDirectory(directory);
  const Real tolerance = 64.0 * std::numeric_limits<Real>::epsilon() * fmax(1.0, fabs(time));
  auto* waveform_last =
      waveform_enabled ? package->MutableParam<Real>("waveform_last_output_time") : nullptr;
  bool waveform_due = false;
  if (waveform_enabled) {
    // AthenaK intentionally schedules waveform work with single-precision time
    // comparisons so restart and output decisions are independent of sub-ULP
    // differences accumulated by the evolution driver.
    const float time_32 = static_cast<float>(time);
    const float next_32 = static_cast<float>(*waveform_last + package->Param<Real>("waveform_dt"));
    waveform_due = time_32 >= next_32 || time_32 == 0.0F;
    if (waveform_due)
      *waveform_last = static_cast<Real>(time_32);
  }
  bool horizon_due = false;
  if (horizon_enabled) {
    const auto start_times = package->Param<std::vector<Real>>("horizon_start_times");
    const auto stop_times = package->Param<std::vector<Real>>("horizon_stop_times");
    PARTHENON_REQUIRE(start_times.size() == stop_times.size(),
                      "invalid apparent-horizon time controls");
    for (std::size_t horizon = 0; horizon < start_times.size(); ++horizon)
      horizon_due = horizon_due || (time + tolerance >= start_times[horizon] &&
                                    time - tolerance <= stop_times[horizon]);
  }
  if (!waveform_due && !horizon_due)
    return;

  // Rebuild diagnostics from the canonical final-stage state. MeshData stage
  // buffers can share derived storage, so using a value left by an earlier RK
  // stage would label the waveform one step ahead of its actual time.
  RefreshDiagnosticFields(mesh, waveform_due, horizon_due);

  if (waveform_due) {
    AppendWaveforms(mesh, package, time, cycle, directory);
  }
  if (horizon_due) {
    const int horizon_count = package->Param<int>("horizon_count");
    for (int horizon = 0; horizon < horizon_count; ++horizon)
      AppendHorizon(mesh, package, horizon, time, cycle, directory);
  }
}

} // namespace pangu::nr
