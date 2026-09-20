#include "radiation/geodesic_grid.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace pangu::radiation {
namespace {

using parthenon::Real;
constexpr Real kPi = 3.141592653589793238462643383279502884;

class GeodesicGrid {
public:
  explicit GeodesicGrid(const int level) : level_(level) {
    angles_ = 10 * level_ * level_ + 2;
    const int rows = level_ + 2;
    const int columns = 2 * level_ + 2;
    normals_.assign(5 * rows * columns * 3, 0.0);
    pole_normals_.assign(6, 0.0);
    indices_.assign(5 * rows * columns, 0);
    pole_indices_.assign(2, 0);
    weights_.assign(angles_, 0.0);

    const Real sin_angle = 2.0 / std::sqrt(5.0);
    const Real cos_angle = 1.0 / std::sqrt(5.0);
    const Real p1[3]{0.0, 0.0, 1.0};
    const Real p2[3]{sin_angle, 0.0, cos_angle};
    const Real p3[3]{sin_angle * std::cos(0.2 * kPi), sin_angle * std::sin(0.2 * kPi), -cos_angle};
    const Real p4[3]{sin_angle * std::cos(-0.4 * kPi), sin_angle * std::sin(-0.4 * kPi), cos_angle};
    const Real p5[3]{sin_angle * std::cos(-0.2 * kPi), sin_angle * std::sin(-0.2 * kPi),
                     -cos_angle};
    const Real p6[3]{0.0, 0.0, -1.0};
    Pole(0, 2) = 1.0;
    Pole(1, 2) = -1.0;

    int row = 1;
    for (int l = 0; l < level_; ++l) {
      int column = 1;
      for (int m = l; m < level_; ++m)
        SetNormalizedCombination(row, column++, m - l + 1, p2, level_ - m - 1, p1, l, p4);
      for (int m = level_ - l; m < level_; ++m)
        SetNormalizedCombination(row, column++, level_ - l, p2, m - level_ + l + 1, p5,
                                 level_ - m - 1, p4);
      for (int m = l; m < level_; ++m)
        SetNormalizedCombination(row, column++, m - l + 1, p3, level_ - m - 1, p2, l, p5);
      for (int m = level_ - l; m < level_; ++m)
        SetNormalizedCombination(row, column++, level_ - l, p3, m - level_ + l + 1, p6,
                                 level_ - m - 1, p5);
      ++row;
    }

    for (int patch = 1; patch < 5; ++patch) {
      for (int l = 1; l < level_ + 1; ++l) {
        for (int m = 1; m < 2 * level_ + 1; ++m) {
          const Real x = Normal(0, l, m, 0);
          const Real y = Normal(0, l, m, 1);
          const Real z = Normal(0, l, m, 2);
          Normal(patch, l, m, 0) =
              x * std::cos(patch * 0.4 * kPi) + y * std::sin(patch * 0.4 * kPi);
          Normal(patch, l, m, 1) =
              y * std::cos(patch * 0.4 * kPi) - x * std::sin(patch * 0.4 * kPi);
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

    for (int point = 0; point < angles_; ++point)
      SolidAngle(point, weights_[point]);
  }

  int Angles() const { return angles_; }
  Real Weight(const int point) const { return weights_[point]; }

  std::array<Real, 3> Direction(const int point) const {
    std::array<Real, 3> result{};
    GridCartPosition(point, result[0], result[1], result[2]);
    return result;
  }

  void RotateOptimally() {
    Real angles[2]{0.0, 0.0};
    OptimalAngles(angles);
    RotateGrid(angles[0], angles[1]);
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
    const Real norm = std::sqrt(norm_squared);
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
    const Real a = std::sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2) + (z3 - z2) * (z3 - z2));
    const Real b = std::sqrt((x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3) + (z1 - z3) * (z1 - z3));
    const Real c = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1));
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
    const Real norm = std::sqrt(xc * xc + yc * yc + zc * zc);
    x = xc / norm;
    y = yc / norm;
    z = zc / norm;
  }

  void SolidAngle(const int point, Real& weight) const {
    int count = 0;
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
      const Real numerator = std::fabs(x0 * (cy1 * cz2 - cz1 * cy2) + y0 * (cz1 * cx2 - cx1 * cz2) +
                                       z0 * (cx1 * cy2 - cy1 * cx2));
      weight += 2.0 * std::atan(numerator / (1.0 + dot01 + dot02 + dot12));
    }
  }

  Real ArcLength(const int first, const int second) const {
    Real x1, y1, z1, x2, y2, z2;
    GridCartPosition(first, x1, y1, z1);
    GridCartPosition(second, x2, y2, z2);
    return std::acos(
        std::clamp(x1 * x2 + y1 * y2 + z1 * z2, static_cast<Real>(-1.0), static_cast<Real>(1.0)));
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
        const Real kx = -std::sin(psi);
        const Real ky = std::cos(psi);
        Real current = 1.0;
        for (int point = 0; point < angles_; ++point) {
          Real x, y, z;
          GridCartPosition(point, x, y, z);
          const Real rx = x * std::cos(zeta) + ky * z * std::sin(zeta) +
                          kx * (kx * x + ky * y) * (1.0 - std::cos(zeta));
          const Real ry = y * std::cos(zeta) - kx * z * std::sin(zeta) +
                          ky * (kx * x + ky * y) * (1.0 - std::cos(zeta));
          const Real rz = z * std::cos(zeta) + (kx * y - ky * x) * std::sin(zeta);
          current = std::min(current, std::fabs(rx));
          current = std::min(current, std::fabs(ry));
          current = std::min(current, std::fabs(rz));
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
    const Real kx = -std::sin(psi);
    const Real ky = std::cos(psi);
    const auto rotate = [&](const Real x, const Real y, const Real z) {
      return std::array<Real, 3>{x * std::cos(zeta) + ky * z * std::sin(zeta) +
                                     kx * (kx * x + ky * y) * (1.0 - std::cos(zeta)),
                                 y * std::cos(zeta) - kx * z * std::sin(zeta) +
                                     ky * (kx * x + ky * y) * (1.0 - std::cos(zeta)),
                                 z * std::cos(zeta) + (kx * y - ky * x) * std::sin(zeta)};
    };
    for (int patch = 0; patch < 5; ++patch) {
      for (int l = 0; l < level_; ++l) {
        for (int m = 0; m < 2 * level_; ++m) {
          const auto result = rotate(Normal(patch, l + 1, m + 1, 0), Normal(patch, l + 1, m + 1, 1),
                                     Normal(patch, l + 1, m + 1, 2));
          for (int component = 0; component < 3; ++component)
            Normal(patch, l + 1, m + 1, component) = result[component];
        }
      }
    }
    for (int pole = 0; pole < 2; ++pole) {
      const auto result = rotate(Pole(pole, 0), Pole(pole, 1), Pole(pole, 2));
      for (int component = 0; component < 3; ++component)
        Pole(pole, component) = result[component];
    }
    FillNormalGhosts();
  }
};

} // namespace

AngularQuadrature BuildGeodesicQuadrature(const int level, const bool rotate) {
  PARTHENON_REQUIRE(level >= 0, "radiation/nlevel must be non-negative");
  AngularQuadrature quadrature;
  if (level == 0) {
    constexpr int angles = 8;
    quadrature.directions.resize(angles);
    quadrature.solid_angles.assign(angles, 4.0 * kPi / angles);
    const Real zeta[2]{kPi / 4.0, 3.0 * kPi / 4.0};
    const Real psi[4]{kPi / 4.0, 3.0 * kPi / 4.0, 5.0 * kPi / 4.0, 7.0 * kPi / 4.0};
    for (int z = 0, angle = 0; z < 2; ++z) {
      for (int p = 0; p < 4; ++p, ++angle) {
        quadrature.directions[angle] = {std::sin(zeta[z]) * std::cos(psi[p]) * std::sqrt(4.0 / 3.0),
                                        std::sin(zeta[z]) * std::sin(psi[p]) * std::sqrt(4.0 / 3.0),
                                        std::cos(zeta[z]) * std::sqrt(2.0 / 3.0)};
      }
    }
    return quadrature;
  }

  GeodesicGrid grid(level);
  if (rotate)
    grid.RotateOptimally();
  quadrature.directions.reserve(grid.Angles());
  quadrature.solid_angles.reserve(grid.Angles());
  for (int angle = 0; angle < grid.Angles(); ++angle) {
    quadrature.directions.push_back(grid.Direction(angle));
    quadrature.solid_angles.push_back(grid.Weight(angle));
  }
  return quadrature;
}

} // namespace pangu::radiation
