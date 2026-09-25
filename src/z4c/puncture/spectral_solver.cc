#include "z4c/puncture/spectral_solver.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace pangu::nr::puncture {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

} // namespace

CompactifiedSpectralGrid::CompactifiedSpectralGrid(const SpectralGridOptions options)
    : points_(options.points), scale_(options.scale) {
  if (points_ < 4) throw std::invalid_argument("puncture spectral grid requires at least 4 points");
  if (!(scale_ > 0.0) || !std::isfinite(scale_))
    throw std::invalid_argument("puncture spectral compactification scale must be positive");

  computational_.resize(points_);
  physical_.resize(points_);
  first_derivative_.assign(static_cast<std::size_t>(points_) * points_, 0.0);
  second_derivative_.assign(static_cast<std::size_t>(points_) * points_, 0.0);
  first_metric_.resize(points_);
  second_metric_.resize(points_);

  for (int i = 0; i < points_; ++i) {
    const double q = std::cos(kPi * static_cast<double>(i) / static_cast<double>(points_ - 1));
    computational_[i] = q;
    const double one_minus_q2 = std::max(0.0, 1.0 - q * q);
    physical_[i] = one_minus_q2 == 0.0
                       ? std::copysign(std::numeric_limits<double>::infinity(), q)
                       : scale_ * q / std::sqrt(one_minus_q2);
    first_metric_[i] = std::pow(one_minus_q2, 1.5) / scale_;
    second_metric_[i] = -3.0 * q * one_minus_q2 * one_minus_q2 / (scale_ * scale_);
  }

  // Chebyshev--Lobatto first-derivative matrix in descending node order.
  for (int i = 0; i < points_; ++i) {
    const double ci = (i == 0 || i == points_ - 1) ? 2.0 : 1.0;
    for (int j = 0; j < points_; ++j) {
      if (i == j) continue;
      const double cj = (j == 0 || j == points_ - 1) ? 2.0 : 1.0;
      const double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
      first_derivative_[static_cast<std::size_t>(i) * points_ + j] =
          sign * ci / (cj * (computational_[i] - computational_[j]));
    }
  }
  // Determine every diagonal from the exact derivative of a constant. This is
  // algebraically equivalent to the closed forms and is less sensitive to a
  // node-ordering convention.
  for (int i = 0; i < points_; ++i) {
    double off_diagonal_sum = 0.0;
    for (int j = 0; j < points_; ++j)
      if (i != j)
        off_diagonal_sum += first_derivative_[static_cast<std::size_t>(i) * points_ + j];
    first_derivative_[static_cast<std::size_t>(i) * points_ + i] = -off_diagonal_sum;
  }

  for (int i = 0; i < points_; ++i)
    for (int j = 0; j < points_; ++j)
      for (int k = 0; k < points_; ++k)
        second_derivative_[static_cast<std::size_t>(i) * points_ + j] +=
            first_derivative_[static_cast<std::size_t>(i) * points_ + k] *
            first_derivative_[static_cast<std::size_t>(k) * points_ + j];
}

std::size_t CompactifiedSpectralGrid::size() const {
  return static_cast<std::size_t>(points_) * points_ * points_;
}

std::size_t CompactifiedSpectralGrid::Index(const int i, const int j, const int k) const {
  return static_cast<std::size_t>(i) +
         static_cast<std::size_t>(points_) *
             (static_cast<std::size_t>(j) + static_cast<std::size_t>(points_) * k);
}

bool CompactifiedSpectralGrid::IsBoundary(const int i, const int j, const int k) const {
  return i == 0 || j == 0 || k == 0 || i == points_ - 1 || j == points_ - 1 ||
         k == points_ - 1;
}

double CompactifiedSpectralGrid::ComputationalCoordinate(const int i) const {
  return computational_.at(i);
}

double CompactifiedSpectralGrid::PhysicalCoordinate(const int i) const {
  return physical_.at(i);
}

void CompactifiedSpectralGrid::ApplyLaplacian(const std::vector<double>& input,
                                              std::vector<double>& output) const {
  if (input.size() != size()) throw std::invalid_argument("puncture spectral field has wrong size");
  output.assign(size(), 0.0);
  for (int k = 0; k < points_; ++k) {
    for (int j = 0; j < points_; ++j) {
      for (int i = 0; i < points_; ++i) {
        const std::size_t center = Index(i, j, k);
        if (IsBoundary(i, j, k)) {
          output[center] = input[center];
          continue;
        }
        double first[3]{};
        double second[3]{};
        for (int sample = 0; sample < points_; ++sample) {
          first[0] += first_derivative_[static_cast<std::size_t>(i) * points_ + sample] *
                      input[Index(sample, j, k)];
          second[0] += second_derivative_[static_cast<std::size_t>(i) * points_ + sample] *
                       input[Index(sample, j, k)];
          first[1] += first_derivative_[static_cast<std::size_t>(j) * points_ + sample] *
                      input[Index(i, sample, k)];
          second[1] += second_derivative_[static_cast<std::size_t>(j) * points_ + sample] *
                       input[Index(i, sample, k)];
          first[2] += first_derivative_[static_cast<std::size_t>(k) * points_ + sample] *
                      input[Index(i, j, sample)];
          second[2] += second_derivative_[static_cast<std::size_t>(k) * points_ + sample] *
                       input[Index(i, j, sample)];
        }
        output[center] = first_metric_[i] * first_metric_[i] * second[0] +
                         second_metric_[i] * first[0] +
                         first_metric_[j] * first_metric_[j] * second[1] +
                         second_metric_[j] * first[1] +
                         first_metric_[k] * first_metric_[k] * second[2] +
                         second_metric_[k] * first[2];
      }
    }
  }
}

} // namespace pangu::nr::puncture
