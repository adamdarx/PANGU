#pragma once

#include <basic_types.hpp>

KOKKOS_INLINE_FUNCTION
void ComputeKerrMetric(const parthenon::Real x, const parthenon::Real y,
                       const parthenon::Real z, const parthenon::Real a,
                       parthenon::Real gcov[4][4], parthenon::Real gcon[4][4]) {
  const parthenon::Real radius = Kokkos::sqrt(x * x + y * y + z * z);
  const parthenon::Real ks_radius =
      Kokkos::sqrt((radius * radius - a * a +
                    Kokkos::sqrt((radius * radius - a * a) *
                                     (radius * radius - a * a) +
                                 4.0 * a * a * z * z)) /
                   2.0) +
      1e-12;

  const parthenon::Real l_cov[4] = {
      1.0,
      (ks_radius * x + a * y) / (ks_radius * ks_radius + a * a),
      (ks_radius * y - a * x) / (ks_radius * ks_radius + a * a),
      z / ks_radius,
  };
  const parthenon::Real l_con[4] = {-1.0, l_cov[1], l_cov[2], l_cov[3]};

  const parthenon::Real factor =
      2.0 * ks_radius * ks_radius * ks_radius /
      (ks_radius * ks_radius * ks_radius * ks_radius + a * a * z * z);

  gcov[0][0] = factor * l_cov[0] * l_cov[0] - 1.0;
  gcov[0][1] = factor * l_cov[0] * l_cov[1];
  gcov[0][2] = factor * l_cov[0] * l_cov[2];
  gcov[0][3] = factor * l_cov[0] * l_cov[3];
  gcov[1][0] = gcov[0][1];
  gcov[1][1] = factor * l_cov[1] * l_cov[1] + 1.0;
  gcov[1][2] = factor * l_cov[1] * l_cov[2];
  gcov[1][3] = factor * l_cov[1] * l_cov[3];
  gcov[2][0] = gcov[0][2];
  gcov[2][1] = gcov[1][2];
  gcov[2][2] = factor * l_cov[2] * l_cov[2] + 1.0;
  gcov[2][3] = factor * l_cov[2] * l_cov[3];
  gcov[3][0] = gcov[0][3];
  gcov[3][1] = gcov[1][3];
  gcov[3][2] = gcov[2][3];
  gcov[3][3] = factor * l_cov[3] * l_cov[3] + 1.0;

  gcon[0][0] = -factor * l_con[0] * l_con[0] - 1.0;
  gcon[0][1] = -factor * l_con[0] * l_con[1];
  gcon[0][2] = -factor * l_con[0] * l_con[2];
  gcon[0][3] = -factor * l_con[0] * l_con[3];
  gcon[1][0] = gcon[0][1];
  gcon[1][1] = -factor * l_con[1] * l_con[1] + 1.0;
  gcon[1][2] = -factor * l_con[1] * l_con[2];
  gcon[1][3] = -factor * l_con[1] * l_con[3];
  gcon[2][0] = gcon[0][2];
  gcon[2][1] = gcon[1][2];
  gcon[2][2] = -factor * l_con[2] * l_con[2] + 1.0;
  gcon[2][3] = -factor * l_con[2] * l_con[3];
  gcon[3][0] = gcon[0][3];
  gcon[3][1] = gcon[1][3];
  gcon[3][2] = gcon[2][3];
  gcon[3][3] = -factor * l_con[3] * l_con[3] + 1.0;
}
