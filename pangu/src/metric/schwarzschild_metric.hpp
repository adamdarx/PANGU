#pragma once

#include <basic_types.hpp>

KOKKOS_INLINE_FUNCTION
void ComputeSchwarzschildMetric(const parthenon::Real x1, const parthenon::Real x2,
                                const parthenon::Real x3, parthenon::Real gcov[4][4],
                                parthenon::Real gcon[4][4]) {
  const parthenon::Real radius = Kokkos::sqrt(x1 * x1 + x2 * x2 + x3 * x3) + 1e-12;
  const parthenon::Real factor = 2.0 / radius;

  const parthenon::Real lx = x1 / radius;
  const parthenon::Real ly = x2 / radius;
  const parthenon::Real lz = x3 / radius;

  const parthenon::Real l_cov[4] = {1.0, lx, ly, lz};
  const parthenon::Real l_con[4] = {-1.0, lx, ly, lz};

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
