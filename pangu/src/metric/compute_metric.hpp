#pragma once

#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"
#include "kerr_metric.hpp"
#include "schwarzschild_metric.hpp"

KOKKOS_INLINE_FUNCTION
void ComputeMinkowskiMetric(parthenon::Real gcov[4][4],
                           parthenon::Real gcon[4][4]) {
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      gcov[mu][nu] = 0.0;
      gcon[mu][nu] = 0.0;
    }
  }
  gcov[0][0] = -1.0;
  gcov[1][1] = 1.0;
  gcov[2][2] = 1.0;
  gcov[3][3] = 1.0;
  gcon[0][0] = -1.0;
  gcon[1][1] = 1.0;
  gcon[2][2] = 1.0;
  gcon[3][3] = 1.0;
}

KOKKOS_INLINE_FUNCTION
void FlattenMetric(const parthenon::Real g[4][4],
                   parthenon::Real out[MetricTensorComponentNumber]) {
  out[Metric00] = g[0][0];
  out[Metric01] = g[0][1];
  out[Metric02] = g[0][2];
  out[Metric03] = g[0][3];
  out[Metric11] = g[1][1];
  out[Metric12] = g[1][2];
  out[Metric13] = g[1][3];
  out[Metric22] = g[2][2];
  out[Metric23] = g[2][3];
  out[Metric33] = g[3][3];
}

parthenon::TaskStatus ComputeMetric(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  const auto meshblock_pointer = resource->GetBlockPointer();
  const auto package = meshblock_pointer->packages.Get("PANGU");
  const auto mode = package->Param<std::string>("Mode");

  if (mode != "GR") {
    return TaskStatus::complete;
  }

  const auto metric_name = package->Param<std::string>("MetricName");
  const auto metric_spin = package->Param<Real>("MetricSpin");
  const int metric_type =
      (metric_name == "Minkowski") ? 0 : ((metric_name == "Schwarzschild") ? 1 : 2);

  const auto bound_x1 = meshblock_pointer->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto bound_x2 = meshblock_pointer->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto bound_x3 = meshblock_pointer->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto &coords = meshblock_pointer->coords;

  PackIndexMap metric_lower_index_map;
  PackIndexMap metric_upper_index_map;
  const auto metric_lower =
      resource->PackVariables({"MetricLower"}, metric_lower_index_map);
  const auto metric_upper =
      resource->PackVariables({"MetricUpper"}, metric_upper_index_map);

  meshblock_pointer->par_for(
      PARTHENON_AUTO_LABEL, bound_x3.s, bound_x3.e, bound_x2.s, bound_x2.e,
      bound_x1.s, bound_x1.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x1 = coords.Xc<X1DIR>(i);
        const Real x2 = coords.Xc<X2DIR>(j);
        const Real x3 = coords.Xc<X3DIR>(k);

        Real gcov[4][4];
        Real gcon[4][4];

        if (metric_type == 0) {
          ComputeMinkowskiMetric(gcov, gcon);
        } else if (metric_type == 1) {
          ComputeSchwarzschildMetric(x1, x2, x3, gcov, gcon);
        } else {
          ComputeKerrMetric(x1, x2, x3, metric_spin, gcov, gcon);
        }

        Real lower_flat[MetricTensorComponentNumber];
        Real upper_flat[MetricTensorComponentNumber];
        FlattenMetric(gcov, lower_flat);
        FlattenMetric(gcon, upper_flat);

        for (int n = 0; n < MetricTensorComponentNumber; ++n) {
          metric_lower(n, k, j, i) = lower_flat[n];
          metric_upper(n, k, j, i) = upper_flat[n];
        }
      });

  return TaskStatus::complete;
}
