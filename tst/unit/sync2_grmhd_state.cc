#include <algorithm>
#include <cmath>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "riemann/registry.h"
#include "z4c/coupling/sync_grhd.h"
#include "z4c/coupling/sync_grmhd.h"

namespace {

pangu::geometry::MetricPoint BuildMetric() {
  using Real = parthenon::Real;
  pangu::geometry::MetricPoint metric{};
  const Real spatial[3][3]{{1.31, 0.07, -0.03},
                           {0.07, 0.94, 0.05},
                           {-0.03, 0.05, 1.17}};
  const Real det =
      spatial[0][0] * (spatial[1][1] * spatial[2][2] - spatial[1][2] * spatial[2][1]) -
      spatial[0][1] * (spatial[1][0] * spatial[2][2] - spatial[1][2] * spatial[2][0]) +
      spatial[0][2] * (spatial[1][0] * spatial[2][1] - spatial[1][1] * spatial[2][0]);
  metric.spatial_det = det;
  metric.lapse = 0.83;
  metric.shift[0] = 0.04;
  metric.shift[1] = -0.025;
  metric.shift[2] = 0.015;
  Real inverse[3][3]{};
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      metric.lower[first + 1][second + 1] = spatial[first][second];
  pangu::nr::InvertSyncSpatialMetric(metric, inverse);
  Real lowered_shift[3]{};
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      lowered_shift[first] += spatial[first][second] * metric.shift[second];
  metric.lower[0][0] = -metric.lapse * metric.lapse;
  for (int axis = 0; axis < 3; ++axis) {
    metric.lower[0][0] += lowered_shift[axis] * metric.shift[axis];
    metric.lower[0][axis + 1] = lowered_shift[axis];
    metric.lower[axis + 1][0] = lowered_shift[axis];
  }
  const Real inverse_lapse2 = 1.0 / (metric.lapse * metric.lapse);
  metric.upper[0][0] = -inverse_lapse2;
  for (int axis = 0; axis < 3; ++axis) {
    metric.upper[0][axis + 1] = metric.shift[axis] * inverse_lapse2;
    metric.upper[axis + 1][0] = metric.upper[0][axis + 1];
    for (int second = 0; second < 3; ++second)
      metric.upper[axis + 1][second + 1] =
          inverse[axis][second] - metric.shift[axis] * metric.shift[second] * inverse_lapse2;
  }
  metric.gdet = metric.lapse * sqrt(det);
  return metric;
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  using Real = parthenon::Real;
  const auto metric = BuildMetric();
  constexpr Real gamma = 5.0 / 3.0;
  const pangu::eos::RelativisticEOS eos{{gamma}, 1.0e-12, 1.0e-14};
  const pangu::relativity::MHDPrimitiveState primitive{
      {1.3, {0.21, -0.13, 0.075}, 0.17}, {0.31, -0.22, 0.09}};
  Kokkos::View<Real[24]> result("SYNC-2 GRMHD state results");
  Kokkos::parallel_for(
      "SYNC-2 dynamic GRMHD state contract", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        const auto conserved = pangu::nr::ConvertSyncGRMHDP2C(primitive, eos, metric);
        const Real volume = sqrt(metric.spatial_det);
        const Real densitized_magnetic[3]{volume * primitive.magnetic[0],
                                          volume * primitive.magnetic[1],
                                          volume * primitive.magnetic[2]};
        const auto recovered = pangu::nr::SolveSyncGRMHDC2P(conserved, densitized_magnetic, eos,
                                                            50.0, 1.0e12, metric);
        const auto rebuilt =
            pangu::nr::ConvertSyncGRMHDP2C(recovered.primitive, eos, metric);
        result(0) = fabs(recovered.primitive.fluid.density - primitive.fluid.density);
        result(1) = fabs(recovered.primitive.fluid.pressure - primitive.fluid.pressure);
        for (int axis = 0; axis < 3; ++axis)
          result(2 + axis) =
              fabs(recovered.primitive.fluid.u[axis] - primitive.fluid.u[axis]);
        result(5) = fabs(rebuilt.density - conserved.density);
        result(6) = fabs(rebuilt.energy - conserved.energy);
        for (int axis = 0; axis < 3; ++axis)
          result(7 + axis) = fabs(rebuilt.momentum[axis] - conserved.momentum[axis]);
        result(10) = recovered.success ? 0.0 : 1.0;

        auto hydro_primitive = primitive.fluid;
        auto zero_magnetic = primitive;
        for (int axis = 0; axis < 3; ++axis)
          zero_magnetic.magnetic[axis] = 0.0;
        const auto hydro_conserved =
            pangu::nr::ConvertSyncGRHDP2C(hydro_primitive, eos, metric);
        const auto zero_conserved =
            pangu::nr::ConvertSyncGRMHDP2C(zero_magnetic, eos, metric);
        result(11) = fabs(hydro_conserved.density - zero_conserved.density);
        result(12) = fabs(hydro_conserved.energy - zero_conserved.energy);
        Real momentum_error = 0.0;
        for (int axis = 0; axis < 3; ++axis)
          momentum_error = fmax(
              momentum_error,
              fabs(hydro_conserved.momentum[axis] - zero_conserved.momentum[axis]));
        result(13) = momentum_error;
        const auto hydro_matter =
            pangu::nr::BuildSyncGRHDStressEnergy(hydro_primitive, hydro_conserved, metric);
        const auto zero_matter =
            pangu::nr::BuildSyncGRMHDStressEnergy(zero_magnetic, zero_conserved, metric);
        Real matter_error = fabs(hydro_matter.energy - zero_matter.energy);
        for (int axis = 0; axis < 3; ++axis)
          matter_error = fmax(
              matter_error,
              fabs(hydro_matter.momentum[axis] - zero_matter.momentum[axis]));
        for (int component = 0; component < 6; ++component)
          matter_error = fmax(
              matter_error,
              fabs(hydro_matter.stress[component] - zero_matter.stress[component]));
        result(14) = matter_error;

        const auto face = pangu::nr::BuildSyncGRMHDFluxState(primitive, eos, 0, metric);
        using pangu::riemann::Implementation;
        using pangu::riemann::Physics;
        using pangu::riemann::Solver;
        const auto hlle =
            Implementation<Physics::relativistic_mhd, Solver::hlle>::Solve(face, face, metric);
        const auto llf =
            Implementation<Physics::relativistic_mhd, Solver::llf>::Solve(face, face, metric);
        Real equal_state_error = 0.0;
        for (int component = 0; component < 5; ++component) {
          const Real expected = volume * metric.lapse * face.flux[component];
          equal_state_error = fmax(equal_state_error, fabs(hlle.flux[component] - expected));
          equal_state_error = fmax(equal_state_error, fabs(llf.flux[component] - expected));
        }
        for (int axis = 0; axis < 3; ++axis) {
          const Real expected = volume * metric.lapse * face.induction[axis];
          equal_state_error =
              fmax(equal_state_error, fabs(hlle.induction[axis] - expected));
          equal_state_error = fmax(equal_state_error, fabs(llf.induction[axis] - expected));
        }
        result(15) = equal_state_error;
        result(16) = face.lambda_minus;
        result(17) = face.lambda_plus;
        const auto matter =
            pangu::nr::BuildSyncGRMHDStressEnergy(primitive, conserved, metric);
        result(18) = matter.energy;
        result(19) = matter.stress[pangu::nr::SpatialSymmetricComponent(0, 0)];
        result(20) = matter.stress[pangu::nr::SpatialSymmetricComponent(0, 1)];
        result(21) = conserved.density;
        result(22) = conserved.energy;
        result(23) = metric.spatial_det;
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  Real roundtrip = 0.0;
  for (int component = 0; component < 10; ++component)
    roundtrip = std::max(roundtrip, host(component));
  const bool finite = std::isfinite(host(18)) && std::isfinite(host(19)) &&
                      std::isfinite(host(20)) && host(18) > 0.0 && host(23) > 0.0;
  const bool valid = roundtrip < 3.0e-11 && host(10) == 0.0 && host(11) < 2.0e-15 &&
                     host(12) < 2.0e-15 && host(13) < 2.0e-15 && host(14) < 3.0e-15 &&
                     host(15) < 3.0e-15 && host(16) < host(17) && finite;
  std::cout.precision(17);
  std::cout << "SYNC-2 GRMHD state: roundtrip_linf=" << roundtrip
            << " zero_B_hydro_linf=" << std::max({host(11), host(12), host(13), host(14)})
            << " equal_state_riemann_linf=" << host(15)
            << " lambda=[" << host(16) << ',' << host(17) << "]"
            << " D=" << host(21) << " tau=" << host(22) << '\n';
  if (!valid) {
    std::cout << "  primitive errors: rho=" << host(0) << " p=" << host(1)
              << " u1=" << host(2) << " u2=" << host(3) << " u3=" << host(4)
              << "\n  conserved rebuild errors: D=" << host(5)
              << " tau=" << host(6) << " S1=" << host(7) << " S2=" << host(8)
              << " S3=" << host(9) << '\n';
  }
  if (!valid) {
    std::cerr << "SYNC-2 dynamic GRMHD state validation failed\n";
    return 1;
  }
  return 0;
}
