#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "geometry/geometry.h"
#include "geometry/metric/cks.h"
#include "geometry/metric/mks.h"
#include "geometry/mode/dynamic.h"
#include "geometry/mode/static.h"
#include "relativity/relativistic_hydro.h"
#include "relativity/relativistic_mhd.h"

namespace {

struct NoExcisionMetric {
  struct Parameters {};
  static constexpr pangu::geometry::Symmetry symmetry = pangu::geometry::Symmetry::axisymmetric_2d;
  static constexpr bool supports_excision = false;
};

struct NoCoordinates {};

struct TestCoordinates {
  template <int Axis> KOKKOS_INLINE_FUNCTION double Xc(const int index) const {
    if constexpr (Axis == 1)
      return 2.0 + 0.5 * index;
    if constexpr (Axis == 2)
      return -1.0 + 0.4 * index;
    return 0.7 + 0.3 * index;
  }

  template <int Axis> KOKKOS_INLINE_FUNCTION double Xf(const int index) const {
    constexpr double spacing = Axis == 1 ? 0.5 : (Axis == 2 ? 0.4 : 0.3);
    return Xc<Axis>(index) - 0.5 * spacing;
  }
};

void Require(const bool condition, const char* message) {
  if (!condition) {
    std::cerr << "phase5 relativity failure: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

bool Near(const double left, const double right, const double tolerance = 2.0e-9) {
  return std::abs(left - right) <= tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const pangu::geometry::Geometry<NoExcisionMetric, pangu::geometry::DynamicMode>
        no_excision_dynamic(NoExcisionMetric::Parameters{});
    pangu::geometry::Geometry<NoExcisionMetric, pangu::geometry::StaticMode> no_excision_static(
        NoExcisionMetric::Parameters{});
    const auto no_excision_parameters =
        pangu::geometry::ConfigureExcision<NoExcisionMetric>(nullptr);
    Require(!no_excision_parameters.enabled,
            "a metric without excision capability acquired excision parameters");
    Require(!no_excision_dynamic.IsExcised(0.0, 0.0, 0.0, 1.0),
            "dynamic Geometry imposed excision on a metric without that capability");
    Require(!no_excision_static.IsExcised(0.0, 0.0, 0.0, 1.0),
            "static Geometry imposed excision on a metric without that capability");
    Require(!no_excision_dynamic.NeedsFluxExcision(NoCoordinates{}, 0, 0, 0, 1.0),
            "flux excision sampled coordinates for a metric without that capability");
    const parthenon::IndexRange compact_bounds{0, 3};
    no_excision_static.Allocate(1, compact_bounds, compact_bounds, compact_bounds);
    pangu::geometry::Geometry<NoExcisionMetric, pangu::geometry::StaticMode>
        extended_azimuth_geometry(NoExcisionMetric::Parameters{});
    extended_azimuth_geometry.Allocate(1, parthenon::IndexRange{0, 31}, compact_bounds,
                                       compact_bounds);
    Require(no_excision_static.StorageBytes() == extended_azimuth_geometry.StorageBytes(),
            "axisymmetric static Geometry storage depends on azimuthal extent");

    using CKS = pangu::geometry::metric::CKS;
    using DynamicCKS = pangu::geometry::Geometry<CKS, pangu::geometry::DynamicMode>;
    using StaticCKS = pangu::geometry::Geometry<CKS, pangu::geometry::StaticMode>;
    const CKS::Parameters cks_parameters{0.5, false};
    const parthenon::IndexRange geometry_bounds{0, 3};
    const TestCoordinates test_coordinates{};
    const DynamicCKS dynamic_cks(cks_parameters);
    StaticCKS static_cks(cks_parameters);
    static_cks.Allocate(1, geometry_bounds, geometry_bounds, geometry_bounds);
    static_cks.InitializeBlock(0, test_coordinates);
    Kokkos::fence("phase-5 static Geometry test initialization");
    double metric_difference = 0.0;
    Kokkos::parallel_reduce(
        "phase-5 dynamic/static Geometry comparison",
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<4>>(
            {0, geometry_bounds.s, geometry_bounds.s, geometry_bounds.s},
            {4, geometry_bounds.e + 1, geometry_bounds.e + 1, geometry_bounds.e + 1}),
        KOKKOS_LAMBDA(const int location_index, const int k, const int j, const int i,
                      double& maximum) {
          const auto location = static_cast<pangu::geometry::Location>(location_index);
          const auto online = dynamic_cks.MetricAt(location, 0, k, j, i, test_coordinates);
          const auto stored = static_cks.MetricAt(location, 0, k, j, i, test_coordinates);
          for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
              maximum = fmax(maximum, fabs(online.lower[mu][nu] - stored.lower[mu][nu]));
              maximum = fmax(maximum, fabs(online.upper[mu][nu] - stored.upper[mu][nu]));
            }
          }
          maximum = fmax(maximum, fabs(online.gdet - stored.gdet));
          maximum = fmax(maximum, fabs(online.lapse - stored.lapse));
          maximum = fmax(maximum, fabs(online.spatial_det - stored.spatial_det));
          for (int axis = 0; axis < 3; ++axis)
            maximum = fmax(maximum, fabs(online.shift[axis] - stored.shift[axis]));
          if (location == pangu::geometry::Location::cell_center) {
            const auto online_derivatives = dynamic_cks.DerivativesAt(0, k, j, i, test_coordinates);
            const auto stored_derivatives = static_cks.DerivativesAt(0, k, j, i, test_coordinates);
            for (int axis = 0; axis < 3; ++axis)
              for (int mu = 0; mu < 4; ++mu)
                for (int nu = 0; nu < 4; ++nu)
                  maximum = fmax(maximum, fabs(online_derivatives.lower[axis][mu][nu] -
                                               stored_derivatives.lower[axis][mu][nu]));
          }
        },
        Kokkos::Max<double>(metric_difference));
    Require(metric_difference <= 8.0 * std::numeric_limits<double>::epsilon(),
            "dynamic/static Geometry values exceed FP64 roundoff");

    const auto metric =
        pangu::geometry::Geometry<pangu::geometry::metric::CKS, pangu::geometry::DynamicMode>(
            pangu::geometry::metric::CKS::Parameters{0.7, false})
            .MetricAt(pangu::geometry::Location::cell_center, 4.0, -1.0, 0.5);
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        double product = 0.0;
        for (int sigma = 0; sigma < 4; ++sigma)
          product += metric.lower[mu][sigma] * metric.upper[sigma][nu];
        Require(Near(product, mu == nu ? 1.0 : 0.0, 2.0e-12),
                "Kerr-Schild metric inverse mismatch");
      }
    }

    constexpr double gamma = 4.0 / 3.0;
    const pangu::eos::RelativisticEOS eos{{gamma}, 1.0e-12, 1.0e-12};
    const pangu::relativity::HydroPrimitiveState primitive{1.3, {0.4, -0.2, 0.1}, 0.7};
    const auto conserved = pangu::relativity::ConvertSRHDP2C(primitive, eos);
    const auto recovered = pangu::relativity::SolveSRHDC2P(conserved, eos, 1000.0);
    Require(recovered.success, "SR Hydro C2P did not converge");
    Require(Near(recovered.primitive.density, primitive.density), "SR Hydro density round trip");
    Require(Near(recovered.primitive.pressure, primitive.pressure), "SR Hydro pressure round trip");
    for (int axis = 0; axis < 3; ++axis)
      Require(Near(recovered.primitive.u[axis], primitive.u[axis]), "SR Hydro velocity round trip");

    const pangu::relativity::MHDPrimitiveState mhd{{1.1, {0.25, -0.12, 0.08}, 0.4},
                                                   {0.3, -0.15, 0.05}};
    const auto mhd_conserved = pangu::relativity::ConvertSRMHDP2C(mhd, eos);
    const auto mhd_recovered =
        pangu::relativity::SolveSRMHDC2P(mhd_conserved, mhd.magnetic, eos, 1000.0, 1.0e6);
    Require(mhd_recovered.success, "SRMHD C2P did not converge");
    Require(Near(mhd_recovered.primitive.fluid.density, mhd.fluid.density, 2.0e-8),
            "SRMHD density round trip");
    Require(Near(mhd_recovered.primitive.fluid.pressure, mhd.fluid.pressure, 2.0e-8),
            "SRMHD pressure round trip");
    for (int axis = 0; axis < 3; ++axis)
      Require(Near(mhd_recovered.primitive.fluid.u[axis], mhd.fluid.u[axis], 2.0e-8),
              "SRMHD velocity round trip");

    const pangu::relativity::HydroPrimitiveState gr_primitive{0.9, {0.18, -0.07, 0.04}, 0.23};
    const auto gr_conserved = pangu::relativity::ConvertGRHDP2C(gr_primitive, eos, metric);
    const auto gr_recovered = pangu::relativity::SolveGRHDC2P(gr_conserved, eos, 1000.0, metric);
    Require(gr_recovered.success, "GR Hydro C2P did not converge");
    Require(Near(gr_recovered.primitive.density, gr_primitive.density, 2.0e-8),
            "GR Hydro density round trip");
    Require(Near(gr_recovered.primitive.pressure, gr_primitive.pressure, 2.0e-8),
            "GR Hydro pressure round trip");
    for (int axis = 0; axis < 3; ++axis)
      Require(Near(gr_recovered.primitive.u[axis], gr_primitive.u[axis], 2.0e-8),
              "GR Hydro velocity round trip");

    const pangu::relativity::MHDPrimitiveState gr_mhd{{1.2, {0.13, -0.09, 0.03}, 0.31},
                                                      {0.22, -0.11, 0.08}};
    const auto gr_mhd_conserved = pangu::relativity::ConvertGRMHDP2C(gr_mhd, eos, metric);
    const auto gr_mhd_recovered = pangu::relativity::SolveGRMHDC2P(
        gr_mhd_conserved, gr_mhd.magnetic, eos, 1000.0, 1.0e6, metric);
    Require(gr_mhd_recovered.success, "GRMHD C2P did not converge");
    Require(Near(gr_mhd_recovered.primitive.fluid.density, gr_mhd.fluid.density, 2.0e-8),
            "GRMHD density round trip");
    Require(Near(gr_mhd_recovered.primitive.fluid.pressure, gr_mhd.fluid.pressure, 2.0e-8),
            "GRMHD pressure round trip");
    for (int axis = 0; axis < 3; ++axis) {
      Require(Near(gr_mhd_recovered.primitive.fluid.u[axis], gr_mhd.fluid.u[axis], 2.0e-8),
              "GRMHD velocity round trip");
      Require(Near(gr_mhd_recovered.primitive.magnetic[axis], gr_mhd.magnetic[axis], 2.0e-12),
              "GRMHD magnetic round trip");
    }

    const pangu::geometry::metric::MKS::Parameters mks_parameters{0.8, 0.3};
    const auto mks_metric = pangu::geometry::metric::MKS::Calculate(
        std::log(4.0), 0.37, 2.1, pangu::geometry::Location::cell_center, mks_parameters);
    Require(std::abs(mks_metric.gdet - 1.0) > 1.0,
            "MKS densitization test point has a unit determinant");
    auto unit_determinant_metric = mks_metric;
    unit_determinant_metric.gdet = 1.0;

    const pangu::relativity::HydroPrimitiveState mks_primitive{0.73, {0.031, -0.004, 0.017}, 0.19};
    const auto mks_conserved = pangu::relativity::ConvertGRHDP2C(mks_primitive, eos, mks_metric);
    const auto undensitized_conserved =
        pangu::relativity::ConvertGRHDP2C(mks_primitive, eos, unit_determinant_metric);
    Require(Near(mks_conserved.density, mks_metric.gdet * undensitized_conserved.density, 2.0e-12),
            "MKS GR Hydro density is not densitized exactly once");
    Require(Near(mks_conserved.energy, mks_metric.gdet * undensitized_conserved.energy, 2.0e-12),
            "MKS GR Hydro energy is not densitized exactly once");
    for (int axis = 0; axis < 3; ++axis)
      Require(Near(mks_conserved.momentum[axis],
                   mks_metric.gdet * undensitized_conserved.momentum[axis], 2.0e-12),
              "MKS GR Hydro momentum is not densitized exactly once");
    const auto mks_recovered =
        pangu::relativity::SolveGRHDC2P(mks_conserved, eos, 1000.0, mks_metric);
    Require(mks_recovered.success, "MKS GR Hydro C2P did not converge");
    Require(Near(mks_recovered.primitive.density, mks_primitive.density, 2.0e-8),
            "MKS GR Hydro density round trip");
    Require(Near(mks_recovered.primitive.pressure, mks_primitive.pressure, 2.0e-8),
            "MKS GR Hydro pressure round trip");
    for (int axis = 0; axis < 3; ++axis)
      Require(Near(mks_recovered.primitive.u[axis], mks_primitive.u[axis], 2.0e-8),
              "MKS GR Hydro velocity round trip");

    const pangu::relativity::MHDPrimitiveState mks_mhd{{0.82, {0.023, -0.003, 0.014}, 0.21},
                                                       {0.017, -0.006, 0.009}};
    const auto mks_mhd_conserved = pangu::relativity::ConvertGRMHDP2C(mks_mhd, eos, mks_metric);
    const auto mks_mhd_recovered = pangu::relativity::SolveGRMHDC2P(
        mks_mhd_conserved, mks_mhd.magnetic, eos, 1000.0, 1.0e6, mks_metric);
    Require(mks_mhd_recovered.success, "MKS GRMHD C2P did not converge");
    Require(Near(mks_mhd_recovered.primitive.fluid.density, mks_mhd.fluid.density, 2.0e-8),
            "MKS GRMHD density round trip");
    Require(Near(mks_mhd_recovered.primitive.fluid.pressure, mks_mhd.fluid.pressure, 2.0e-8),
            "MKS GRMHD pressure round trip");
    for (int axis = 0; axis < 3; ++axis) {
      Require(Near(mks_mhd_recovered.primitive.fluid.u[axis], mks_mhd.fluid.u[axis], 2.0e-8),
              "MKS GRMHD velocity round trip");
      Require(Near(mks_mhd_recovered.primitive.magnetic[axis], mks_mhd.magnetic[axis], 2.0e-12),
              "MKS GRMHD magnetic round trip");
    }

    // Magnetised Bondi exercises the inversion in the cancellation-dominated
    // sigma~10^3 regime used by the MKS regression.  This state is sampled at
    // the innermost cell of the N=64 initial condition.
    constexpr double bondi_gamma = 5.0 / 3.0;
    const auto bondi_metric = pangu::geometry::metric::MKS::Calculate(
        0.6065987399697118, 0.5, 0.0, pangu::geometry::Location::cell_center,
        pangu::geometry::metric::MKS::Parameters{0.0, 0.3});
    const pangu::relativity::MHDPrimitiveState bondi_mhd{
        {0.07369433, {-0.00804170, 0.0, 0.0},
         (bondi_gamma - 1.0) * 0.01943039},
        {4.92404364, 0.0, 0.0}};
    const pangu::eos::RelativisticEOS bondi_eos{{bondi_gamma}, 1.0e-38, 1.0e-38};
    const auto bondi_cons = pangu::relativity::ConvertGRMHDP2C(bondi_mhd, bondi_eos, bondi_metric);
    const auto bondi_recovered = pangu::relativity::SolveGRMHDC2P(
        bondi_cons, bondi_mhd.magnetic, bondi_eos, std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(), bondi_metric);
    Require(bondi_recovered.success, "sigma=1e3 MKS Bondi C2P did not converge");
    Require(Near(bondi_recovered.primitive.fluid.density, bondi_mhd.fluid.density, 5.0e-12),
            "sigma=1e3 MKS Bondi density round trip");
    Require(Near(bondi_recovered.primitive.fluid.pressure, bondi_mhd.fluid.pressure, 5.0e-12),
            "sigma=1e3 MKS Bondi pressure round trip");
    Require(Near(bondi_recovered.primitive.fluid.u[0], bondi_mhd.fluid.u[0], 5.0e-12),
            "sigma=1e3 MKS Bondi velocity round trip");

    for (int direction = 0; direction < 3; ++direction) {
      double plus = 0.0;
      double minus = 0.0;
      pangu::relativity::ComputeCoordinateLightSpeeds(mks_metric, direction, plus, minus);
      const int spatial = direction + 1;
      const auto null_residual = [&](const double speed) {
        return mks_metric.lower[0][0] + 2.0 * mks_metric.lower[0][spatial] * speed +
               mks_metric.lower[spatial][spatial] * speed * speed;
      };
      Require(std::isfinite(plus) && std::isfinite(minus) && plus >= minus,
              "MKS coordinate light-speed ordering is invalid");
      Require(std::abs(null_residual(plus)) <= 2.0e-11 && std::abs(null_residual(minus)) <= 2.0e-11,
              "MKS coordinate light speeds do not satisfy the null condition");
    }

    const pangu::relativity::HydroPrimitiveState extreme{1.0e-8, {1.0e5, 0.0, 0.0}, 1.0e-10};
    const pangu::eos::RelativisticEOS floored_eos{{gamma}, 1.0e-10, 1.0e-12};
    const auto extreme_cons = pangu::relativity::ConvertSRHDP2C(extreme, floored_eos);
    const auto limited = pangu::relativity::SolveSRHDC2P(extreme_cons, floored_eos, 20.0);
    const double lorentz = std::sqrt(1.0 + limited.primitive.u[0] * limited.primitive.u[0] +
                                     limited.primitive.u[1] * limited.primitive.u[1] +
                                     limited.primitive.u[2] * limited.primitive.u[2]);
    Require(std::isfinite(lorentz) && lorentz <= 20.0 * (1.0 + 1.0e-12),
            "Lorentz ceiling is not controlled");

    const pangu::relativity::MHDPrimitiveState magnetized{{1.0e-8, {0.0, 0.0, 0.0}, 1.0e-10},
                                                          {100.0, 0.0, 0.0}};
    const auto magnetized_cons = pangu::relativity::ConvertSRMHDP2C(magnetized, floored_eos);
    const auto sigma_limited = pangu::relativity::SolveSRMHDC2P(
        magnetized_cons, magnetized.magnetic, floored_eos, 1000.0, 10.0);
    const double sigma =
        magnetized.magnetic[0] * magnetized.magnetic[0] / sigma_limited.primitive.fluid.density;
    Require(sigma_limited.success && sigma_limited.sigma_ceiling,
            "magnetization ceiling did not report a controlled repair");
    Require(std::isfinite(sigma_limited.primitive.fluid.density) &&
                std::isfinite(sigma_limited.primitive.fluid.pressure) &&
                sigma <= 10.0 * (1.0 + 1.0e-12),
            "magnetization ceiling produced an invalid state");
  }
  Kokkos::finalize();
  std::cout << "PANGU phase-5 relativity kernels: PASS\n";
  return EXIT_SUCCESS;
}
