#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "geometry/metric/mks.h"
#include "geometry/mode/dynamic.h"
#include "geometry/mode/static.h"

namespace {

using MKS = pangu::geometry::metric::MKS;
using DynamicMKS = pangu::geometry::Geometry<MKS, pangu::geometry::DynamicMode>;
using StaticMKS = pangu::geometry::Geometry<MKS, pangu::geometry::StaticMode>;

struct MKSTestCoordinates {
  template <int Axis> KOKKOS_INLINE_FUNCTION double Xc(const int index) const {
    if constexpr (Axis == 1)
      return -0.35 + 0.18 * index;
    if constexpr (Axis == 2)
      return 0.10 + 0.15 * index;
    return 0.20 + 0.45 * index;
  }

  template <int Axis> KOKKOS_INLINE_FUNCTION double Xf(const int index) const {
    constexpr double spacing = Axis == 1 ? 0.18 : (Axis == 2 ? 0.15 : 0.45);
    return Xc<Axis>(index) - 0.5 * spacing;
  }
};

void Require(const bool condition, const char* message) {
  if (!condition) {
    std::cerr << "MKS geometry failure: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

KOKKOS_INLINE_FUNCTION double RelativeDifference(const double left, const double right) {
  return fabs(left - right) / fmax(1.0, fmax(fabs(left), fabs(right)));
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    static_assert(MKS::symmetry == pangu::geometry::Symmetry::axisymmetric_2d);
    static_assert(!MKS::supports_excision);

    const MKS::Parameters parameters{0.8, 0.3};
    const DynamicMKS dynamic_geometry(parameters);
    StaticMKS static_geometry(parameters);
    const parthenon::IndexRange compact_bounds{0, 3};
    const MKSTestCoordinates coordinates{};
    static_geometry.Allocate(1, compact_bounds, compact_bounds, compact_bounds);
    static_geometry.InitializeBlock(0, coordinates);
    Kokkos::fence("MKS static Geometry initialization");

    StaticMKS extended_azimuth_geometry(parameters);
    extended_azimuth_geometry.Allocate(1, parthenon::IndexRange{0, 31}, compact_bounds,
                                       compact_bounds);
    Require(static_geometry.StorageBytes() == extended_azimuth_geometry.StorageBytes(),
            "axisymmetric storage depends on the X3 extent");
    Require(static_geometry.StorageBytes() == 2420 * sizeof(parthenon::Real),
            "axisymmetric storage does not have the expected reduced shape");

    double storage_error = 0.0;
    Kokkos::parallel_reduce(
        "MKS dynamic/static point comparison",
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<4>>(
            {0, compact_bounds.s, compact_bounds.s, compact_bounds.s},
            {4, compact_bounds.e + 1, compact_bounds.e + 1, compact_bounds.e + 1}),
        KOKKOS_LAMBDA(const int location_index, const int k, const int j, const int i,
                      double& maximum) {
          const auto location = static_cast<pangu::geometry::Location>(location_index);
          const auto online = dynamic_geometry.MetricAt(location, 0, k, j, i, coordinates);
          const auto stored = static_geometry.MetricAt(location, 0, k, j, i, coordinates);
          for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
              maximum =
                  fmax(maximum, RelativeDifference(online.lower[mu][nu], stored.lower[mu][nu]));
              maximum =
                  fmax(maximum, RelativeDifference(online.upper[mu][nu], stored.upper[mu][nu]));
            }
          }
          maximum = fmax(maximum, RelativeDifference(online.gdet, stored.gdet));
          maximum = fmax(maximum, RelativeDifference(online.lapse, stored.lapse));
          maximum = fmax(maximum, RelativeDifference(online.spatial_det, stored.spatial_det));
          for (int axis = 0; axis < 3; ++axis)
            maximum = fmax(maximum, RelativeDifference(online.shift[axis], stored.shift[axis]));
          if (location == pangu::geometry::Location::cell_center) {
            const auto online_derivatives = dynamic_geometry.DerivativesAt(0, k, j, i, coordinates);
            const auto stored_derivatives = static_geometry.DerivativesAt(0, k, j, i, coordinates);
            for (int axis = 0; axis < 3; ++axis)
              for (int mu = 0; mu < 4; ++mu)
                for (int nu = 0; nu < 4; ++nu)
                  maximum =
                      fmax(maximum, RelativeDifference(online_derivatives.lower[axis][mu][nu],
                                                       stored_derivatives.lower[axis][mu][nu]));
          }
        },
        Kokkos::Max<double>(storage_error));
    std::cout << "MKS dynamic/static maximum error: " << storage_error << '\n';
    Require(storage_error <= 16.0 * std::numeric_limits<double>::epsilon(),
            "dynamic/static values exceed FP64 roundoff");

    double invariant_error = 0.0;
    Kokkos::parallel_reduce(
        "MKS analytic geometry identities",
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>({0, 0, 0}, {4, 4, 4}),
        KOKKOS_LAMBDA(const int k, const int j, const int i, double& maximum) {
          const double x1 = -0.45 + 0.31 * i;
          const double x2 = 0.08 + 0.21 * j;
          const double x3 = 0.17 + 0.73 * k;
          const auto metric =
              MKS::Calculate(x1, x2, x3, pangu::geometry::Location::cell_center, parameters);
          const auto shifted_phi = MKS::Calculate(
              x1, x2, x3 + 1.2345, pangu::geometry::Location::cell_center, parameters);
          for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
              double product = 0.0;
              for (int sigma = 0; sigma < 4; ++sigma)
                product += metric.lower[mu][sigma] * metric.upper[sigma][nu];
              maximum = fmax(maximum, fabs(product - (mu == nu ? 1.0 : 0.0)));
              maximum = fmax(maximum, fabs(metric.lower[mu][nu] - shifted_phi.lower[mu][nu]));
              maximum = fmax(maximum, fabs(metric.upper[mu][nu] - shifted_phi.upper[mu][nu]));
            }
          }
          const double r = exp(x1);
          const double theta = MKS::PolarAngle(x2, parameters);
          const double jacobian = MKS::PolarJacobian(x2, parameters);
          const double rho2 = r * r + parameters.spin * parameters.spin * cos(theta) * cos(theta);
          const double expected_gdet = fabs(rho2 * sin(theta) * r * jacobian);
          maximum = fmax(maximum, RelativeDifference(metric.gdet, expected_gdet));

          const double native[4]{0.4, x1, x2, x3};
          double embedding[4]{};
          double recovered[4]{};
          MKS::NativeToEmbedding(native, embedding, parameters);
          MKS::EmbeddingToNative(embedding, recovered, parameters);
          for (int axis = 0; axis < 4; ++axis)
            maximum = fmax(maximum, fabs(native[axis] - recovered[axis]));

          constexpr double step = 1.0e-5;
          const auto derivatives = MKS::CalculateDerivatives(x1, x2, x3, parameters);
          const auto radial_plus =
              MKS::Calculate(x1 + step, x2, x3, pangu::geometry::Location::cell_center, parameters);
          const auto radial_minus =
              MKS::Calculate(x1 - step, x2, x3, pangu::geometry::Location::cell_center, parameters);
          const auto polar_plus =
              MKS::Calculate(x1, x2 + step, x3, pangu::geometry::Location::cell_center, parameters);
          const auto polar_minus =
              MKS::Calculate(x1, x2 - step, x3, pangu::geometry::Location::cell_center, parameters);
          for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
              const double radial_difference =
                  (radial_plus.lower[mu][nu] - radial_minus.lower[mu][nu]) / (2.0 * step);
              const double polar_difference =
                  (polar_plus.lower[mu][nu] - polar_minus.lower[mu][nu]) / (2.0 * step);
              maximum = fmax(maximum,
                             RelativeDifference(derivatives.lower[0][mu][nu], radial_difference));
              maximum =
                  fmax(maximum, RelativeDifference(derivatives.lower[1][mu][nu], polar_difference));
              maximum = fmax(maximum, fabs(derivatives.lower[2][mu][nu]));
            }
          }
        },
        Kokkos::Max<double>(invariant_error));
    std::cout << "MKS identity/derivative maximum error: " << invariant_error << '\n';
    Require(invariant_error <= 2.0e-8,
            "MKS inverse, determinant, symmetry, round-trip, or derivative check failed");

    constexpr double reference_lower[4][4]{
        {-0.50046572018581185, 1.9981371192567526, 0.0, -0.39031302013511362},
        {1.9981371192567526, 23.992548477027010, 0.0, -4.6866673796090062},
        {0.0, 0.0, 42.874076872093823, 0.0},
        {-0.39031302013511362, -4.6866673796090062, 0.0, 16.557132126295453}};
    constexpr double reference_upper[4][4]{
        {-1.4995342798141882, 0.12488356995354704, 0.0, 0.0},
        {0.12488356995354704, 0.033718563887457700, 0.0, 0.012488356995354704},
        {0.0, 0.0, 0.023324117344457321, 0.0},
        {0.0, 0.012488356995354704, 0.0, 0.063931891542613397}};
    constexpr double reference_gdet = 103.58541384075336;
    double reference_error = 0.0;
    Kokkos::parallel_reduce(
        "MKS KHARMA convention reference", 1,
        KOKKOS_LAMBDA(const int, double& maximum) {
          const auto metric = MKS::Calculate(log(4.0), 0.37, 2.1,
                                             pangu::geometry::Location::cell_center, parameters);
          for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
              maximum =
                  fmax(maximum, RelativeDifference(metric.lower[mu][nu], reference_lower[mu][nu]));
              maximum =
                  fmax(maximum, RelativeDifference(metric.upper[mu][nu], reference_upper[mu][nu]));
            }
          }
          maximum = fmax(maximum, RelativeDifference(metric.gdet, reference_gdet));
        },
        Kokkos::Max<double>(reference_error));
    std::cout << "MKS KHARMA-reference maximum error: " << reference_error << '\n';
    Require(reference_error <= 4.0e-15,
            "MKS metric differs from the independent KHARMA-convention reference");
  }
  Kokkos::finalize();
  std::cout << "PANGU MKS geometry: PASS\n";
  return EXIT_SUCCESS;
}
