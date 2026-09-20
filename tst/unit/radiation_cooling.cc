#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "geometry/metric/cks.h"
#include "geometry/metric/mks.h"
#include "radiation/cooling_model.h"
#include "radiation/cooling_source.h"

namespace {

using Real = parthenon::Real;
using pangu::geometry::Location;
using pangu::geometry::MetricPoint;
using pangu::radiation::BuildCoolingIncrement;
using pangu::radiation::BuildFourVelocity;
using pangu::radiation::CoolingRamp;
using pangu::radiation::EvaluateTargetThicknessCooling;

KOKKOS_INLINE_FUNCTION MetricPoint MinkowskiMetric() {
  MetricPoint metric{};
  for (int axis = 0; axis < 4; ++axis) {
    metric.lower[axis][axis] = axis == 0 ? -1.0 : 1.0;
    metric.upper[axis][axis] = axis == 0 ? -1.0 : 1.0;
  }
  return metric;
}

KOKKOS_INLINE_FUNCTION Real RelativeDifference(const Real actual, const Real expected) {
  return fabs(actual - expected) / fmax(1.0, fmax(fabs(actual), fabs(expected)));
}

void Require(const bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  try {
    constexpr int checks = 26;
    Kokkos::View<Real[checks]> errors("radiation cooling errors");
    Kokkos::parallel_for(
        "PANGU RC-2 covariant cooling vectors", 1, KOKKOS_LAMBDA(const int) {
          int check = 0;
          pangu::relativity::HydroPrimitiveState rest{1.0, {0.0, 0.0, 0.0}, 0.2};
          const auto flat = MinkowskiMetric();
          const auto rest_increment = BuildCoolingIncrement(rest, flat, 0.1);
          errors(check++) = RelativeDifference(rest_increment.energy, 0.1);
          for (int axis = 0; axis < 3; ++axis)
            errors(check++) = fabs(rest_increment.momentum[axis]);

          pangu::relativity::HydroPrimitiveState boosted{0.7, {0.75, -0.2, 0.1}, 0.08};
          const auto boosted_velocity = BuildFourVelocity(boosted, flat);
          const auto boosted_increment = BuildCoolingIncrement(boosted, flat, 0.03);
          const Real normalization = -0.03 * boosted_velocity.upper[0];
          errors(check++) = RelativeDifference(boosted_increment.energy,
                                               normalization * boosted_velocity.lower[0]);
          for (int axis = 0; axis < 3; ++axis)
            errors(check++) = RelativeDifference(boosted_increment.momentum[axis],
                                                 normalization * boosted_velocity.lower[axis + 1]);

          const pangu::geometry::metric::CKS::Parameters cks_parameters{0.7, false};
          errors(check++) =
              RelativeDifference(1.0 + sqrt(1.0 - 0.9375 * 0.9375), 1.3479852726768764);
          const auto cks = pangu::geometry::metric::CKS::Calculate(
              4.0, -1.2, 0.8, Location::cell_center, cks_parameters);
          const auto cks_velocity = BuildFourVelocity(boosted, cks);
          const auto cks_increment = BuildCoolingIncrement(boosted, cks, 0.03);
          errors(check++) = RelativeDifference(cks_increment.energy / cks_velocity.lower[0],
                                               -cks.gdet * 0.03 * cks_velocity.upper[0]);
          errors(check++) = RelativeDifference(cks_increment.momentum[1] / cks_velocity.lower[2],
                                               -cks.gdet * 0.03 * cks_velocity.upper[0]);

          const pangu::geometry::metric::MKS::Parameters mks_parameters{0.7, 0.3};
          const auto mks = pangu::geometry::metric::MKS::Calculate(
              log(4.0), 0.37, 2.1, Location::cell_center, mks_parameters);
          const auto mks_velocity = BuildFourVelocity(boosted, mks);
          const auto mks_increment = BuildCoolingIncrement(boosted, mks, 0.03);
          errors(check++) = RelativeDifference(mks_increment.energy / mks_velocity.lower[0],
                                               -mks.gdet * 0.03 * mks_velocity.upper[0]);
          errors(check++) = RelativeDifference(mks_increment.momentum[2] / mks_velocity.lower[3],
                                               -mks.gdet * 0.03 * mks_velocity.upper[0]);

          const pangu::eos::IdealGas gas{4.0 / 3.0};
          const auto cooling =
              EvaluateTargetThicknessCooling(1.0, 0.4, gas, 10.0, 0.5, 0.1, 2.0, 0.25, 1.25, 1.0);
          const Real omega = 1.0 / (10.0 * sqrt(10.0) + 0.5);
          const Real target = 0.1 * 0.1 * (10.0 * omega) * (10.0 * omega) / (1.0 / 3.0);
          const Real fraction = 1.0 - exp(-(0.25 / 1.25) / (2.0 / omega));
          errors(check++) = RelativeDifference(cooling.target_internal_energy, target);
          errors(check++) = RelativeDifference(cooling.removed_fraction, fraction);
          errors(check++) =
              RelativeDifference(cooling.removed_internal_energy, (0.4 - target) * fraction);
          const auto stiff = EvaluateTargetThicknessCooling(1.0, 0.4, gas, 10.0, 0.5, 0.1, 1.0e-12,
                                                            1.0e12, 1.0, 1.0);
          errors(check++) =
              stiff.removed_internal_energy <= 0.4 - stiff.target_internal_energy ? 0.0 : 1.0;
          errors(check++) = fabs(CoolingRamp(-1.0, 0.0, 10.0));
          errors(check++) = RelativeDifference(CoolingRamp(5.0, 0.0, 10.0), 0.5);
          errors(check++) = RelativeDifference(CoolingRamp(11.0, 0.0, 10.0), 1.0);

          pangu::relativity::MHDPrimitiveState c2p_primitive{};
          c2p_primitive.fluid = rest;
          c2p_primitive.fluid.pressure = (4.0 / 3.0 - 1.0) * 0.4;
          const pangu::eos::RelativisticEOS eos{{4.0 / 3.0}, 1.0e-12, 1.0e-12, 1.0e-30};
          const auto c2p_base =
              pangu::relativity::ConvertGRMHDP2CWithInternalEnergy(c2p_primitive, 0.4, eos, flat);
          const auto safe_increment = BuildCoolingIncrement(rest, flat, 0.1);
          bool base_valid = false;
          const Real safe_scale = pangu::radiation::LimitCoolingIncrement(
              c2p_base, safe_increment, c2p_primitive.magnetic, eos, 100.0, 1.0e6, flat,
              base_valid);
          errors(check++) = base_valid ? fabs(safe_scale - 1.0) : 1.0;
          const auto c2p_cooled = pangu::radiation::AddCoolingIncrement(c2p_base, safe_increment);
          const auto recovered = pangu::relativity::SolveGRMHDC2P(
              c2p_cooled, c2p_primitive.magnetic, eos, 100.0, 1.0e6, flat);
          errors(check++) =
              recovered.success ? RelativeDifference(recovered.internal_energy, 0.3) : 1.0;
          for (int axis = 0; axis < 3; ++axis)
            errors(check++) = fabs(recovered.primitive.fluid.u[axis]);
          const auto excessive_increment = BuildCoolingIncrement(rest, flat, 0.39);
          auto raised_pressure_eos = eos;
          raised_pressure_eos.pressure_floor = 0.05;
          const Real limited_scale = pangu::radiation::LimitCoolingIncrement(
              c2p_base, excessive_increment, c2p_primitive.magnetic, raised_pressure_eos, 100.0,
              1.0e6, flat, base_valid);
          errors(check++) = base_valid && limited_scale > 0.0 && limited_scale < 1.0 ? 0.0 : 1.0;
        });
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
    Real maximum = 0.0;
    for (int check = 0; check < checks; ++check) {
      Require(std::isfinite(host(check)), "non-finite covariant cooling test result");
      maximum = fmax(maximum, host(check));
    }
    Require(maximum <= 2.0e-14, "covariant cooling vector exceeds the roundoff tolerance");
    std::cout << "PANGU RC-2 covariant cooling: PASS, max error=" << std::scientific << maximum
              << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "PANGU RC-2 covariant cooling: FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
