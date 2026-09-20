#include "z4c/tov.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "utils/error_checking.hpp"

namespace pangu::nr {
namespace {

using parthenon::Real;

struct TOVState {
  Real pressure;
  Real mass;
  Real log_lapse;
  Real isotropic_radius;
};

TOVState RightHandSide(const TOVParameters& parameters, const Real radius,
                       const TOVState& state) {
  if (radius < 1.0e-3 * parameters.radial_step)
    return {0.0, 0.0, 0.0, 1.0};
  const Real pressure = fmax(state.pressure, 0.0);
  const Real density = pressure > 0.0
                           ? pow(pressure / parameters.polytropic_constant,
                                 1.0 / parameters.gamma)
                           : 0.0;
  const Real energy = density + pressure / (parameters.gamma - 1.0);
  const Real radial_metric = 1.0 / (1.0 - 2.0 * state.mass / radius);
  const Real gravity =
      (state.mass + 4.0 * M_PI * radius * radius * radius * pressure) / (radius * radius);
  return {-(energy + pressure) * radial_metric * gravity,
          4.0 * M_PI * radius * radius * energy, radial_metric * gravity,
          state.isotropic_radius / radius * sqrt(radial_metric)};
}

TOVState AddScaled(const TOVState& state, const TOVState& derivative, const Real scale) {
  return {fmax(state.pressure + scale * derivative.pressure, 0.0),
          state.mass + scale * derivative.mass,
          state.log_lapse + scale * derivative.log_lapse,
          state.isotropic_radius + scale * derivative.isotropic_radius};
}

Real Interpolate(const Real coordinate, const Real left_coordinate, const Real right_coordinate,
                 const Real left_value, const Real right_value) {
  if (right_coordinate == left_coordinate)
    return left_value;
  return left_value + (coordinate - left_coordinate) / (right_coordinate - left_coordinate) *
                          (right_value - left_value);
}

} // namespace

TOVProfile BuildPolytropicTOV(const TOVParameters& parameters) {
  PARTHENON_REQUIRE(parameters.central_density > 0.0,
                    "TOV central density must be positive");
  PARTHENON_REQUIRE(parameters.polytropic_constant > 0.0,
                    "TOV polytropic constant must be positive");
  PARTHENON_REQUIRE(parameters.gamma > 1.0, "TOV polytropic gamma must exceed one");
  PARTHENON_REQUIRE(parameters.density_floor > 0.0 &&
                        parameters.density_floor < parameters.central_density,
                    "TOV density floor must be positive and below the central density");
  PARTHENON_REQUIRE(parameters.radial_step > 0.0, "TOV radial step must be positive");
  PARTHENON_REQUIRE(parameters.maximum_points >= 3,
                    "TOV radial profile requires at least three points");

  const Real pressure_floor = parameters.polytropic_constant *
                              pow(parameters.density_floor, parameters.gamma);
  std::vector<Real> radius(parameters.maximum_points, 0.0);
  std::vector<Real> isotropic(parameters.maximum_points, 0.0);
  std::vector<Real> mass(parameters.maximum_points, 0.0);
  std::vector<Real> pressure(parameters.maximum_points, 0.0);
  std::vector<Real> log_lapse(parameters.maximum_points, 0.0);
  pressure[0] = parameters.polytropic_constant *
                pow(parameters.central_density, parameters.gamma);

  int surface_index = 0;
  for (int index = 0; index < parameters.maximum_points - 1; ++index) {
    const Real r0 = index * parameters.radial_step;
    const TOVState state{pressure[index], mass[index], log_lapse[index], isotropic[index]};
    const auto k1 = RightHandSide(parameters, r0, state);
    const auto k2 = RightHandSide(parameters, r0 + 0.5 * parameters.radial_step,
                                  AddScaled(state, k1, 0.5 * parameters.radial_step));
    const auto k3 = RightHandSide(parameters, r0 + 0.5 * parameters.radial_step,
                                  AddScaled(state, k2, 0.5 * parameters.radial_step));
    const auto k4 = RightHandSide(parameters, r0 + parameters.radial_step,
                                  AddScaled(state, k3, parameters.radial_step));
    const int next = index + 1;
    radius[next] = next * parameters.radial_step;
    pressure[next] = pressure[index] +
                     parameters.radial_step / 6.0 *
                         (k1.pressure + 2.0 * k2.pressure + 2.0 * k3.pressure + k4.pressure);
    mass[next] = mass[index] +
                 parameters.radial_step / 6.0 *
                     (k1.mass + 2.0 * k2.mass + 2.0 * k3.mass + k4.mass);
    log_lapse[next] = log_lapse[index] +
                      parameters.radial_step / 6.0 *
                          (k1.log_lapse + 2.0 * k2.log_lapse + 2.0 * k3.log_lapse +
                           k4.log_lapse);
    isotropic[next] = isotropic[index] +
                      parameters.radial_step / 6.0 *
                          (k1.isotropic_radius + 2.0 * k2.isotropic_radius +
                           2.0 * k3.isotropic_radius + k4.isotropic_radius);
    if (pressure[next] <= pressure_floor) {
      surface_index = next;
      break;
    }
  }
  PARTHENON_REQUIRE(surface_index > 1,
                    "TOV solver did not find the stellar surface; increase maximum_points");

  const int inside = surface_index - 1;
  const Real surface_radius =
      Interpolate(pressure_floor, pressure[inside], pressure[surface_index], radius[inside],
                  radius[surface_index]);
  const Real total_mass = Interpolate(surface_radius, radius[inside], radius[surface_index],
                                      mass[inside], mass[surface_index]);
  const Real surface_log_lapse =
      Interpolate(surface_radius, radius[inside], radius[surface_index], log_lapse[inside],
                  log_lapse[surface_index]);
  // Preserve AthenaK's discrete surface-matching convention.  The
  // Schwarzschild-coordinate pressure, mass, and lapse are interpolated to
  // the pressure floor, while the isotropic-radius accumulator keeps the
  // first radial integration point at or below that floor.  Mixing an
  // interpolated isotropic radius with AthenaK's convention changes the
  // global isotropic scale by O(dr) and consequently produces an O(dr)
  // mismatch in every spatial-metric component even though the underlying
  // TOV solution is otherwise identical.
  const Real surface_isotropic_unscaled = isotropic[surface_index];
  const Real surface_isotropic_radius =
      0.5 * (surface_radius - total_mass +
             sqrt(surface_radius * (surface_radius - 2.0 * total_mass)));
  const Real lapse_scale = sqrt(1.0 - 2.0 * total_mass / surface_radius) /
                           exp(surface_log_lapse);
  const Real isotropic_scale = surface_isotropic_radius / surface_isotropic_unscaled;

  radius[surface_index] = surface_radius;
  mass[surface_index] = total_mass;
  pressure[surface_index] = pressure_floor;
  log_lapse[surface_index] = surface_log_lapse;
  isotropic[surface_index] = surface_isotropic_unscaled;
  const int points = surface_index + 1;

  TOVProfile profile{};
  profile.schwarzschild_radius = Kokkos::View<Real*>("TOV Schwarzschild radius", points);
  profile.isotropic_radius = Kokkos::View<Real*>("TOV isotropic radius", points);
  profile.mass = Kokkos::View<Real*>("TOV enclosed mass", points);
  profile.pressure = Kokkos::View<Real*>("TOV pressure", points);
  profile.lapse = Kokkos::View<Real*>("TOV lapse", points);
  auto host_radius = Kokkos::create_mirror_view(profile.schwarzschild_radius);
  auto host_isotropic = Kokkos::create_mirror_view(profile.isotropic_radius);
  auto host_mass = Kokkos::create_mirror_view(profile.mass);
  auto host_pressure = Kokkos::create_mirror_view(profile.pressure);
  auto host_lapse = Kokkos::create_mirror_view(profile.lapse);
  for (int index = 0; index < points; ++index) {
    host_radius(index) = radius[index];
    host_isotropic(index) = isotropic[index] * isotropic_scale;
    host_mass(index) = mass[index];
    host_pressure(index) = fmax(pressure[index], pressure_floor);
    host_lapse(index) = exp(log_lapse[index]) * lapse_scale;
  }
  Kokkos::deep_copy(profile.schwarzschild_radius, host_radius);
  Kokkos::deep_copy(profile.isotropic_radius, host_isotropic);
  Kokkos::deep_copy(profile.mass, host_mass);
  Kokkos::deep_copy(profile.pressure, host_pressure);
  Kokkos::deep_copy(profile.lapse, host_lapse);
  profile.central_density = parameters.central_density;
  profile.polytropic_constant = parameters.polytropic_constant;
  profile.gamma = parameters.gamma;
  profile.density_floor = parameters.density_floor;
  profile.pressure_floor = pressure_floor;
  profile.surface_radius = surface_radius;
  profile.surface_isotropic_radius = surface_isotropic_radius;
  profile.total_mass = total_mass;
  profile.points = points;
  return profile;
}

TOVProfile BuildPolytropicTOV(parthenon::ParameterInput* pin) {
  TOVParameters parameters{};
  parameters.central_density =
      pin->GetOrAddReal("problem", "rhoc", parameters.central_density);
  parameters.polytropic_constant =
      pin->GetOrAddReal("problem", "kappa", parameters.polytropic_constant);
  if (pin->DoesParameterExist("hydro", "gamma"))
    parameters.gamma = pin->GetReal("hydro", "gamma");
  else
    parameters.gamma = pin->GetReal("mhd", "gamma");
  parameters.density_floor = pin->GetOrAddReal(
      "problem", "rho_cut", parameters.central_density * 1.0e-10);
  parameters.radial_step = pin->GetOrAddReal("problem", "dr", parameters.radial_step);
  parameters.maximum_points =
      pin->GetOrAddInteger("problem", "npoints", parameters.maximum_points);
  return BuildPolytropicTOV(parameters);
}

} // namespace pangu::nr
