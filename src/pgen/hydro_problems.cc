#include "pgen/hydro_problems.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <Kokkos_Random.hpp>

#include "bvals/boundary_conditions.hpp"
#include "eos/newtonian_eos.h"
#include "geometry_assembly.h"
#include "globals.hpp"
#include "hydro/hydro_types.h"
#include "kokkos_abstraction.hpp"
#include "z4c/sync_grhd.h"
#include "pgen/bondi.h"
#include "relativity/relativistic_hydro.h"
#include "utils/error_checking.hpp"

namespace pangu::pgen {
using namespace parthenon;

namespace {

enum class ProblemType : int {
  advection,
  sod,
  linear_wave,
  blast,
  kh,
  rt,
  sr_shock,
  sr_linear_wave,
  sr_blast,
  gr_linear_wave,
  bondi,
  gr_torus
};

ProblemType GetProblemType(const std::string& name) {
  if (name == "hydro_advection")
    return ProblemType::advection;
  if (name == "sod")
    return ProblemType::sod;
  if (name == "linear_wave")
    return ProblemType::linear_wave;
  if (name == "blast")
    return ProblemType::blast;
  if (name == "kh")
    return ProblemType::kh;
  if (name == "rt")
    return ProblemType::rt;
  if (name == "sr_shock")
    return ProblemType::sr_shock;
  if (name == "sync_grhd_shock")
    return ProblemType::sr_shock;
  if (name == "sr_linear_wave")
    return ProblemType::sr_linear_wave;
  if (name == "sr_blast")
    return ProblemType::sr_blast;
  if (name == "gr_linear_wave")
    return ProblemType::gr_linear_wave;
  if (name == "sync_grhd_linear_wave")
    return ProblemType::gr_linear_wave;
  if (name == "bondi_cks" || name == "bondi_mks")
    return ProblemType::bondi;
  if (name == "gr_torus")
    return ProblemType::gr_torus;
  PARTHENON_FAIL("Unknown Newtonian Hydro problem_id='" + name + "'");
}

template <class GeometryType, class Coordinates, class Z4cPack, class ADMPack>
KOKKOS_INLINE_FUNCTION geometry::MetricPoint
InitialHydroMetric(const GeometryType& spacetime, const int geometry_block, const int k,
                   const int j, const int i, const Coordinates& coordinates, const Z4cPack& z4c,
                   const ADMPack& adm) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized()) {
    // A block-local VariablePack has a single leading pack entry.
    return spacetime.MetricAt(geometry::Location::cell_center, 0, k, j, i, coordinates, z4c, adm);
  } else {
    return spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i,
                              coordinates);
  }
}

KOKKOS_INLINE_FUNCTION relativity::HydroConservedState
ConvertInitialGRHDState(const relativity::HydroPrimitiveState& primitive,
                        const eos::RelativisticEOS& eos, const geometry::MetricPoint& metric) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized())
    return nr::ConvertSyncGRHDP2C(primitive, eos, metric);
  return relativity::ConvertGRHDP2C(primitive, eos, metric);
}

eos::NewtonianEOS GetEos(const std::shared_ptr<StateDescriptor>& package) {
  return {static_cast<hydro::EosMode>(package->Param<int>("eos_mode")),
          package->Param<Real>("gamma"), package->Param<Real>("iso_sound_speed"),
          package->Param<Real>("density_floor"), package->Param<Real>("pressure_floor")};
}

KOKKOS_INLINE_FUNCTION
Real PeriodicCoordinate(Real coordinate, const Real minimum, const Real maximum) {
  const Real length = maximum - minimum;
  while (coordinate < minimum)
    coordinate += length;
  while (coordinate >= maximum)
    coordinate -= length;
  return coordinate;
}

struct ProfileRow {
  Real x;
  Real y;
  Real z;
  Real density;
  Real velocity1;
  Real velocity2;
  Real velocity3;
  Real pressure;
};

struct RelHydroWave {
  Real density0;
  Real pressure0;
  Real velocity0[3];
  Real gamma;
  int wave_flag;
  Real eigenvalue;
  Real delta_density;
  Real delta_pressure;
  Real delta_velocity[3];
};

// Relativistic hydrodynamic primitive eigenvector used by AthenaK's linear-wave
// generator (Falle & Komissarov notation).  The returned perturbation has unit
// Euclidean norm in (rho,p,v1,v2,v3), exactly as in the reference generator.
RelHydroWave BuildRelHydroWave(const Real density, const Real pressure, const Real velocity1,
                               const Real velocity2, const Real velocity3,
                               const eos::IdealGas& eos, const int wave_flag) {
  PARTHENON_REQUIRE(wave_flag >= 0 && wave_flag <= 4,
                    "relativistic Hydro problem/wave_flag must be in [0,4]");
  const Real velocity_squared =
      velocity1 * velocity1 + velocity2 * velocity2 + velocity3 * velocity3;
  PARTHENON_REQUIRE(velocity_squared < 1.0,
                    "relativistic linear-wave background velocity must be subluminal");
  const Real u0 = 1.0 / std::sqrt(1.0 - velocity_squared);
  const Real u[4]{u0, u0 * velocity1, u0 * velocity2, u0 * velocity3};
  const Real enthalpy = eos.EnthalpyDensity(density, pressure);
  const Real sound_squared = eos.gamma * pressure / enthalpy;
  Real delta_density = 0.0;
  Real delta_pressure = 0.0;
  Real delta_u[4]{};
  Real eigenvalue = velocity1;
  if (wave_flag == 1) {
    delta_density = 1.0;
  } else if (wave_flag == 2) {
    delta_u[1] = velocity1 * velocity2 / (1.0 - velocity1 * velocity1);
    delta_u[2] = 1.0;
  } else if (wave_flag == 3) {
    delta_u[1] = velocity1 * velocity3 / (1.0 - velocity1 * velocity1);
    delta_u[3] = 1.0;
  } else {
    const Real delta = u0 * u0 * (1.0 - sound_squared) + sound_squared;
    const Real first = velocity1 * sound_squared;
    const Real second =
        std::sqrt(sound_squared * (u0 * u0 * (1.0 - sound_squared) * (1.0 - velocity1 * velocity1) +
                                   sound_squared));
    const Real velocity_minus_eigenvalue =
        wave_flag == 0 ? (first + second) / delta : (first - second) / delta;
    eigenvalue = velocity1 - velocity_minus_eigenvalue;
    delta_density = density;
    delta_pressure = enthalpy * sound_squared;
    delta_u[1] = -sound_squared * u[1] - sound_squared / u[0] / velocity_minus_eigenvalue;
    delta_u[2] = -sound_squared * u[2];
    delta_u[3] = -sound_squared * u[3];
  }
  Real delta_velocity[3]{((1.0 - velocity1 * velocity1) * delta_u[1] -
                          velocity1 * velocity2 * delta_u[2] - velocity1 * velocity3 * delta_u[3]) /
                             u0,
                         (-velocity1 * velocity2 * delta_u[1] +
                          (1.0 - velocity2 * velocity2) * delta_u[2] -
                          velocity2 * velocity3 * delta_u[3]) /
                             u0,
                         (-velocity1 * velocity3 * delta_u[1] - velocity2 * velocity3 * delta_u[2] +
                          (1.0 - velocity3 * velocity3) * delta_u[3]) /
                             u0};
  Real norm_squared = delta_density * delta_density + delta_pressure * delta_pressure;
  for (const auto value : delta_velocity)
    norm_squared += value * value;
  const Real inverse_norm = 1.0 / std::sqrt(norm_squared);
  RelHydroWave wave{density,
                    pressure,
                    {velocity1, velocity2, velocity3},
                    eos.gamma,
                    wave_flag,
                    eigenvalue,
                    delta_density * inverse_norm,
                    delta_pressure * inverse_norm,
                    {delta_velocity[0] * inverse_norm, delta_velocity[1] * inverse_norm,
                     delta_velocity[2] * inverse_norm}};
  return wave;
}

// Fishbone--Moncrief equilibrium torus parameters.  The analytic expressions
// below follow AthenaK's gr_torus generator (BSD-3-Clause), specialized to the
// fixed-metric, non-radiating, untilted hydrodynamic branch.
struct FMTorusParameters {
  Real spin;
  Real gamma;
  Real density_excision;
  Real pressure_excision;
  Real density_atmosphere;
  Real density_power;
  Real pressure_atmosphere;
  Real pressure_power;
  Real density_peak;
  Real radius_edge;
  Real radius_peak;
  Real angular_momentum;
  Real log_enthalpy_edge;
  Real density_normalization;
  bool prograde;
};

KOKKOS_INLINE_FUNCTION
Real FMTorusAngularMomentum(const Real spin, const Real radius, const bool prograde) {
  const Real sign = prograde ? 1.0 : -1.0;
  const Real numerator = sign * (radius * radius * radius * radius + spin * spin * radius * radius -
                                 2.0 * spin * spin * radius) -
                         spin * (radius * radius - spin * spin) * sqrt(radius);
  const Real denominator = radius * radius - 3.0 * radius + sign * 2.0 * spin * sqrt(radius);
  return numerator / (radius * sqrt(radius) * denominator);
}

KOKKOS_INLINE_FUNCTION
Real FMTorusLogEnthalpy(const Real spin, const Real angular_momentum, const Real radius,
                        const Real sin_theta) {
  const Real sin2 = sin_theta * sin_theta;
  const Real cos2 = 1.0 - sin2;
  const Real delta = radius * radius - 2.0 * radius + spin * spin;
  const Real sigma = radius * radius + spin * spin * cos2;
  const Real aa = (radius * radius + spin * spin) * (radius * radius + spin * spin) -
                  delta * spin * spin * sin2;
  const Real exp2nu = sigma * delta / aa;
  const Real exp2psi = aa / sigma * sin2;
  const Real exp_neg2chi = exp2nu / exp2psi;
  const Real omega = 2.0 * spin * radius / aa;
  const Real a = sqrt(1.0 + 4.0 * angular_momentum * angular_momentum * exp_neg2chi);
  return 0.5 * log((1.0 + a) / (sigma * delta / aa)) - 0.5 * a - angular_momentum * omega;
}

KOKKOS_INLINE_FUNCTION
void FMTorusBoyerLindquistVelocity(const FMTorusParameters& p, const Real radius,
                                   const Real sin_theta, Real& u0, Real& u3) {
  const Real sin2 = sin_theta * sin_theta;
  const Real cos2 = 1.0 - sin2;
  const Real spin2 = p.spin * p.spin;
  const Real delta = radius * radius - 2.0 * radius + spin2;
  const Real sigma = radius * radius + spin2 * cos2;
  const Real aa = (radius * radius + spin2) * (radius * radius + spin2) - delta * spin2 * sin2;
  const Real g00 = -(1.0 - 2.0 * radius / sigma);
  const Real g03 = -2.0 * p.spin * radius / sigma * sin2;
  const Real g33 = (sigma + (1.0 + 2.0 * radius / sigma) * spin2 * sin2) * sin2;
  const Real exp2nu = sigma * delta / aa;
  const Real exp2psi = aa / sigma * sin2;
  const Real projected2 =
      0.5 * (-1.0 + sqrt(1.0 + 4.0 * p.angular_momentum * p.angular_momentum * exp2nu / exp2psi));
  Real projected = sqrt(fmax(projected2, 0.0));
  projected *= p.prograde ? 1.0 : -1.0;
  const Real u3a = (1.0 + projected * projected) / (aa * sigma * delta);
  u3 = 2.0 * p.spin * radius * sqrt(u3a) + sqrt(sigma / aa) / sin_theta * projected;
  const Real u0a = (g03 * g03 - g00 * g33) * u3 * u3;
  u0 = -(g03 * u3 + sqrt(fmax(u0a - g00, 0.0))) / g00;
}

template <class GeometryType>
KOKKOS_INLINE_FUNCTION bool
FMTorusPrimitive(const FMTorusParameters& p, const Real x, const Real y, const Real z,
                 const Real dx, const Real dy, const Real dz, const GeometryType& spacetime,
                 const geometry::MetricPoint& metric, hydro::Primitive& state) {
  const auto spherical = spacetime.SphericalCoordinates(x, y, z);
  const Real radius = spherical.radius;
  const Real theta = spherical.theta;
  const Real sin_theta = fabs(sin(theta));
  const Real log_h =
      radius >= p.radius_edge
          ? FMTorusLogEnthalpy(p.spin, p.angular_momentum, radius, sin_theta) - p.log_enthalpy_edge
          : -1.0;

  bool inside_excision = false;
  if constexpr (GeometryType::supports_excision) {
    const auto outer_corner = spacetime.SphericalCoordinates(
        x + copysign(0.5 * dx, x), y + copysign(0.5 * dy, y), z + copysign(0.5 * dz, z));
    inside_excision = outer_corner.radius <= 1.0;
  }
  const Real background_density =
      inside_excision ? p.density_excision : p.density_atmosphere * pow(radius, p.density_power);
  const Real background_pressure =
      inside_excision ? p.pressure_excision : p.pressure_atmosphere * pow(radius, p.pressure_power);
  state = {background_density, {0.0, 0.0, 0.0}, background_pressure};
  if (!(log_h >= 0.0))
    return false;

  const Real gm1 = p.gamma - 1.0;
  const Real pressure_over_density = gm1 / p.gamma * (exp(log_h) - 1.0);
  const Real density = pow(pressure_over_density, 1.0 / gm1) / p.density_normalization;
  const Real pressure = pressure_over_density * density;
  state.density = fmax(density, background_density);
  state.pressure = fmax(pressure, background_pressure);

  Real u0_bl = 0.0, u3_bl = 0.0;
  FMTorusBoyerLindquistVelocity(p, radius, sin_theta, u0_bl, u3_bl);
  const Real boyer_lindquist[3]{0.0, 0.0, u3_bl};
  Real native[3]{};
  spacetime.BoyerLindquistSpatialToNative(x, y, z, boyer_lindquist, native);
  for (int axis = 0; axis < 3; ++axis)
    state.velocity[axis] = native[axis] - metric.upper[0][axis + 1] / metric.upper[0][0] * u0_bl;
  return true;
}

template <int Direction, bool Inner>
void TorusBoundary(std::shared_ptr<MeshBlockData<Real>>& data, const bool coarse) {
  // AthenaK's NoInflowTorus boundary converts the current stage conserved
  // state at the active boundary cell back to primitives, copies those
  // primitives into every ghost cell, clips only the outward-normal velocity,
  // and converts at the ghost cell's Kerr--Schild metric.  Copying conserved
  // quantities (the stock Parthenon outflow operation) is not equivalent in a
  // spatially varying metric.
  if (coarse)
    return;

  auto block = data->GetBlockPointer();
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const auto hydro_package = block->packages.Get("hydro");
  const auto geometry_package = block->packages.Get("geometry");
  const auto eos = pangu::eos::ReadRelativistic(*hydro_package);
  const Real gamma_max = hydro_package->Param<Real>("gamma_max");
  const auto background =
      static_cast<geometry::Background>(geometry_package->Param<int>("background"));
  const Real spin = geometry_package->Param<Real>("bh_spin");
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto ei = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto ej = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto ek = block->cellbounds.GetBoundsK(IndexDomain::entire);

  int il = ei.s, iu = ei.e, jl = ej.s, ju = ej.e, kl = ek.s, ku = ek.e;
  int source = 0;
  if constexpr (Direction == 0) {
    il = Inner ? ei.s : ib.e + 1;
    iu = Inner ? ib.s - 1 : ei.e;
    source = Inner ? ib.s : ib.e;
  } else if constexpr (Direction == 1) {
    jl = Inner ? ej.s : jb.e + 1;
    ju = Inner ? jb.s - 1 : ej.e;
    source = Inner ? jb.s : jb.e;
  } else {
    kl = Inner ? ek.s : kb.e + 1;
    ku = Inner ? kb.s - 1 : ek.e;
    source = Inner ? kb.s : kb.e;
  }
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  block->par_for(
      "PANGU FM torus no-inflow boundary", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int si = Direction == 0 ? source : i;
        const int sj = Direction == 1 ? source : j;
        const int sk = Direction == 2 ? source : k;
        const relativity::HydroConservedState source_conserved{conserved(hydro::IDN, sk, sj, si),
                                                               {conserved(hydro::IM1, sk, sj, si),
                                                                conserved(hydro::IM2, sk, sj, si),
                                                                conserved(hydro::IM3, sk, sj, si)},
                                                               conserved(hydro::IEN, sk, sj, si)};
        const auto source_metric =
            spacetime.MetricAt(geometry::Location::cell_center, geometry_block, sk, sj, si, coords);
        auto state =
            relativity::SolveGRHDC2P(source_conserved, eos, gamma_max, source_metric).primitive;
        state.u[Direction] = Inner ? fmin(0.0, state.u[Direction]) : fmax(0.0, state.u[Direction]);

        primitive(hydro::IDN, k, j, i) = state.density;
        primitive(hydro::IV1, k, j, i) = state.u[0];
        primitive(hydro::IV2, k, j, i) = state.u[1];
        primitive(hydro::IV3, k, j, i) = state.u[2];
        primitive(hydro::IPR, k, j, i) = state.pressure;
        const auto metric =
            spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
        const auto converted = relativity::ConvertGRHDP2C(state, eos, metric);
        conserved(hydro::IDN, k, j, i) = converted.density;
        conserved(hydro::IM1, k, j, i) = converted.momentum[0];
        conserved(hydro::IM2, k, j, i) = converted.momentum[1];
        conserved(hydro::IM3, k, j, i) = converted.momentum[2];
        conserved(hydro::IEN, k, j, i) = converted.energy;
      });
}

template <int Direction, bool Inner>
void BondiBoundary(std::shared_ptr<MeshBlockData<Real>>& data, const bool coarse) {
  if (coarse)
    return;

  auto block = data->GetBlockPointer();
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const auto hydro_package = block->packages.Get("hydro");
  const auto geometry_package = block->packages.Get("geometry");
  const auto eos = pangu::eos::ReadRelativistic(*hydro_package);
  const Real spin = geometry_package->Param<Real>("bh_spin");
  const Real configured_excision_density = geometry_package->Param<Real>("excision_density");
  const Real configured_excision_pressure = geometry_package->Param<Real>("excision_pressure");
  const auto bondi_parameters = bondi::MakeParameters(
      eos.gamma, hydro_package->Param<Real>("bondi_k_adi"),
      hydro_package->Param<Real>("bondi_r_crit"),
      configured_excision_density > 0.0 ? configured_excision_density : eos.density_floor,
      configured_excision_pressure > 0.0 ? configured_excision_pressure : eos.pressure_floor);
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto ei = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto ej = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto ek = block->cellbounds.GetBoundsK(IndexDomain::entire);
  int il = ei.s, iu = ei.e, jl = ej.s, ju = ej.e, kl = ek.s, ku = ek.e;
  if constexpr (Direction == 0) {
    il = Inner ? ei.s : ib.e + 1;
    iu = Inner ? ib.s - 1 : ei.e;
  } else if constexpr (Direction == 1) {
    jl = Inner ? ej.s : jb.e + 1;
    ju = Inner ? jb.s - 1 : ej.e;
  } else {
    kl = Inner ? ek.s : kb.e + 1;
    ku = Inner ? kb.s - 1 : ek.e;
  }
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  block->par_for(
      "PANGU fixed Bondi inflow boundary", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        hydro::Primitive state{};
        const auto metric =
            spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
        bondi::Primitive(bondi_parameters, coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k),
                         spacetime, metric, state);
        primitive(hydro::IDN, k, j, i) = state.density;
        primitive(hydro::IV1, k, j, i) = state.velocity[0];
        primitive(hydro::IV2, k, j, i) = state.velocity[1];
        primitive(hydro::IV3, k, j, i) = state.velocity[2];
        primitive(hydro::IPR, k, j, i) = state.pressure;
        const relativity::HydroPrimitiveState relativistic{
            state.density,
            {state.velocity[0], state.velocity[1], state.velocity[2]},
            state.pressure};
        const auto converted = relativity::ConvertGRHDP2C(relativistic, eos, metric);
        conserved(hydro::IDN, k, j, i) = converted.density;
        conserved(hydro::IM1, k, j, i) = converted.momentum[0];
        conserved(hydro::IM2, k, j, i) = converted.momentum[1];
        conserved(hydro::IM3, k, j, i) = converted.momentum[2];
        conserved(hydro::IEN, k, j, i) = converted.energy;
      });
}

} // namespace

void RegisterTorusBoundaries(ApplicationInput* app_input) {
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x1, "torus", TorusBoundary<0, true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x1, "torus", TorusBoundary<0, false>);
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x2, "torus", TorusBoundary<1, true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x2, "torus", TorusBoundary<1, false>);
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x3, "torus", TorusBoundary<2, true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x3, "torus", TorusBoundary<2, false>);
}

void RegisterBondiBoundaries(ApplicationInput* app_input) {
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x1, "bondi", BondiBoundary<0, true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x1, "bondi", BondiBoundary<0, false>);
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x2, "bondi", BondiBoundary<1, true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x2, "bondi", BondiBoundary<1, false>);
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x3, "bondi", BondiBoundary<2, true>);
  app_input->RegisterBoundaryCondition(BoundaryFace::outer_x3, "bondi", BondiBoundary<2, false>);
}

void HydroProblem(MeshBlock* block, ParameterInput* pin) {
  const auto problem_name = pin->GetString("parthenon/job", "problem_id");
  const auto problem = GetProblemType(problem_name);
  // Dense CSV is useful for compact regression cases but duplicates PHDF
  // output in production runs. Register the optional switch before the
  // unused-parameter audit; HydroAfterLoop consumes it.
  pin->GetOrAddBoolean("problem", "write_final_csv", true);
  const auto package = block->packages.Get("hydro");
  const auto eos = GetEos(package);
  const auto relativistic_eos = pangu::eos::ReadRelativistic(*package);
  const auto physics = static_cast<relativity::HydroMode>(package->Param<int>("physics_mode"));
  const auto geometry_package = block->packages.Get("geometry");
  const auto background =
      static_cast<geometry::Background>(geometry_package->Param<int>("background"));
  const Real spin = geometry_package->Param<Real>("bh_spin");
  const Real configured_excision_density = geometry_package->Param<Real>("excision_density");
  const Real configured_excision_pressure = geometry_package->Param<Real>("excision_pressure");
  const Real excision_density =
      configured_excision_density > 0.0 ? configured_excision_density : eos.density_floor;
  const Real excision_pressure =
      configured_excision_pressure > 0.0 ? configured_excision_pressure : eos.pressure_floor;
  const auto amplitude = pin->GetOrAddReal("problem", "amplitude", 1.0e-4);
  const auto radius = pin->GetOrAddReal("problem", "radius", 0.1);
  const auto width = pin->GetOrAddReal("problem", "width", 0.025);
  const auto x0 = pin->GetOrAddReal("problem", "x0", 0.0);
  const auto y0 = pin->GetOrAddReal("problem", "y0", 0.0);
  const auto z0 = pin->GetOrAddReal("problem", "z0", 0.0);
  const auto density_left = pin->GetOrAddReal("problem", "density_left", 1.0);
  const auto density_right = pin->GetOrAddReal("problem", "density_right", 0.125);
  const auto pressure_left = pin->GetOrAddReal("problem", "pressure_left", 1.0);
  const auto pressure_right = pin->GetOrAddReal("problem", "pressure_right", 0.1);
  const auto velocity_left = pin->GetOrAddReal("problem", "velocity_left", 0.0);
  const auto velocity_right = pin->GetOrAddReal("problem", "velocity_right", 0.0);
  const auto wave_velocity1 = pin->GetOrAddReal("problem", "vx0", 0.0);
  const auto wave_velocity2 = pin->GetOrAddReal("problem", "vy0", 0.0);
  const auto wave_velocity3 = pin->GetOrAddReal("problem", "vz0", 0.0);
  const auto wave_flag = pin->GetOrAddInteger("problem", "wave_flag", 4);
  const auto ambient_density = pin->GetOrAddReal("problem", "ambient_density", 1.0);
  const auto ambient_pressure = pin->GetOrAddReal("problem", "ambient_pressure", 1.0);
  const auto inner_pressure = pin->GetOrAddReal("problem", "inner_pressure", 10.0);
  const auto flow_velocity = pin->GetOrAddReal("problem", "velocity", 1.0);
  const auto density_inner = pin->GetOrAddReal("problem", "density_inner", 2.0);
  const auto density_outer = pin->GetOrAddReal("problem", "density_outer", 1.0);
  const auto shear_inner = pin->GetOrAddReal("problem", "shear_inner", 0.5);
  const auto shear_outer = pin->GetOrAddReal("problem", "shear_outer", -0.5);
  const auto interface = pin->GetOrAddReal("problem", "interface", 0.0);
  const int nscalars = package->Param<int>("nscalars");
  const Real bondi_k = package->Param<Real>("bondi_k_adi");
  const Real bondi_critical_radius = package->Param<Real>("bondi_r_crit");
  const auto bondi_parameters = bondi::MakeParameters(eos.gamma, bondi_k, bondi_critical_radius,
                                                       excision_density, excision_pressure);
  const bool torus_fm = pin->GetOrAddBoolean("problem", "fm_torus", true);
  const bool torus_prograde = pin->GetOrAddBoolean("problem", "prograde", true);
  const Real torus_edge = pin->GetOrAddReal("problem", "r_edge", 6.0);
  const Real torus_peak = pin->GetOrAddReal("problem", "r_peak", 12.0);
  const Real torus_density_peak = pin->GetOrAddReal("problem", "rho_max", 1.0);
  const Real torus_density_atmosphere =
      pin->GetOrAddReal("problem", "rho_min", package->Param<Real>("density_floor"));
  const Real torus_density_power = pin->GetOrAddReal("problem", "rho_pow", -1.5);
  const Real torus_pressure_atmosphere =
      pin->GetOrAddReal("problem", "pgas_min", package->Param<Real>("pressure_floor"));
  const Real torus_pressure_power = pin->GetOrAddReal("problem", "pgas_pow", -2.5);
  const Real torus_tilt = pin->GetOrAddReal("problem", "tilt_angle", 0.0);
  const Real torus_perturbation = pin->GetOrAddReal("problem", "pert_amp", 0.0);
  PARTHENON_REQUIRE(problem != ProblemType::gr_torus || torus_fm,
                    "PANGU gr_torus currently requires problem/fm_torus=true");
  PARTHENON_REQUIRE(problem != ProblemType::gr_torus || torus_tilt == 0.0,
                    "PANGU fixed-GR hydrodynamic torus currently requires tilt_angle=0");
  PARTHENON_REQUIRE(problem != ProblemType::gr_torus ||
                        (torus_perturbation >= 0.0 && torus_perturbation < 1.0),
                    "PANGU gr_torus requires problem/pert_amp in [0,1)");
  const Real torus_l = FMTorusAngularMomentum(spin, torus_peak, torus_prograde);
  const Real torus_log_edge = FMTorusLogEnthalpy(spin, torus_l, torus_edge, 1.0);
  const Real torus_log_peak = FMTorusLogEnthalpy(spin, torus_l, torus_peak, 1.0) - torus_log_edge;
  const Real torus_pressure_over_density_peak =
      (eos.gamma - 1.0) / eos.gamma * (exp(torus_log_peak) - 1.0);
  const Real torus_density_normalization =
      pow(torus_pressure_over_density_peak, 1.0 / (eos.gamma - 1.0)) / torus_density_peak;
  const FMTorusParameters torus{spin,
                                eos.gamma,
                                excision_density,
                                excision_pressure,
                                torus_density_atmosphere,
                                torus_density_power,
                                torus_pressure_atmosphere,
                                torus_pressure_power,
                                torus_density_peak,
                                torus_edge,
                                torus_peak,
                                torus_l,
                                torus_log_edge,
                                torus_density_normalization,
                                torus_prograde};
  RelHydroWave relativistic_wave{};
  if (problem == ProblemType::sr_linear_wave || problem == ProblemType::gr_linear_wave) {
    relativistic_wave = BuildRelHydroWave(ambient_density, ambient_pressure, wave_velocity1,
                                          wave_velocity2, wave_velocity3, relativistic_eos,
                                          wave_flag);
  }

  auto data = block->meshblock_data.Get();
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto fofc = data->PackVariables(std::vector<std::string>{"hydro.fofc"});
  parthenon::VariablePack<Real> z4c;
  parthenon::VariablePack<Real> adm;
  if constexpr (geometry::ConfiguredGeometryIsSynchronized()) {
    z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
    adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  }
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  const Real mesh_xmin = block->pmy_mesh->mesh_size.xmin(X1DIR);
  const Real mesh_xmax = block->pmy_mesh->mesh_size.xmax(X1DIR);
  const Real mesh_xlength = mesh_xmax - mesh_xmin;
  constexpr Real two_pi = 6.283185307179586476925286766559;
  const auto ndim = block->pmy_mesh->ndim;
  // Match AthenaK's torus pressure-perturbation mechanism: one XorShift64 pool
  // seeded by the MeshBlock gid, with one draw for every cell inside the torus.
  Kokkos::Random_XorShift64_Pool<> torus_random_pool(block->gid);
  const Real acceleration2 = package->Param<Real>("accel2");
  const auto kind = static_cast<int>(problem);
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;

  block->par_for(
      "PANGU Newtonian Hydro initial condition", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coordinates.Xc<1>(i);
        const Real y = coordinates.Xc<2>(j);
        const Real z = coordinates.Xc<3>(k);
        hydro::Primitive state{ambient_density, {0.0, 0.0, 0.0}, ambient_pressure};
        geometry::MetricPoint metric{};
        if (physics == relativity::HydroMode::gr) {
          metric = InitialHydroMetric(spacetime, geometry_block, k, j, i, coordinates, z4c, adm);
        }

        if (kind == static_cast<int>(ProblemType::advection)) {
          state.density = ambient_density + amplitude * sin(two_pi * (x - x0) / mesh_xlength);
          state.velocity[0] = flow_velocity;
        } else if (kind == static_cast<int>(ProblemType::sod)) {
          const bool left = x < x0;
          state.density = left ? density_left : density_right;
          state.pressure = left ? pressure_left : pressure_right;
          state.velocity[0] = left ? velocity_left : velocity_right;
        } else if (kind == static_cast<int>(ProblemType::linear_wave)) {
          const Real phase = two_pi * (x - x0) / mesh_xlength;
          const Real sound = eos.HasEnergy() ? sqrt(eos.gamma * ambient_pressure / ambient_density)
                                             : eos.iso_sound_speed;
          state.density = ambient_density + amplitude * sin(phase);
          state.velocity[0] = amplitude * sound / ambient_density * sin(phase);
          state.pressure = ambient_pressure + amplitude * sound * sound * sin(phase);
        } else if (kind == static_cast<int>(ProblemType::blast)) {
          Real radius_squared = (x - x0) * (x - x0);
          if (ndim >= 2)
            radius_squared += (y - y0) * (y - y0);
          if (ndim >= 3)
            radius_squared += (z - z0) * (z - z0);
          state.pressure = radius_squared < radius * radius ? inner_pressure : ambient_pressure;
        } else if (kind == static_cast<int>(ProblemType::kh)) {
          const Real distance = fabs(y - y0);
          const Real outer_fraction = 0.5 * (1.0 + tanh((distance - 0.25) / width));
          state.density = density_inner * (1.0 - outer_fraction) + density_outer * outer_fraction;
          state.velocity[0] = shear_inner * (1.0 - outer_fraction) + shear_outer * outer_fraction;
          const Real interface_distance = distance - 0.25;
          state.velocity[1] = amplitude * sin(4.0 * two_pi * x) *
                              exp(-interface_distance * interface_distance / (width * width));
        } else if (kind == static_cast<int>(ProblemType::rt)) {
          const bool above = y >= interface;
          state.density = above ? density_inner : density_outer;
          state.pressure = ambient_pressure + state.density * acceleration2 * (y - interface);
          state.pressure = fmax(state.pressure, eos.pressure_floor);
          state.velocity[1] = amplitude * (1.0 + cos(two_pi * x)) *
                              exp(-((y - interface) * (y - interface)) / (width * width));
        } else if (kind == static_cast<int>(ProblemType::sr_shock)) {
          const bool left = x < x0;
          state.density = left ? density_left : density_right;
          state.pressure = left ? pressure_left : pressure_right;
          state.velocity[0] = left ? velocity_left : velocity_right;
        } else if (kind == static_cast<int>(ProblemType::sr_linear_wave) ||
                   kind == static_cast<int>(ProblemType::gr_linear_wave)) {
          const Real phase = two_pi * (x - x0) / mesh_xlength;
          const Real perturbation = amplitude * sin(phase);
          state.density =
              relativistic_wave.density0 + perturbation * relativistic_wave.delta_density;
          state.pressure =
              relativistic_wave.pressure0 + perturbation * relativistic_wave.delta_pressure;
          Real minkowski_velocity[3];
          Real lorentz_inverse = 1.0;
          for (int axis = 0; axis < 3; ++axis) {
            minkowski_velocity[axis] = relativistic_wave.velocity0[axis] +
                                       perturbation * relativistic_wave.delta_velocity[axis];
            lorentz_inverse -= minkowski_velocity[axis] * minkowski_velocity[axis];
          }
          const Real lorentz = 1.0 / sqrt(lorentz_inverse);
          for (int axis = 0; axis < 3; ++axis)
            state.velocity[axis] = lorentz * minkowski_velocity[axis];
        } else if (kind == static_cast<int>(ProblemType::sr_blast)) {
          Real radius_squared = (x - x0) * (x - x0);
          if (ndim >= 2)
            radius_squared += (y - y0) * (y - y0);
          if (ndim >= 3)
            radius_squared += (z - z0) * (z - z0);
          state.pressure = radius_squared < radius * radius ? inner_pressure : ambient_pressure;
        } else if (kind == static_cast<int>(ProblemType::bondi)) {
          bondi::Primitive(bondi_parameters, x, y, z, spacetime, metric, state);
        } else if (kind == static_cast<int>(ProblemType::gr_torus)) {
          const bool in_torus = FMTorusPrimitive(
              torus, x, y, z, coordinates.Dxc<X1DIR>(k, j, i), coordinates.Dxc<X2DIR>(k, j, i),
              coordinates.Dxc<X3DIR>(k, j, i), spacetime, metric, state);
          if (in_torus && torus_perturbation > 0.0) {
            auto generator = torus_random_pool.get_state();
            const Real perturbation = 2.0 * torus_perturbation * (generator.frand() - 0.5);
            torus_random_pool.free_state(generator);
            state.pressure *= 1.0 + perturbation;
          }
        }

        if (!eos.HasEnergy()) {
          state.pressure = eos.iso_sound_speed * eos.iso_sound_speed * state.density;
        }
        hydro::FluxState converted{};
        relativity::HydroConservedState relativistic{};
        if (physics == relativity::HydroMode::newtonian) {
          converted = eos.PrimitiveToConservedAndFlux(state, 0);
        } else {
          const relativity::HydroPrimitiveState rel_state{
              state.density,
              {state.velocity[0], state.velocity[1], state.velocity[2]},
              state.pressure};
          if (physics == relativity::HydroMode::sr) {
            relativistic = relativity::ConvertSRHDP2C(rel_state, relativistic_eos);
          } else {
            relativistic = ConvertInitialGRHDState(rel_state, relativistic_eos, metric);
          }
          converted.conserved[hydro::IDN] = relativistic.density;
          converted.conserved[hydro::IM1] = relativistic.momentum[0];
          converted.conserved[hydro::IM2] = relativistic.momentum[1];
          converted.conserved[hydro::IM3] = relativistic.momentum[2];
          converted.conserved[hydro::IEN] = relativistic.energy;
        }
        const int components =
            eos.HasEnergy() ? hydro::kIdealComponents : hydro::kIsothermalComponents;
        for (int n = 0; n < components; ++n) {
          conserved(n, k, j, i) = converted.conserved[n];
        }
        primitive(hydro::IDN, k, j, i) = state.density;
        primitive(hydro::IV1, k, j, i) = state.velocity[0];
        primitive(hydro::IV2, k, j, i) = state.velocity[1];
        primitive(hydro::IV3, k, j, i) = state.velocity[2];
        if (eos.HasEnergy())
          primitive(hydro::IPR, k, j, i) = state.pressure;
        for (int scalar = 0; scalar < nscalars; ++scalar) {
          Real fraction = 0.0;
          if (kind == static_cast<int>(ProblemType::advection)) {
            fraction = 0.5 * (1.0 + sin(two_pi * (x - x0) / mesh_xlength));
          } else if (kind == static_cast<int>(ProblemType::kh)) {
            fraction = state.density > 0.5 * (density_inner + density_outer) ? 1.0 : 0.0;
          }
          primitive(components + scalar, k, j, i) = fraction;
          conserved(components + scalar, k, j, i) = state.density * fraction;
        }
        fofc(0, k, j, i) = 0.0;
      });
}

void HydroAfterLoop(Mesh* mesh, ParameterInput* pin, SimTime& time) {
  const auto problem_name = pin->GetString("parthenon/job", "problem_id");
  const auto problem = GetProblemType(problem_name);
  const bool write_final_csv = pin->GetBoolean("problem", "write_final_csv");
  const auto package = mesh->packages.Get("hydro");
  const auto eos = GetEos(package);
  const auto amplitude = pin->GetOrAddReal("problem", "amplitude", 1.0e-4);
  const auto x0 = pin->GetOrAddReal("problem", "x0", 0.0);
  const auto ambient_density = pin->GetOrAddReal("problem", "ambient_density", 1.0);
  const auto ambient_pressure = pin->GetOrAddReal("problem", "ambient_pressure", 1.0);
  const auto flow_velocity = pin->GetOrAddReal("problem", "velocity", 1.0);
  const bool has_analytic =
      problem == ProblemType::advection || problem == ProblemType::linear_wave;
  if (!write_final_csv && !has_analytic)
    return;
  const Real xmin = mesh->mesh_size.xmin(X1DIR);
  const Real xmax = mesh->mesh_size.xmax(X1DIR);
  const Real length = xmax - xmin;
  const Real sound =
      eos.HasEnergy() ? sqrt(eos.gamma * ambient_pressure / ambient_density) : eos.iso_sound_speed;
  constexpr Real two_pi = 6.283185307179586476925286766559;

  Real l1 = 0.0;
  Real l2 = 0.0;
  Real linf = 0.0;
  Real volume = 0.0;
  std::vector<ProfileRow> rows;
  for (const auto& block : mesh->block_list) {
    const auto data = block->meshblock_data.Get();
    const auto primitive = data->Get("hydro.prim").data.GetHostMirrorAndCopy();
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          const Real x = block->coords.Xc<1>(i);
          const Real y = block->coords.Xc<2>(j);
          const Real z = block->coords.Xc<3>(k);
          const Real density = primitive(hydro::IDN, k, j, i);
          const Real pressure = eos.HasEnergy()
                                    ? primitive(hydro::IPR, k, j, i)
                                    : eos.iso_sound_speed * eos.iso_sound_speed * density;
          if (write_final_csv) {
            rows.push_back({x, y, z, density, primitive(hydro::IV1, k, j, i),
                            primitive(hydro::IV2, k, j, i), primitive(hydro::IV3, k, j, i),
                            pressure});
          }
          if (has_analytic) {
            const Real speed = problem == ProblemType::advection ? flow_velocity : sound;
            const Real reference_x = PeriodicCoordinate(x - speed * time.time, xmin, xmax);
            const Real reference =
                ambient_density + amplitude * sin(two_pi * (reference_x - x0) / length);
            const Real error = fabs(density - reference);
            const Real cell_volume = block->coords.CellVolume(k, j, i);
            l1 += error * cell_volume;
            l2 += error * error * cell_volume;
            linf = std::max(linf, error);
            volume += cell_volume;
          }
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  if (has_analytic) {
    if (Globals::my_rank == 0) {
      PARTHENON_MPI_CHECK(
          MPI_Reduce(MPI_IN_PLACE, &l1, 1, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
      PARTHENON_MPI_CHECK(
          MPI_Reduce(MPI_IN_PLACE, &l2, 1, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
      PARTHENON_MPI_CHECK(
          MPI_Reduce(MPI_IN_PLACE, &linf, 1, MPI_PARTHENON_REAL, MPI_MAX, 0, MPI_COMM_WORLD));
      PARTHENON_MPI_CHECK(
          MPI_Reduce(MPI_IN_PLACE, &volume, 1, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
    } else {
      PARTHENON_MPI_CHECK(MPI_Reduce(&l1, &l1, 1, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
      PARTHENON_MPI_CHECK(MPI_Reduce(&l2, &l2, 1, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
      PARTHENON_MPI_CHECK(
          MPI_Reduce(&linf, &linf, 1, MPI_PARTHENON_REAL, MPI_MAX, 0, MPI_COMM_WORLD));
      PARTHENON_MPI_CHECK(
          MPI_Reduce(&volume, &volume, 1, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
    }
  }
#endif

  if (write_final_csv) {
    std::sort(rows.begin(), rows.end(), [](const ProfileRow& left, const ProfileRow& right) {
      if (left.z != right.z)
        return left.z < right.z;
      if (left.y != right.y)
        return left.y < right.y;
      return left.x < right.x;
    });
    char profile_name[96];
    if (Globals::nranks == 1) {
      std::snprintf(profile_name, sizeof(profile_name), "pangu-hydro-final.csv");
    } else {
      std::snprintf(profile_name, sizeof(profile_name), "pangu-hydro-final.rank%05d.csv",
                    Globals::my_rank);
    }
    auto* profile = std::fopen(profile_name, "w");
    PARTHENON_REQUIRE(profile != nullptr, "Unable to write PANGU Hydro final profile");
    std::fprintf(profile, "x,y,z,density,velocity1,velocity2,velocity3,pressure\n");
    for (const auto& row : rows) {
      std::fprintf(profile, "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                   static_cast<double>(row.x), static_cast<double>(row.y),
                   static_cast<double>(row.z), static_cast<double>(row.density),
                   static_cast<double>(row.velocity1), static_cast<double>(row.velocity2),
                   static_cast<double>(row.velocity3), static_cast<double>(row.pressure));
    }
    std::fclose(profile);
  }

  if (has_analytic && Globals::my_rank == 0) {
    auto* output = std::fopen("pangu-hydro-errors.dat", "w");
    PARTHENON_REQUIRE(output != nullptr, "Unable to write PANGU Hydro error file");
    std::fprintf(output, "# problem cycle time L1 L2 Linf\n");
    std::fprintf(output, "%s %d %.17e %.17e %.17e %.17e\n", problem_name.c_str(), time.ncycle,
                 static_cast<double>(time.time), static_cast<double>(l1 / volume),
                 static_cast<double>(sqrt(l2 / volume)), static_cast<double>(linf));
    std::fclose(output);
  }
}

} // namespace pangu::pgen
