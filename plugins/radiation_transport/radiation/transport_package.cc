#include "radiation/transport_package.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "pangu.h"
#include "radiation/geodesic_grid.h"
#include "radiation/task_contribution.h"
#include "reconstruct/hydro_reconstruction.h"

namespace pangu::radiation {
using namespace parthenon::package::prelude;

namespace {

hydro::Reconstruction ParseReconstruction(const std::string& name) {
  const auto method = reconstruct::Parse(name);
  PARTHENON_REQUIRE(method.has_value(), "radiation/reconstruct is not a registered reconstruction");
  PARTHENON_REQUIRE(reconstruct::Describe(*method).radiation_transport,
                    "radiation/reconstruct is not registered for radiation transport");
  return *method;
}

KOKKOS_INLINE_FUNCTION
bool FourthPolynomialRoot(const Real coefficient4, const Real constant, Real& root) {
  Real cubic = constant * constant * constant;
  Real delta1 = 0.25 - 64.0 * cubic * coefficient4 / 27.0;
  if (delta1 < 0.0)
    return false;
  delta1 = sqrt(delta1);
  if (delta1 < 0.5)
    return false;
  Real zroot = 0.0;
  if (delta1 > 1.0e11)
    zroot = pow(delta1, -2.0 / 3.0) / 3.0;
  else
    zroot = pow(0.5 + delta1, 1.0 / 3.0) - pow(-0.5 + delta1, 1.0 / 3.0);
  if (zroot < 0.0)
    return false;
  zroot *= pow(coefficient4, -2.0 / 3.0);
  const Real root_coefficient = sqrt(zroot);
  Real delta2 = -zroot + 2.0 / (coefficient4 * root_coefficient);
  if (delta2 < 0.0)
    return false;
  delta2 = sqrt(delta2);
  root = 0.5 * (delta2 - root_coefficient);
  return root >= 0.0;
}

template <int Direction, class Pack>
KOKKOS_INLINE_FUNCTION Real ReadIntensity(const Pack& intensity, const int block, const int angle,
                                          const int k, const int j, const int i, const int offset) {
  if constexpr (Direction == 0)
    return intensity(block, angle, k, j, i + offset);
  if constexpr (Direction == 1)
    return intensity(block, angle, k, j + offset, i);
  return intensity(block, angle, k + offset, j, i);
}

template <hydro::Reconstruction Method, int Direction, class Pack>
KOKKOS_INLINE_FUNCTION Real ReconstructUpwindIntensity(const Pack& intensity, const int block,
                                                       const int angle, const int k, const int j,
                                                       const int i, const Real speed) {
  Real left = 0.0;
  Real right = 0.0;
  reconstruct::ReconstructFaceStencil<Method>(
      [&](const int offset) {
        return ReadIntensity<Direction>(intensity, block, angle, k, j, i, offset);
      },
      left, right);
  return speed > 0.0 ? left : right;
}

template <hydro::Reconstruction Method, int Direction>
void CalculateDirectionalFluxes(MeshData<Real>* data) {
  auto intensity = data->PackVariablesAndFluxes(std::vector<std::string>{"radiation.intensity"});
  const auto directions = data->GetParentPointer()
                              ->packages.Get("radiation")
                              ->Param<Kokkos::View<Real**>>("angle_directions");
  const int nangles = data->GetParentPointer()->packages.Get("radiation")->Param<int>("nangles");
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  IndexRange krange = kb;
  IndexRange jrange = jb;
  IndexRange irange = ib;
  if constexpr (Direction == 0)
    irange.e += 1;
  else if constexpr (Direction == 1)
    jrange.e += 1;
  else
    krange.e += 1;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU radiation transport flux", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, 0, nangles - 1, krange.s, krange.e, jrange.s, jrange.e, irange.s,
      irange.e,
      KOKKOS_LAMBDA(const int block, const int angle, const int k, const int j, const int i) {
        const Real speed = directions(angle, Direction + 1);
        const Real upwind = ReconstructUpwindIntensity<Method, Direction>(
            intensity, block, angle, k, j, i, speed);
        intensity(block).flux(Direction + 1, angle, k, j, i) = speed * upwind;
      });
}

void FillMomentsMesh(MeshData<Real>* data) {
  const auto package = data->GetParentPointer()->packages.Get("radiation");
  const auto directions = package->Param<Kokkos::View<Real**>>("angle_directions");
  const auto weights = package->Param<Kokkos::View<Real*>>("solid_angles");
  const int nangles = package->Param<int>("nangles");
  const auto intensity = data->PackVariables(std::vector<std::string>{"radiation.intensity"});
  auto moments = data->PackVariables(std::vector<std::string>{"radiation.moments"});
  const auto ib = data->GetBoundsI(IndexDomain::entire);
  const auto jb = data->GetBoundsJ(IndexDomain::entire);
  const auto kb = data->GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU radiation moments", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        Real energy = 0.0;
        Real flux[3]{};
        for (int angle = 0; angle < nangles; ++angle) {
          const Real weighted = intensity(block, angle, k, j, i) * weights(angle);
          energy -= weighted;
          for (int axis = 0; axis < 3; ++axis)
            flux[axis] -= directions(angle, axis + 1) * weighted;
        }
        moments(block, 0, k, j, i) = energy;
        for (int axis = 0; axis < 3; ++axis)
          moments(block, axis + 1, k, j, i) = flux[axis];
      });
}

Real EstimateTransportTimestepMesh(MeshData<Real>* data) {
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int ndim = data->GetParentPointer()->ndim;
  const auto intensity = data->PackVariables(std::vector<std::string>{"radiation.intensity"});
  Real minimum = std::numeric_limits<Real>::max();
  ParReduce(
      "PANGU radiation transport timestep", 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i, Real& local) {
        const auto& coordinates = intensity.GetCoords(block);
        Real cell_dt = coordinates.Dxc<X1DIR>(k, j, i);
        if (ndim >= 2)
          cell_dt = fmin(cell_dt, coordinates.Dxc<X2DIR>(k, j, i));
        if (ndim >= 3)
          cell_dt = fmin(cell_dt, coordinates.Dxc<X3DIR>(k, j, i));
        local = fmin(local, cell_dt);
      },
      Kokkos::Min<Real>(minimum));
  return data->GetParentPointer()->packages.Get("hydro")->Param<Real>("cfl") * minimum;
}

} // namespace

std::shared_ptr<StateDescriptor> InitializeTransport(ParameterInput* pin,
                                            const parthenon::Packages_t& packages) {
  PARTHENON_REQUIRE(packages.AllPackages().count("hydro") == 1,
                    "the first transport gate requires a Hydro package");
  PARTHENON_REQUIRE(packages.AllPackages().count("mhd") == 0,
                    "MHD radiation transport is enabled after the Hydro gates");
  PARTHENON_REQUIRE(pin->GetString("geometry", "background") == "minkowski",
                    "the first transport gate is restricted to Minkowski geometry");
  const int nlevel = pin->GetInteger("radiation", "nlevel");
  const bool rotate = pin->GetOrAddBoolean("radiation", "rotate_geo", true);
  const bool angular_fluxes = pin->GetOrAddBoolean("radiation", "angular_fluxes", false);
  PARTHENON_REQUIRE(!angular_fluxes,
                    "angular radiation fluxes are not enabled in this transport gate");
  PARTHENON_REQUIRE(nlevel > 0 || !rotate,
                    "radiation/nlevel=0 is incompatible with rotate_geo=true");

  const auto quadrature = BuildGeodesicQuadrature(nlevel, rotate);
  const int nangles = static_cast<int>(quadrature.directions.size());

  auto package = std::make_shared<StateDescriptor>("radiation");
  package->AddParam<driver::StageContribution>(
      std::string(driver::stage_contribution_key), TransportStageContribution());
  package->AddParam<std::string>("backend", std::string("transport"));
  package->AddParam<int>("nlevel", nlevel);
  package->AddParam<int>("nangles", nangles);
  package->AddParam<int>("reconstruction", static_cast<int>(ParseReconstruction(pin->GetOrAddString(
                                               "radiation", "reconstruct", "plm"))));
  package->AddParam<bool>("rad_source", pin->GetOrAddBoolean("radiation", "rad_source", true));
  package->AddParam<bool>("fixed_fluid", pin->GetOrAddBoolean("radiation", "fixed_fluid", false));
  package->AddParam<bool>("affect_fluid", pin->GetOrAddBoolean("radiation", "affect_fluid", true));
  const bool power_opacity = pin->GetOrAddBoolean("radiation", "power_opacity", false);
  package->AddParam<bool>("power_opacity", power_opacity);
  const bool beam_source = pin->GetOrAddBoolean("radiation", "beam_source", false);
  package->AddParam<bool>("beam_source", beam_source);
  package->AddParam<Real>("beam_rate", pin->GetOrAddReal("radiation", "dii_dt", 0.0));
  package->AddParam<Real>("beam_position_1", pin->GetOrAddReal("problem", "pos_1", 0.0));
  package->AddParam<Real>("beam_position_2", pin->GetOrAddReal("problem", "pos_2", 0.0));
  package->AddParam<Real>("beam_position_3", pin->GetOrAddReal("problem", "pos_3", 0.0));
  package->AddParam<Real>("beam_direction_1", pin->GetOrAddReal("problem", "dir_1", 0.0));
  package->AddParam<Real>("beam_direction_2", pin->GetOrAddReal("problem", "dir_2", 0.0));
  package->AddParam<Real>("beam_direction_3", pin->GetOrAddReal("problem", "dir_3", 0.0));
  package->AddParam<Real>("beam_width", pin->GetOrAddReal("problem", "width", 0.0));
  package->AddParam<Real>("beam_spread", pin->GetOrAddReal("problem", "spread", 0.0));
  if (beam_source) {
    PARTHENON_REQUIRE(pin->GetString("geometry", "background") == "minkowski",
                      "the current beam-source gate requires Minkowski geometry");
    PARTHENON_REQUIRE(package->Param<Real>("beam_rate") >= 0.0,
                      "radiation/dii_dt must be non-negative");
    PARTHENON_REQUIRE(package->Param<Real>("beam_width") > 0.0,
                      "problem/width must be positive for a radiation beam");
    PARTHENON_REQUIRE(package->Param<Real>("beam_spread") > 0.0 &&
                          package->Param<Real>("beam_spread") <= 360.0,
                      "problem/spread must lie in (0,360] degrees");
  }
  package->AddParam<Real>("kappa_a", power_opacity ? pin->GetOrAddReal("radiation", "kappa_a", 0.0)
                                                   : pin->GetReal("radiation", "kappa_a"));
  package->AddParam<Real>("kappa_s", pin->GetReal("radiation", "kappa_s"));
  package->AddParam<Real>("kappa_p", pin->GetReal("radiation", "kappa_p"));
  package->AddParam<Real>("arad", pin->GetReal("radiation", "arad"));

  Kokkos::View<Real**> directions("PANGU radiation directions", nangles, 4);
  Kokkos::View<Real*> weights("PANGU radiation solid angles", nangles);
  auto host_directions = Kokkos::create_mirror_view(directions);
  auto host_weights = Kokkos::create_mirror_view(weights);
  for (int angle = 0; angle < nangles; ++angle) {
    host_directions(angle, 0) = 1.0;
    for (int axis = 0; axis < 3; ++axis)
      host_directions(angle, axis + 1) = quadrature.directions[angle][axis];
    host_weights(angle) = quadrature.solid_angles[angle];
  }
  Kokkos::deep_copy(directions, host_directions);
  Kokkos::deep_copy(weights, host_weights);
  package->AddParam<Kokkos::View<Real**>>("angle_directions", directions);
  package->AddParam<Kokkos::View<Real*>>("solid_angles", weights);

  std::vector<std::string> labels;
  labels.reserve(nangles);
  for (int angle = 0; angle < nangles; ++angle)
    labels.emplace_back("intensity_" + std::to_string(angle));
  Metadata intensity({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
                      Metadata::FillGhost, Metadata::Restart},
                     std::vector<int>{nangles}, labels);
  intensity.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                                  parthenon::refinement_ops::RestrictAverage>();
  package->AddField("radiation.intensity", intensity);
  package->AddField("radiation.moments",
                    Metadata({Metadata::Cell, Metadata::Derived}, std::vector<int>{4},
                             std::vector<std::string>{"energy", "flux_1", "flux_2", "flux_3"}));
  package->FillDerivedMesh = FillMomentsMesh;
  package->EstimateTimestepMesh = EstimateTransportTimestepMesh;
  return package;
}

TaskStatus CalculateTransportFluxesMeshTask(MeshData<Real>* data) {
  const auto reconstruction = static_cast<hydro::Reconstruction>(
      data->GetParentPointer()->packages.Get("radiation")->Param<int>("reconstruction"));
  reconstruct::VisitRadiationTransport(reconstruction, [&]<hydro::Reconstruction Method>() {
    CalculateDirectionalFluxes<Method, 0>(data);
    if (data->GetParentPointer()->ndim >= 2)
      CalculateDirectionalFluxes<Method, 1>(data);
    if (data->GetParentPointer()->ndim >= 3)
      CalculateDirectionalFluxes<Method, 2>(data);
  });
  return TaskStatus::complete;
}

TaskStatus UpdateTransportMeshTask(MeshData<Real>* current, MeshData<Real>* base, const Real gam0,
                                   const Real gam1, const Real beta_dt, MeshData<Real>* next) {
  const auto source =
      current->PackVariablesAndFluxes(std::vector<std::string>{"radiation.intensity"});
  const auto initial = base->PackVariables(std::vector<std::string>{"radiation.intensity"});
  auto output = next->PackVariables(std::vector<std::string>{"radiation.intensity"});
  const auto ib = current->GetBoundsI(IndexDomain::interior);
  const auto jb = current->GetBoundsJ(IndexDomain::interior);
  const auto kb = current->GetBoundsK(IndexDomain::interior);
  const int ndim = current->GetParentPointer()->ndim;
  const int nangles = current->GetParentPointer()->packages.Get("radiation")->Param<int>("nangles");
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU radiation RK update", parthenon::DevExecSpace(), 0,
      current->NumBlocks() - 1, 0, nangles - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int angle, const int k, const int j, const int i) {
        const auto block_source = source(block);
        const auto& coordinates = source.GetCoords(block);
        Real divergence = (block_source.flux(X1DIR, angle, k, j, i + 1) -
                           block_source.flux(X1DIR, angle, k, j, i)) /
                          coordinates.Dxc<X1DIR>(k, j, i);
        if (ndim >= 2)
          divergence += (block_source.flux(X2DIR, angle, k, j + 1, i) -
                         block_source.flux(X2DIR, angle, k, j, i)) /
                        coordinates.Dxc<X2DIR>(k, j, i);
        if (ndim >= 3)
          divergence += (block_source.flux(X3DIR, angle, k + 1, j, i) -
                         block_source.flux(X3DIR, angle, k, j, i)) /
                        coordinates.Dxc<X3DIR>(k, j, i);
        const Real updated = gam0 * source(block, angle, k, j, i) +
                             gam1 * initial(block, angle, k, j, i) - beta_dt * divergence;
        // In AthenaK i0=n^0 n_0 I.  Minkowski n^0 n_0=-1, so a physical
        // non-negative intensity is represented by a non-positive i0.
        output(block, angle, k, j, i) = fmin(updated, 0.0);
      });
  return TaskStatus::complete;
}

TaskStatus CoupleTransportToFluidMeshTask(MeshData<Real>* data, const Real beta_dt) {
  const auto package = data->GetParentPointer()->packages.Get("radiation");
  if (!package->Param<bool>("rad_source"))
    return TaskStatus::complete;
  const bool fixed_fluid = package->Param<bool>("fixed_fluid");
  const bool affect_fluid = package->Param<bool>("affect_fluid");
  const bool power_opacity = package->Param<bool>("power_opacity");
  const Real arad = package->Param<Real>("arad");
  const Real kappa_a = package->Param<Real>("kappa_a");
  const Real kappa_s = package->Param<Real>("kappa_s");
  const Real kappa_p = package->Param<Real>("kappa_p");
  const auto directions = package->Param<Kokkos::View<Real**>>("angle_directions");
  const auto weights = package->Param<Kokkos::View<Real*>>("solid_angles");
  const int nangles = package->Param<int>("nangles");
  const Real gamma = data->GetParentPointer()->packages.Get("hydro")->Param<Real>("gamma");
  const Real gamma_minus_one = gamma - 1.0;
  auto intensity = data->PackVariables(std::vector<std::string>{"radiation.intensity"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU implicit radiation-fluid coupling", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const Real density = primitive(block, hydro::IDN, k, j, i);
        const Real ux = primitive(block, hydro::IV1, k, j, i);
        const Real uy = primitive(block, hydro::IV2, k, j, i);
        const Real uz = primitive(block, hydro::IV3, k, j, i);
        const Real pressure = primitive(block, hydro::IPR, k, j, i);
        const Real temperature = pressure / density;
        const Real lorentz = sqrt(1.0 + ux * ux + uy * uy + uz * uz);
        // AthenaK input opacities are mass absorption/scattering coefficients;
        // its OpacityFunction converts them to inverse-length coefficients by
        // multiplying by the local rest-mass density.
        const Real sigma_absorption =
            power_opacity ? density * density * pow(gamma_minus_one / temperature, 3.5)
                          : density * kappa_a;
        const Real sigma_planck = power_opacity ? 0.0 : density * kappa_p;
        const Real dt_absorption = beta_dt * sigma_absorption;
        const Real dt_scattering = beta_dt * density * kappa_s;
        const Real dt_planck = beta_dt * sigma_planck;
        const Real proper_absorption = dt_absorption / lorentz;
        const Real proper_planck = dt_planck / lorentz;

        Real weight_sum = 0.0;
        Real suma1 = 0.0;
        Real suma2 = 0.0;
        for (int angle = 0; angle < nangles; ++angle) {
          const Real n0_comoving = lorentz - ux * directions(angle, 1) - uy * directions(angle, 2) -
                                   uz * directions(angle, 3);
          const Real omega_comoving = weights(angle) / (n0_comoving * n0_comoving);
          const Real intensity_comoving = -4.0 * M_PI * intensity(block, angle, k, j, i) *
                                          n0_comoving * n0_comoving * n0_comoving * n0_comoving;
          const Real inverse = 1.0 / (1.0 + (dt_absorption + dt_scattering) * n0_comoving);
          const Real inverse_weighted = n0_comoving * inverse;
          weight_sum += omega_comoving;
          suma1 += omega_comoving * inverse_weighted;
          suma2 += intensity_comoving * omega_comoving * inverse;
        }
        suma1 /= weight_sum;
        suma2 /= weight_sum;
        const Real suma3 = suma1 * (dt_scattering - dt_planck);
        suma1 *= dt_absorption + dt_planck;
        const Real coefficient4 = (proper_absorption + proper_planck -
                                   (proper_absorption + proper_planck) * suma1 / (1.0 - suma3)) *
                                  arad * gamma_minus_one / density;
        const Real constant = -temperature - (proper_absorption + proper_planck) * suma2 *
                                                 gamma_minus_one / (density * (1.0 - suma3));
        Real new_temperature = temperature;
        bool bad = false;
        if (fabs(coefficient4) > 1.0e-20) {
          bad = !FourthPolynomialRoot(coefficient4, constant, new_temperature) ||
                !isfinite(new_temperature);
        } else {
          new_temperature = -constant;
        }
        if (bad)
          return;

        const Real emission =
            arad * new_temperature * new_temperature * new_temperature * new_temperature;
        const Real mean_intensity = (suma1 * emission + suma2) / (1.0 - suma3);
        Real old_moment[4]{};
        Real new_moment[4]{};
        for (int angle = 0; angle < nangles; ++angle) {
          const Real old_value = intensity(block, angle, k, j, i);
          old_moment[0] += old_value * weights(angle);
          for (int axis = 0; axis < 3; ++axis)
            old_moment[axis + 1] += -directions(angle, axis + 1) * old_value * weights(angle);
          const Real n0_comoving = lorentz - ux * directions(angle, 1) - uy * directions(angle, 2) -
                                   uz * directions(angle, 3);
          const Real intensity_comoving =
              -4.0 * M_PI * old_value * n0_comoving * n0_comoving * n0_comoving * n0_comoving;
          const Real inverse = 1.0 / (1.0 + (dt_absorption + dt_scattering) * n0_comoving);
          const Real delta_comoving = ((dt_scattering - dt_planck) * mean_intensity +
                                       (dt_absorption + dt_planck) * emission -
                                       (dt_scattering + dt_absorption) * intensity_comoving) *
                                      n0_comoving * inverse;
          const Real physical =
              fmax(-old_value + delta_comoving / (4.0 * M_PI * n0_comoving * n0_comoving *
                                                  n0_comoving * n0_comoving),
                   0.0);
          const Real new_value = -physical;
          intensity(block, angle, k, j, i) = new_value;
          new_moment[0] += new_value * weights(angle);
          for (int axis = 0; axis < 3; ++axis)
            new_moment[axis + 1] += -directions(angle, axis + 1) * new_value * weights(angle);
        }
        if (!fixed_fluid && affect_fluid) {
          conserved(block, hydro::IEN, k, j, i) += old_moment[0] - new_moment[0];
          conserved(block, hydro::IM1, k, j, i) += old_moment[1] - new_moment[1];
          conserved(block, hydro::IM2, k, j, i) += old_moment[2] - new_moment[2];
          conserved(block, hydro::IM3, k, j, i) += old_moment[3] - new_moment[3];
        }
      });
  return TaskStatus::complete;
}

TaskStatus ApplyBeamSourceMeshTask(MeshData<Real>* data, const Real beta_dt) {
  const auto package = data->GetParentPointer()->packages.Get("radiation");
  if (!package->Param<bool>("beam_source"))
    return TaskStatus::complete;
  const int nangles = package->Param<int>("nangles");
  const auto directions = package->Param<Kokkos::View<Real**>>("angle_directions");
  const Real rate = package->Param<Real>("beam_rate");
  const Real position1 = package->Param<Real>("beam_position_1");
  const Real position2 = package->Param<Real>("beam_position_2");
  const Real position3 = package->Param<Real>("beam_position_3");
  const Real direction1 = package->Param<Real>("beam_direction_1");
  const Real direction2 = package->Param<Real>("beam_direction_2");
  const Real direction3 = package->Param<Real>("beam_direction_3");
  const Real half_width = 0.5 * package->Param<Real>("beam_width");
  const Real cosine_minimum = cos(0.5 * package->Param<Real>("beam_spread") * M_PI / 180.0);
  auto intensity = data->PackVariables(std::vector<std::string>{"radiation.intensity"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU flat radiation beam source", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const auto& coordinates = intensity.GetCoords(block);
        const Real dx1 = coordinates.Xc<1>(i) - position1;
        const Real dx2 = coordinates.Xc<2>(j) - position2;
        const Real dx3 = coordinates.Xc<3>(k) - position3;
        if (dx1 * dx1 + dx2 * dx2 + dx3 * dx3 >= half_width * half_width)
          return;
        for (int angle = 0; angle < nangles; ++angle) {
          const Real cosine = directions(angle, 1) * direction1 +
                              directions(angle, 2) * direction2 + directions(angle, 3) * direction3;
          if (cosine > cosine_minimum)
            intensity(block, angle, k, j, i) -= rate * beta_dt;
        }
      });
  return TaskStatus::complete;
}

} // namespace pangu::radiation
