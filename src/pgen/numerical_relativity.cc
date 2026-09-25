#include "pgen/numerical_relativity.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "bvals/boundary_conditions.hpp"
#include "bvals/boundary_conditions_generic.hpp"
#include "geometry_assembly.h"
#include "globals.hpp"
#include "hydro/hydro_types.h"
#include "mhd/mhd_types.h"
#include "pangu_config.h"
#include "pgen/hydro_problems.h"
#include "pgen/relativistic_mhd_wave.h"
#include "relativity/relativistic_hydro.h"
#include "utils/error_checking.hpp"
#include "z4c/core/adm_conversion.h"
#include "z4c/core/component_indices.h"
#include "z4c/core/finite_difference.h"
#include "z4c/coupling/sync_grhd.h"
#include "z4c/coupling/sync_grmhd.h"
#include "z4c/evolution/package.h"
#include "z4c/evolution/z4c_rhs.h"
#include "z4c/initial_data/analytic.h"
#include "z4c/initial_data/tov.h"
#include "z4c/puncture/initializer.h"

namespace pangu::pgen {
using namespace parthenon;

namespace {

namespace boundary_names {
PAR_VAR(nr, z4c);
} // namespace boundary_names

template <class Pack> struct InverseMetricAccessor {
  Pack z4c;
  int first;
  int second;
  int k;
  int j;
  int i;

  KOKKOS_INLINE_FUNCTION Real operator()(const int dk, const int dj,
                                         const int di) const {
    Real metric[6]{};
    for (int component = 0; component < 6; ++component)
      metric[component] = z4c(nr::Index(nr::Z4cComponent::gxx) + component,
                              k + dk, j + dj, i + di);
    const Real determinant = nr::rhs::SpatialDeterminant(metric);
    Real inverse[6]{};
    nr::rhs::SpatialInverse(1.0 / determinant, metric, inverse);
    return inverse[nr::SpatialSymmetricComponent(first, second)];
  }
};

template <int Order>
void InitializeGaugeWaveGamma(MeshBlock *block,
                              const parthenon::VariablePack<Real> &z4c) {
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  block->par_for(
      "PANGU NR gauge-wave conformal connection", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real inverse_spacing[3] = {1.0 / coordinates.Dxc<X1DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X2DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X3DIR>(k, j, i)};
        for (int upper = 0; upper < 3; ++upper) {
          Real gamma = 0.0;
          for (int derivative = 0; derivative < 3; ++derivative) {
            const InverseMetricAccessor<parthenon::VariablePack<Real>> accessor{
                z4c, derivative, upper, k, j, i};
            gamma -= nr::fd::First<Order>(
                derivative, inverse_spacing[derivative], accessor);
          }
          z4c(nr::Index(nr::Z4cComponent::gamx) + upper, k, j, i) = gamma;
        }
      });
}

KOKKOS_INLINE_FUNCTION Real DeterministicNoise(const std::uint64_t plane,
                                               const std::uint64_t component) {
  std::uint64_t value = 0x9e3779b97f4a7c15ULL * (plane + 1ULL) ^
                        0xbf58476d1ce4e5b9ULL * (component + 1ULL);
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  value ^= value >> 31U;
  const Real unit = static_cast<Real>(value >> 11U) * 0x1.0p-53;
  return 2.0 * unit - 1.0;
}

void InitializeConstraintMask(MeshBlock *block, const Real mask_radius = 0.0) {
  auto data = block->meshblock_data.Get();
  const auto mask =
      data->PackVariables(std::vector<std::string>{"nr.constraint_mask"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coordinates = block->coords;
  const Real mask_radius_squared = mask_radius * mask_radius;
  block->par_for(
      "PANGU NR initialize constraint mask", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coordinates.Xc<X1DIR>(k, j, i);
        const Real y = coordinates.Xc<X2DIR>(k, j, i);
        const Real z = coordinates.Xc<X3DIR>(k, j, i);
        const bool masked =
            mask_radius > 0.0 && x * x + y * y + z * z <= mask_radius_squared;
        mask(0, k, j, i) = masked ? 0.0 : 1.0;
      });
}

void FinalizeNumericalRelativityInitialData(MeshBlockData<Real> *data) {
  nr::BuildStressEnergyBlockTask(data, 0.0);
  nr::Z4cToADMBlockTask(data);
}

void InitializeMinkowskiZ4cFields(MeshBlock *block) {
  InitializeConstraintMask(block);
  auto data = block->meshblock_data.Get();
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  // Initialize ghost cells as well. Parthenon's first package-level
  // FillDerived pass precedes the first boundary exchange, and the coupled
  // GRHD C2P callback needs a valid stage-zero metric at every cell it visits.
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  block->par_for(
      "PANGU NR Minkowski initial data", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        for (int component = 0; component < nr::kZ4cComponents; ++component)
          z4c(component, k, j, i) = 0.0;
        z4c(nr::Index(nr::Z4cComponent::chi), k, j, i) = 1.0;
        z4c(nr::Index(nr::Z4cComponent::gxx), k, j, i) = 1.0;
        z4c(nr::Index(nr::Z4cComponent::gyy), k, j, i) = 1.0;
        z4c(nr::Index(nr::Z4cComponent::gzz), k, j, i) = 1.0;
        z4c(nr::Index(nr::Z4cComponent::alpha), k, j, i) = 1.0;
      });
}

struct TOVMetricAccessor {
  nr::TOVProfile profile;
  Real x;
  Real y;
  Real z;
  Real dx;
  Real dy;
  Real dz;
  int component;

  KOKKOS_INLINE_FUNCTION Real operator()(const int dk, const int dj,
                                         const int di) const {
    if (component != nr::SpatialSymmetricComponent(0, 0) &&
        component != nr::SpatialSymmetricComponent(1, 1) &&
        component != nr::SpatialSymmetricComponent(2, 2))
      return 0.0;
    const Real shifted_x = x + di * dx;
    const Real shifted_y = y + dj * dy;
    const Real shifted_z = z + dk * dz;
    const Real radius = sqrt(shifted_x * shifted_x + shifted_y * shifted_y +
                             shifted_z * shifted_z);
    return profile.EvaluateIsotropic(radius).conformal_factor_four;
  }
};

template <int Order>
void InitializeSyncGRHDTOV(MeshBlock *block, ParameterInput *pin) {
  InitializeConstraintMask(block);
  auto data = block->meshblock_data.Get();
  const auto nr_package = block->packages.Get("numerical_relativity");
  const auto hydro_package = block->packages.Get("hydro");
  const auto profile = nr_package->Param<nr::TOVProfile>("tov_profile");
  const auto options = nr_package->Param<nr::Z4cOptions>("z4c_options");
  const auto eos = pangu::eos::ReadRelativistic(*hydro_package);
  const Real radial_velocity = pin->GetOrAddReal("problem", "v_pert", 0.0);
  pin->GetOrAddBoolean("problem", "write_final_csv", false);
  PARTHENON_REQUIRE(fabs(eos.gamma - profile.gamma) <=
                        16.0 * std::numeric_limits<Real>::epsilon(),
                    "TOV polytropic gamma must match hydro/gamma");
  PARTHENON_REQUIRE(
      eos.density_floor <= profile.density_floor,
      "hydro density floor must not exceed the TOV atmosphere density");
  PARTHENON_REQUIRE(
      eos.pressure_floor <= profile.pressure_floor,
      "hydro pressure floor must not exceed the TOV atmosphere pressure");
  PARTHENON_REQUIRE(
      fabs(radial_velocity) < 1.0,
      "TOV radial velocity perturbation must have magnitude below one");

  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto primitive =
      data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coordinates = block->coords;
  block->par_for(
      "PANGU SYNC TOV ADM and primitive initial data", kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coordinates.Xc<X1DIR>(k, j, i);
        const Real y = coordinates.Xc<X2DIR>(k, j, i);
        const Real z = coordinates.Xc<X3DIR>(k, j, i);
        const Real radius = sqrt(x * x + y * y + z * z);
        const auto point = profile.EvaluateIsotropic(radius);

        nr::ADMState adm_state{};
        adm_state.lapse = point.lapse;
        for (int axis = 0; axis < 3; ++axis)
          adm_state.metric[nr::SpatialSymmetricComponent(axis, axis)] =
              point.conformal_factor_four;
        nr::ADMMetricDerivatives derivatives{};
        const Real spacing[3] = {coordinates.Dxc<X1DIR>(k, j, i),
                                 coordinates.Dxc<X2DIR>(k, j, i),
                                 coordinates.Dxc<X3DIR>(k, j, i)};
        for (int derivative = 0; derivative < 3; ++derivative) {
          for (int component = 0; component < 6; ++component) {
            const TOVMetricAccessor accessor{profile,    x,          y,
                                             z,          spacing[0], spacing[1],
                                             spacing[2], component};
            derivatives.dmetric[derivative][component] = nr::fd::First<Order>(
                derivative, 1.0 / spacing[derivative], accessor);
          }
        }
        nr::Z4cState converted{};
        if (!nr::ConvertADMToZ4c(adm_state, derivatives, options.chi_psi_power,
                                 converted))
          Kokkos::abort("failed to convert TOV ADM initial data to Z4c");
        for (int component = 0; component < nr::kZ4cComponents; ++component)
          z4c(component, k, j, i) = converted.values[component];

        primitive(hydro::IDN, k, j, i) = fmax(point.density, eos.density_floor);
        primitive(hydro::IPR, k, j, i) =
            fmax(point.pressure, eos.pressure_floor);
        for (int axis = 0; axis < 3; ++axis)
          primitive(hydro::IV1 + axis, k, j, i) = 0.0;
        if (point.inside && radius > 0.0 && radial_velocity != 0.0) {
          const Real fraction =
              point.schwarzschild_radius / profile.surface_radius;
          // Match AthenaK's dynamic TOV convention exactly: the perturbation
          // parameter initializes the primitive projected spatial four-
          // velocity W v^i, not the Eulerian three-velocity v^i.  Applying a
          // second Lorentz/metric conversion here changes the prescribed
          // initial data in curved space and is visible before the first RK
          // stage.
          const Real projected_radial_velocity =
              0.5 * radial_velocity *
              (3.0 * fraction - fraction * fraction * fraction);
          const Real normalization = projected_radial_velocity / radius;
          primitive(hydro::IV1, k, j, i) = normalization * x;
          primitive(hydro::IV2, k, j, i) = normalization * y;
          primitive(hydro::IV3, k, j, i) = normalization * z;
        }
      });

  nr::EnforceAlgebraicConstraintsBlockTask(data.get());
  nr::Z4cToADMFieldsBlockTask(data.get());
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto conserved =
      data->PackVariables(std::vector<std::string>{"hydro.cons"});
  const auto fofc = data->PackVariables(std::vector<std::string>{"hydro.fofc"});
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int nscalars = hydro_package->Param<int>("nscalars");
  block->par_for(
      "PANGU SYNC TOV Valencia initial data", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto metric = spacetime.MetricAt(
            geometry::Location::cell_center, 0, k, j, i, coordinates, z4c, adm);
        const relativity::HydroPrimitiveState state{
            primitive(hydro::IDN, k, j, i),
            {primitive(hydro::IV1, k, j, i), primitive(hydro::IV2, k, j, i),
             primitive(hydro::IV3, k, j, i)},
            primitive(hydro::IPR, k, j, i)};
        const auto converted = nr::ConvertSyncGRHDP2C(state, eos, metric);
        conserved(hydro::IDN, k, j, i) = converted.density;
        conserved(hydro::IM1, k, j, i) = converted.momentum[0];
        conserved(hydro::IM2, k, j, i) = converted.momentum[1];
        conserved(hydro::IM3, k, j, i) = converted.momentum[2];
        conserved(hydro::IEN, k, j, i) = converted.energy;
        for (int scalar = 0; scalar < nscalars; ++scalar) {
          primitive(hydro::kIdealComponents + scalar, k, j, i) = 0.0;
          conserved(hydro::kIdealComponents + scalar, k, j, i) = 0.0;
        }
        fofc(0, k, j, i) = 0.0;
      });
  nr::BuildStressEnergyBlockTask(data.get(), 0.0);
  nr::Z4cToADMBlockTask(data.get());
}

template <int Component>
KOKKOS_INLINE_FUNCTION Real TOVVectorPotential(const nr::TOVProfile &profile,
                                               const Real pressure_cut,
                                               const Real magnetic_index,
                                               const Real x, const Real y,
                                               const Real z) {
  const Real radius = sqrt(x * x + y * y + z * z);
  const auto point = profile.EvaluateIsotropic(radius);
  const Real pressure_weight = fmax(point.pressure - pressure_cut, 0.0);
  const Real density_weight = pow(
      fmax(1.0 - point.density / profile.central_density, 0.0), magnetic_index);
  if constexpr (Component == 0)
    return -y * pressure_weight * density_weight;
  if constexpr (Component == 1)
    return x * pressure_weight * density_weight;
  return 0.0;
}

template <int Order>
void InitializeSyncGRMHDTOV(MeshBlock *block, ParameterInput *pin) {
  InitializeConstraintMask(block);
  auto data = block->meshblock_data.Get();
  const auto nr_package = block->packages.Get("numerical_relativity");
  const auto mhd_package = block->packages.Get("mhd");
  const auto profile = nr_package->Param<nr::TOVProfile>("tov_profile");
  const auto options = nr_package->Param<nr::Z4cOptions>("z4c_options");
  const auto eos = pangu::eos::ReadRelativistic(*mhd_package);
  const Real radial_velocity = pin->GetOrAddReal("problem", "v_pert", 0.0);
  const Real magnetic_normalization =
      pin->GetOrAddReal("problem", "b_norm", 1.0);
  const Real pressure_cut = pin->GetOrAddReal("problem", "pcut", 1.0e-6);
  const Real magnetic_index = pin->GetOrAddReal("problem", "magindex", 1.0);
  pin->GetOrAddBoolean("problem", "write_final_csv", false);
  PARTHENON_REQUIRE(fabs(eos.gamma - profile.gamma) <=
                        16.0 * std::numeric_limits<Real>::epsilon(),
                    "magnetized TOV polytropic gamma must match mhd/gamma");
  PARTHENON_REQUIRE(
      eos.density_floor <= profile.density_floor,
      "MHD density floor must not exceed the TOV atmosphere density");
  PARTHENON_REQUIRE(
      eos.pressure_floor <= profile.pressure_floor,
      "MHD pressure floor must not exceed the TOV atmosphere pressure");
  PARTHENON_REQUIRE(
      fabs(radial_velocity) < 1.0,
      "TOV radial velocity perturbation must have magnitude below one");
  PARTHENON_REQUIRE(magnetic_normalization >= 0.0,
                    "TOV magnetic normalization must be nonnegative");
  PARTHENON_REQUIRE(pressure_cut >= 0.0,
                    "TOV magnetic pressure cut must be nonnegative");
  PARTHENON_REQUIRE(magnetic_index >= 0.0,
                    "TOV magnetic index must be nonnegative");

  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto primitive =
      data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto divb = data->PackVariables(std::vector<std::string>{"mhd.divb"});
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto recovery = data->PackVariables(std::vector<std::string>{"mhd.recovery"});
  auto bface = data->Get("mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto interior_ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto interior_jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto interior_kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;

  block->par_for(
      "PANGU SYNC magnetized TOV ADM and primitive initial data", kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coordinates.Xc<X1DIR>(k, j, i);
        const Real y = coordinates.Xc<X2DIR>(k, j, i);
        const Real z = coordinates.Xc<X3DIR>(k, j, i);
        const Real radius = sqrt(x * x + y * y + z * z);
        const auto point = profile.EvaluateIsotropic(radius);
        nr::ADMState adm_state{};
        adm_state.lapse = point.lapse;
        for (int axis = 0; axis < 3; ++axis)
          adm_state.metric[nr::SpatialSymmetricComponent(axis, axis)] =
              point.conformal_factor_four;
        nr::ADMMetricDerivatives derivatives{};
        const Real spacing[3] = {coordinates.Dxc<X1DIR>(k, j, i),
                                 coordinates.Dxc<X2DIR>(k, j, i),
                                 coordinates.Dxc<X3DIR>(k, j, i)};
        for (int derivative = 0; derivative < 3; ++derivative) {
          for (int component = 0; component < 6; ++component) {
            const TOVMetricAccessor accessor{profile,    x,          y,
                                             z,          spacing[0], spacing[1],
                                             spacing[2], component};
            derivatives.dmetric[derivative][component] = nr::fd::First<Order>(
                derivative, 1.0 / spacing[derivative], accessor);
          }
        }
        nr::Z4cState converted{};
        if (!nr::ConvertADMToZ4c(adm_state, derivatives, options.chi_psi_power,
                                 converted))
          Kokkos::abort(
              "failed to convert magnetized TOV ADM initial data to Z4c");
        for (int component = 0; component < nr::kZ4cComponents; ++component)
          z4c(component, k, j, i) = converted.values[component];

        primitive(mhd::IDN, k, j, i) = fmax(point.density, eos.density_floor);
        primitive(mhd::IPR, k, j, i) =
            eos.InternalEnergyDensity(fmax(point.pressure, eos.pressure_floor));
        for (int axis = 0; axis < 3; ++axis)
          primitive(mhd::IV1 + axis, k, j, i) = 0.0;
        if (point.inside && radius > 0.0 && radial_velocity != 0.0) {
          const Real fraction =
              point.schwarzschild_radius / profile.surface_radius;
          const Real projected_radial_velocity =
              0.5 * radial_velocity *
              (3.0 * fraction - fraction * fraction * fraction);
          const Real normalization = projected_radial_velocity / radius;
          primitive(mhd::IV1, k, j, i) = normalization * x;
          primitive(mhd::IV2, k, j, i) = normalization * y;
          primitive(mhd::IV3, k, j, i) = normalization * z;
        }
        for (int axis = 0; axis < 3; ++axis)
          bcell(axis, k, j, i) = 0.0;
        divb(0, k, j, i) = 0.0;
        flags(0, k, j, i) = 0.0;
        for (int component = 0; component < 3; ++component)
          recovery(component, k, j, i) = 0.0;
      });

  nr::EnforceAlgebraicConstraintsBlockTask(data.get());
  nr::Z4cToADMFieldsBlockTask(data.get());

  const int potential_ni = interior_ib.e - interior_ib.s + 2;
  const int potential_nj = interior_jb.e - interior_jb.s + 2;
  const int potential_nk = interior_kb.e - interior_kb.s + 2;
  ParArray4DRaw<Real> potential("PANGU SYNC magnetized TOV vector potential", 3,
                                potential_nk, potential_nj, potential_ni);
  block->par_for(
      "PANGU SYNC magnetized TOV vector potential", interior_kb.s,
      interior_kb.e + 1, interior_jb.s, interior_jb.e + 1, interior_ib.s,
      interior_ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int ii = i - interior_ib.s;
        const int jj = j - interior_jb.s;
        const int kk = k - interior_kb.s;
        potential(0, kk, jj, ii) =
            magnetic_normalization *
            TOVVectorPotential<0>(profile, pressure_cut, magnetic_index,
                                  coordinates.Xc<X1DIR>(k, j, i),
                                  coordinates.Xf<X2DIR>(k, j, i),
                                  coordinates.Xf<X3DIR>(k, j, i));
        potential(1, kk, jj, ii) =
            magnetic_normalization *
            TOVVectorPotential<1>(profile, pressure_cut, magnetic_index,
                                  coordinates.Xf<X1DIR>(k, j, i),
                                  coordinates.Xc<X2DIR>(k, j, i),
                                  coordinates.Xf<X3DIR>(k, j, i));
        potential(2, kk, jj, ii) = 0.0;
      });
  block->par_for(
      "PANGU SYNC magnetized TOV B1", interior_kb.s, interior_kb.e,
      interior_jb.s, interior_jb.e, interior_ib.s, interior_ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int ii = i - interior_ib.s;
        const int jj = j - interior_jb.s;
        const int kk = k - interior_kb.s;
        bface(0, 0, 0, 0, k, j, i) =
            (potential(2, kk, jj + 1, ii) - potential(2, kk, jj, ii)) /
                coordinates.Dxc<X2DIR>(k, j, i) -
            (potential(1, kk + 1, jj, ii) - potential(1, kk, jj, ii)) /
                coordinates.Dxc<X3DIR>(k, j, i);
      });
  block->par_for(
      "PANGU SYNC magnetized TOV B2", interior_kb.s, interior_kb.e,
      interior_jb.s, interior_jb.e + 1, interior_ib.s, interior_ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int ii = i - interior_ib.s;
        const int jj = j - interior_jb.s;
        const int kk = k - interior_kb.s;
        bface(1, 0, 0, 0, k, j, i) =
            (potential(0, kk + 1, jj, ii) - potential(0, kk, jj, ii)) /
                coordinates.Dxc<X3DIR>(k, j, i) -
            (potential(2, kk, jj, ii + 1) - potential(2, kk, jj, ii)) /
                coordinates.Dxc<X1DIR>(k, j, i);
      });
  block->par_for(
      "PANGU SYNC magnetized TOV B3", interior_kb.s, interior_kb.e + 1,
      interior_jb.s, interior_jb.e, interior_ib.s, interior_ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int ii = i - interior_ib.s;
        const int jj = j - interior_jb.s;
        const int kk = k - interior_kb.s;
        bface(2, 0, 0, 0, k, j, i) =
            (potential(1, kk, jj, ii + 1) - potential(1, kk, jj, ii)) /
                coordinates.Dxc<X1DIR>(k, j, i) -
            (potential(0, kk, jj + 1, ii) - potential(0, kk, jj, ii)) /
                coordinates.Dxc<X2DIR>(k, j, i);
      });
  block->par_for(
      "PANGU SYNC magnetized TOV cell magnetic field", interior_kb.s,
      interior_kb.e, interior_jb.s, interior_jb.e, interior_ib.s, interior_ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        bcell(0, k, j, i) =
            0.5 * (bface(0, 0, 0, 0, k, j, i) + bface(0, 0, 0, 0, k, j, i + 1));
        bcell(1, k, j, i) =
            0.5 * (bface(1, 0, 0, 0, k, j, i) + bface(1, 0, 0, 0, k, j + 1, i));
        bcell(2, k, j, i) =
            0.5 * (bface(2, 0, 0, 0, k, j, i) + bface(2, 0, 0, 0, k + 1, j, i));
      });
  nr::SyncGRMHDPrimitiveToConservedBlock(data.get());
  nr::BuildStressEnergyBlockTask(data.get(), 0.0);
  nr::Z4cToADMBlockTask(data.get());
}

KOKKOS_INLINE_FUNCTION int ReflectionSign(const int component,
                                          const int direction) {
  // A normal-indexed component of a spatial vector changes sign at a
  // reflecting face.  For a covariant spatial two-tensor, exactly one normal
  // index changes sign; the normal-normal component is even.  The Z4c
  // conformal metric and A tensor use the same Cartesian tensor parity, while
  // chi, Khat, Theta, and alpha are scalars.
  if (component >= nr::Index(nr::Z4cComponent::gxx) &&
      component <= nr::Index(nr::Z4cComponent::gzz)) {
    const int spatial = component - nr::Index(nr::Z4cComponent::gxx);
    int first = 0;
    int second = 0;
    for (int candidate_first = 0; candidate_first < 3; ++candidate_first) {
      for (int candidate_second = candidate_first; candidate_second < 3;
           ++candidate_second) {
        if (nr::SpatialSymmetricComponent(candidate_first, candidate_second) ==
            spatial) {
          first = candidate_first;
          second = candidate_second;
        }
      }
    }
    return ((first == direction) ^ (second == direction)) ? -1 : 1;
  }
  if (component >= nr::Index(nr::Z4cComponent::axx) &&
      component <= nr::Index(nr::Z4cComponent::azz)) {
    const int spatial = component - nr::Index(nr::Z4cComponent::axx);
    int first = 0;
    int second = 0;
    for (int candidate_first = 0; candidate_first < 3; ++candidate_first) {
      for (int candidate_second = candidate_first; candidate_second < 3;
           ++candidate_second) {
        if (nr::SpatialSymmetricComponent(candidate_first, candidate_second) ==
            spatial) {
          first = candidate_first;
          second = candidate_second;
        }
      }
    }
    return ((first == direction) ^ (second == direction)) ? -1 : 1;
  }
  if (component >= nr::Index(nr::Z4cComponent::gamx) &&
      component <= nr::Index(nr::Z4cComponent::gamz))
    return component - nr::Index(nr::Z4cComponent::gamx) == direction ? -1 : 1;
  if (component >= nr::Index(nr::Z4cComponent::betax) &&
      component <= nr::Index(nr::Z4cComponent::betaz))
    return component - nr::Index(nr::Z4cComponent::betax) == direction ? -1 : 1;
  return 1;
}

template <int Direction, bool Inner>
void NumericalRelativityReflectingBoundary(
    std::shared_ptr<MeshBlockData<Real>> &data, const bool coarse) {
  auto block = data->GetBlockPointer();
  const auto &bounds = coarse ? block->c_cellbounds : block->cellbounds;
  const auto ib = bounds.GetBoundsI(IndexDomain::interior);
  const auto jb = bounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = bounds.GetBoundsK(IndexDomain::interior);
  const auto ei = bounds.GetBoundsI(IndexDomain::entire);
  const auto ej = bounds.GetBoundsJ(IndexDomain::entire);
  const auto ek = bounds.GetBoundsK(IndexDomain::entire);
  const auto z4c =
      data->PackVariables(std::vector<std::string>{"nr.z4c"}, coarse);
  int il = ei.s;
  int iu = ei.e;
  int jl = ej.s;
  int ju = ej.e;
  int kl = ek.s;
  int ku = ek.e;
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
  block->par_for(
      "PANGU NR tensor-parity reflecting boundary", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        constexpr int offset = Inner ? -1 : 1;
        const int reflected_k = Direction == 2 ? 2 * source + offset - k : k;
        const int reflected_j = Direction == 1 ? 2 * source + offset - j : j;
        const int reflected_i = Direction == 0 ? 2 * source + offset - i : i;
        for (int component = 0; component < nr::kZ4cComponents; ++component)
          z4c(component, k, j, i) =
              static_cast<Real>(ReflectionSign(component, Direction)) *
              z4c(component, reflected_k, reflected_j, reflected_i);
      });
}

template <int Direction, bool Inner>
void NumericalRelativityExtrapolationBoundary(
    std::shared_ptr<MeshBlockData<Real>> &data, const bool coarse) {
  // AthenaK applies the fluid/magnetic outflow operation and the Z4c
  // extrapolation on the same physical face.  Parthenon associates one user
  // callback with that face, so compose the two operations here: first fill
  // every communicable fluid/face field with its stock outflow rule, then
  // overwrite nr.z4c with the AthenaK polynomial extrapolation below.  This
  // also makes face-centred CT data valid before an initial AMR prolongation.
  if constexpr (Direction == 0 && Inner)
    parthenon::BoundaryFunction::OutflowInnerX1(data, coarse);
  else if constexpr (Direction == 0)
    parthenon::BoundaryFunction::OutflowOuterX1(data, coarse);
  else if constexpr (Direction == 1 && Inner)
    parthenon::BoundaryFunction::OutflowInnerX2(data, coarse);
  else if constexpr (Direction == 1)
    parthenon::BoundaryFunction::OutflowOuterX2(data, coarse);
  else if constexpr (Inner)
    parthenon::BoundaryFunction::OutflowInnerX3(data, coarse);
  else
    parthenon::BoundaryFunction::OutflowOuterX3(data, coarse);

  auto block = data->GetBlockPointer();
  const int order = block->packages.Get("numerical_relativity")
                        ->Param<int>("boundary_extrapolation_order");
  if (order == 0)
    return;
  if (order == 2) {
    // Express AthenaK's default linear polynomial as Parthenon's equivalent
    // constant-derivative operation.  Keeping both parts in SparsePack avoids
    // invalidating an in-flight generic boundary pack during initial AMR.
    using parthenon::BoundaryFunction::BCSide;
    using parthenon::BoundaryFunction::BCType;
    using parthenon::BoundaryFunction::GenericBC;
    if constexpr (Direction == 0 && Inner)
      GenericBC<parthenon::X1DIR, BCSide::Inner, BCType::ConstantDeriv,
                boundary_names::z4c>(data, coarse);
    else if constexpr (Direction == 0)
      GenericBC<parthenon::X1DIR, BCSide::Outer, BCType::ConstantDeriv,
                boundary_names::z4c>(data, coarse);
    else if constexpr (Direction == 1 && Inner)
      GenericBC<parthenon::X2DIR, BCSide::Inner, BCType::ConstantDeriv,
                boundary_names::z4c>(data, coarse);
    else if constexpr (Direction == 1)
      GenericBC<parthenon::X2DIR, BCSide::Outer, BCType::ConstantDeriv,
                boundary_names::z4c>(data, coarse);
    else if constexpr (Inner)
      GenericBC<parthenon::X3DIR, BCSide::Inner, BCType::ConstantDeriv,
                boundary_names::z4c>(data, coarse);
    else
      GenericBC<parthenon::X3DIR, BCSide::Outer, BCType::ConstantDeriv,
                boundary_names::z4c>(data, coarse);
    return;
  }

  const auto &bounds = coarse ? block->c_cellbounds : block->cellbounds;
  const auto ib = bounds.GetBoundsI(IndexDomain::interior);
  const auto jb = bounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = bounds.GetBoundsK(IndexDomain::interior);
  const auto ei = bounds.GetBoundsI(IndexDomain::entire);
  const auto ej = bounds.GetBoundsJ(IndexDomain::entire);
  const auto ek = bounds.GetBoundsK(IndexDomain::entire);
  const auto z4c =
      data->PackVariables(std::vector<std::string>{"nr.z4c"}, coarse);
  int il = ei.s;
  int iu = ei.e;
  int jl = ej.s;
  int ju = ej.e;
  int kl = ek.s;
  int ku = ek.e;
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
  block->par_for(
      "PANGU NR extrapolation boundary", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int ghost_coordinate =
            Direction == 2 ? k : (Direction == 1 ? j : i);
        const int reflected_distance =
            Inner ? source - ghost_coordinate : ghost_coordinate - source;
        const int sk = Direction == 2 ? source : k;
        const int sj = Direction == 1 ? source : j;
        const int si = Direction == 0 ? source : i;
        const int inward = Inner ? 1 : -1;
        const int offk = Direction == 2 ? inward : 0;
        const int offj = Direction == 1 ? inward : 0;
        const int offi = Direction == 0 ? inward : 0;
        for (int component = 0; component < nr::kZ4cComponents; ++component) {
          const Real delta = reflected_distance;
          const Real f0 = z4c(component, sk, sj, si);
          if (order == 0) {
            z4c(component, k, j, i) = f0;
          } else if (order == 2) {
            const Real f1 = z4c(component, sk + offk, sj + offj, si + offi);
            z4c(component, k, j, i) = f0 + delta * (f0 - f1);
          } else if (order == 3) {
            const Real f1 = z4c(component, sk + offk, sj + offj, si + offi);
            const Real f2 =
                z4c(component, sk + 2 * offk, sj + 2 * offj, si + 2 * offi);
            z4c(component, k, j, i) =
                0.5 * (f0 * (1.0 + delta) * (2.0 + delta) +
                       delta * (f2 + delta * f2 - 2.0 * f1 * (2.0 + delta)));
          } else {
            const Real f1 = z4c(component, sk + offk, sj + offj, si + offi);
            const Real f2 =
                z4c(component, sk + 2 * offk, sj + 2 * offj, si + 2 * offi);
            const Real f3 =
                z4c(component, sk + 3 * offk, sj + 3 * offj, si + 3 * offi);
            z4c(component, k, j, i) =
                (-3.0 * f1 * delta * (2.0 + delta) * (3.0 + delta) +
                 f0 * (1.0 + delta) * (2.0 + delta) * (3.0 + delta) +
                 delta * (1.0 + delta) *
                     (-f3 * (2.0 + delta) + 3.0 * f2 * (3.0 + delta))) /
                6.0;
          }
        }
      });
}

} // namespace

void NumericalRelativityMinkowski(MeshBlock *block, ParameterInput *) {
  InitializeMinkowskiZ4cFields(block);
  auto data = block->meshblock_data.Get();
  FinalizeNumericalRelativityInitialData(data.get());
}

void SyncGRHDProblem(MeshBlock *block, ParameterInput *pin) {
  const auto nr_package = block->packages.Get("numerical_relativity");
  const auto hydro_package = block->packages.Get("hydro");
  const auto matter_source =
      nr_package->Param<std::string>("matter_source_name");
  PARTHENON_REQUIRE(
      matter_source == "grhd" || matter_source == "zero",
      "sync GRHD problems require numerical_relativity/matter_source=grhd; "
      "zero is accepted only as the explicit test-fluid reference path");
  PARTHENON_REQUIRE(hydro_package->Param<int>("physics_mode") ==
                        static_cast<int>(relativity::HydroMode::gr),
                    "sync GRHD problems require hydro/physics=gr");

  InitializeMinkowskiZ4cFields(block);
  auto data = block->meshblock_data.Get();
  // Establish the stage-zero ADM geometry before converting the fluid.  Once
  // the fluid is initialized, rebuild Tmunu and only then evaluate the
  // matter-corrected constraints and optional Weyl diagnostics.
  nr::Z4cToADMFieldsBlockTask(data.get());
  HydroProblem(block, pin);
  nr::BuildStressEnergyBlockTask(data.get(), 0.0);
  nr::Z4cToADMBlockTask(data.get());
}

void SyncGRMHDProblem(MeshBlock *block, ParameterInput *pin) {
  const auto nr_package = block->packages.Get("numerical_relativity");
  const auto mhd_package = block->packages.Get("mhd");
  const auto matter_source =
      nr_package->Param<std::string>("matter_source_name");
  PARTHENON_REQUIRE(
      matter_source == "grmhd" || matter_source == "zero",
      "sync GRMHD problems require numerical_relativity/matter_source=grmhd; "
      "zero is accepted only as the explicit test-fluid reference path");
  PARTHENON_REQUIRE(mhd_package->Param<int>("physics_mode") ==
                        static_cast<int>(relativity::HydroMode::gr),
                    "sync GRMHD problems require mhd/physics=gr");

  InitializeMinkowskiZ4cFields(block);
  auto data = block->meshblock_data.Get();
  nr::Z4cToADMFieldsBlockTask(data.get());

  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto divb = data->PackVariables(std::vector<std::string>{"mhd.divb"});
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto recovery = data->PackVariables(std::vector<std::string>{"mhd.recovery"});
  auto bface = data->Get("mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto interior_ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto interior_jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto interior_kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  const int ndim = block->pmy_mesh->ndim;
  const auto eos = pangu::eos::ReadRelativistic(*mhd_package);
  const Real amplitude = pin->GetOrAddReal("problem", "amp", 1.0e-6);
  const int wave_flag = pin->GetOrAddInteger("problem", "wave_flag", 0);
  PARTHENON_REQUIRE(
      wave_flag >= 0 && wave_flag <= 6,
      "sync GRMHD linear wave requires 0 <= problem/wave_flag <= 6");
  const bool along_x1 = pin->GetOrAddBoolean("problem", "along_x1", true);
  const bool along_x2 = pin->GetOrAddBoolean("problem", "along_x2", false);
  const bool along_x3 = pin->GetOrAddBoolean("problem", "along_x3", false);
  const int aligned_directions = static_cast<int>(along_x1) +
                                 static_cast<int>(along_x2) +
                                 static_cast<int>(along_x3);
  PARTHENON_REQUIRE(
      aligned_directions <= 1,
      "only one of problem/along_x1, along_x2, and along_x3 may be true");
  PARTHENON_REQUIRE(!along_x2 || ndim >= 2,
                    "problem/along_x2 requires a multidimensional mesh");
  PARTHENON_REQUIRE(!along_x3 || ndim >= 3,
                    "problem/along_x3 requires a three-dimensional mesh");
  const auto wave =
      relativistic_mhd_wave::Build(pin->GetOrAddReal("problem", "dens", 4.0),
                                   pin->GetOrAddReal("problem", "pgas", 1.0),
                                   pin->GetOrAddReal("problem", "vx0", -0.1),
                                   pin->GetOrAddReal("problem", "vy0", 0.3),
                                   pin->GetOrAddReal("problem", "vz0", -0.05),
                                   pin->GetOrAddReal("problem", "bx0", 2.5),
                                   pin->GetOrAddReal("problem", "by0", 1.8),
                                   pin->GetOrAddReal("problem", "bz0", -1.2),
                                   eos.gamma, amplitude, wave_flag);
  const Real x1_length = pin->GetReal("parthenon/mesh", "x1max") -
                         pin->GetReal("parthenon/mesh", "x1min");
  const Real x2_length = pin->GetReal("parthenon/mesh", "x2max") -
                         pin->GetReal("parthenon/mesh", "x2min");
  const Real x3_length = pin->GetReal("parthenon/mesh", "x3max") -
                         pin->GetReal("parthenon/mesh", "x3min");
  const auto wave_geometry = relativistic_mhd_wave::BuildGeometry(
      x1_length, x2_length, x3_length, along_x1, along_x2, along_x3);
  const Real wave_number = wave_geometry.wave_number;
  if (block->gid == 0)
    std::cout << std::setprecision(17)
              << "SYNC-2 GRMHD eigenmode: wave_flag=" << wave_flag
              << " speed=" << wave.wave_speed
              << " wavelength=" << wave_geometry.wavelength
              << " amplitude=" << amplitude << '\n';
  block->par_for(
      "PANGU sync GRMHD primitive initialization", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x1 = coordinates.Xc<X1DIR>(k, j, i);
        const Real x2 = coordinates.Xc<X2DIR>(k, j, i);
        const Real x3 = coordinates.Xc<X3DIR>(k, j, i);
        const Real sinusoid =
            sin(wave_number * relativistic_mhd_wave::WaveCoordinate(
                                  x1, x2, x3, wave_geometry));
        const Real parallel =
            wave.four_velocity[1] +
            amplitude * sinusoid * wave.delta_four_velocity[1];
        const Real transverse_y =
            wave.four_velocity[2] +
            amplitude * sinusoid * wave.delta_four_velocity[2];
        const Real transverse_z =
            wave.four_velocity[3] +
            amplitude * sinusoid * wave.delta_four_velocity[3];
        primitive(mhd::IDN, k, j, i) =
            wave.density + amplitude * sinusoid * wave.delta_density;
        primitive(mhd::IV1, k, j, i) =
            parallel * wave_geometry.cos_a2 * wave_geometry.cos_a3 -
            transverse_y * wave_geometry.sin_a3 -
            transverse_z * wave_geometry.sin_a2 * wave_geometry.cos_a3;
        primitive(mhd::IV2, k, j, i) =
            parallel * wave_geometry.cos_a2 * wave_geometry.sin_a3 +
            transverse_y * wave_geometry.cos_a3 -
            transverse_z * wave_geometry.sin_a2 * wave_geometry.sin_a3;
        primitive(mhd::IV3, k, j, i) = parallel * wave_geometry.sin_a2 +
                                       transverse_z * wave_geometry.cos_a2;
        primitive(mhd::IPR, k, j, i) = eos.InternalEnergyDensity(
            wave.pressure + amplitude * sinusoid * wave.delta_pressure);
        divb(0, k, j, i) = 0.0;
        flags(0, k, j, i) = 0.0;
        for (int component = 0; component < 3; ++component)
          recovery(component, k, j, i) = 0.0;
      });
  if (along_x1) {
    // Keep the one-dimensional expression tree unchanged: this is the strict
    // roundoff-level PANGU/AthenaK oracle used by the multi-time gate.
    block->par_for(
        "PANGU sync GRMHD B1 initialization", kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          bface(0, 0, 0, 0, k, j, i) = wave.magnetic[0];
        });
    block->par_for(
        "PANGU sync GRMHD B2 initialization", kb.s, kb.e, jb.s,
        jb.e + (ndim >= 2), ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real left = coordinates.Xf<X1DIR>(k, j, i);
          const Real right = coordinates.Xf<X1DIR>(k, j, i + 1);
          const Real potential_left = -wave.magnetic[1] * left +
                                      wave.delta_transverse_magnetic[0] /
                                          wave_number * cos(wave_number * left);
          const Real potential_right =
              -wave.magnetic[1] * right + wave.delta_transverse_magnetic[0] /
                                              wave_number *
                                              cos(wave_number * right);
          bface(1, 0, 0, 0, k, j, i) =
              -(potential_right - potential_left) / (right - left);
        });
    block->par_for(
        "PANGU sync GRMHD B3 initialization", kb.s, kb.e + (ndim >= 3), jb.s,
        jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real left = coordinates.Xf<X1DIR>(k, j, i);
          const Real right = coordinates.Xf<X1DIR>(k, j, i + 1);
          const Real potential_left = wave.magnetic[2] * left -
                                      wave.delta_transverse_magnetic[1] /
                                          wave_number * cos(wave_number * left);
          const Real potential_right =
              wave.magnetic[2] * right - wave.delta_transverse_magnetic[1] /
                                             wave_number *
                                             cos(wave_number * right);
          bface(2, 0, 0, 0, k, j, i) =
              (potential_right - potential_left) / (right - left);
        });
  } else {
    PARTHENON_REQUIRE(ndim == 3,
                      "the diagonal sync GRMHD eigenmode requires a 3D mesh");
    int a1_finer_mask = 0;
    int a2_finer_mask = 0;
    int a3_finer_mask = 0;
    for (const auto &neighbor : block->GetNeighbors()) {
      if (neighbor.loc.level() <= block->loc.level())
        continue;
      const int ox1 = neighbor.offsets(X1DIR);
      const int ox2 = neighbor.offsets(X2DIR);
      const int ox3 = neighbor.offsets(X3DIR);
      if (ox1 == 0)
        a1_finer_mask |= 1 << ((ox2 + 1) * 3 + (ox3 + 1));
      if (ox2 == 0)
        a2_finer_mask |= 1 << ((ox1 + 1) * 3 + (ox3 + 1));
      if (ox3 == 0)
        a3_finer_mask |= 1 << ((ox1 + 1) * 3 + (ox2 + 1));
    }
    const int potential_ni = ib.e - ib.s + 2;
    const int potential_nj = jb.e - jb.s + 2;
    const int potential_nk = kb.e - kb.s + 2;
    ParArray4DRaw<Real> potential("PANGU sync GRMHD vector potential", 3,
                                  potential_nk, potential_nj, potential_ni);
    block->par_for(
        "PANGU sync GRMHD vector potential", kb.s, kb.e + 1, jb.s, jb.e + 1,
        ib.s, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const int ii = i - ib.s;
          const int jj = j - jb.s;
          const int kk = k - kb.s;
          const Real x1_center = coordinates.Xc<X1DIR>(k, j, i);
          const Real x2_center = coordinates.Xc<X2DIR>(k, j, i);
          const Real x3_center = coordinates.Xc<X3DIR>(k, j, i);
          const Real x1_face = coordinates.Xf<X1DIR>(k, j, i);
          const Real x2_face = coordinates.Xf<X2DIR>(k, j, i);
          const Real x3_face = coordinates.Xf<X3DIR>(k, j, i);
          const Real dx1 = coordinates.Dxc<X1DIR>(k, j, i);
          const Real dx2 = coordinates.Dxc<X2DIR>(k, j, i);
          const Real dx3 = coordinates.Dxc<X3DIR>(k, j, i);
          Real a1 = relativistic_mhd_wave::VectorPotential<0>(
              x1_center, x2_face, x3_face, wave, wave_geometry);
          Real a2 = relativistic_mhd_wave::VectorPotential<1>(
              x1_face, x2_center, x3_face, wave, wave_geometry);
          Real a3 = relativistic_mhd_wave::VectorPotential<2>(
              x1_face, x2_face, x3_center, wave, wave_geometry);

          // Match the integrated edge potential on a coarse edge to the sum
          // of its two fine children.  The discrete curl then gives exactly
          // the same magnetic flux on both sides of every coarse/fine face.
          const bool a1_correct =
              ((a1_finer_mask & (1 << 1)) && j == interior_jb.s) ||
              ((a1_finer_mask & (1 << 7)) && j == interior_jb.e + 1) ||
              ((a1_finer_mask & (1 << 3)) && k == interior_kb.s) ||
              ((a1_finer_mask & (1 << 5)) && k == interior_kb.e + 1) ||
              ((a1_finer_mask & (1 << 0)) && j == interior_jb.s &&
               k == interior_kb.s) ||
              ((a1_finer_mask & (1 << 2)) && j == interior_jb.s &&
               k == interior_kb.e + 1) ||
              ((a1_finer_mask & (1 << 6)) && j == interior_jb.e + 1 &&
               k == interior_kb.s) ||
              ((a1_finer_mask & (1 << 8)) && j == interior_jb.e + 1 &&
               k == interior_kb.e + 1);
          if (a1_correct)
            a1 = 0.5 * (relativistic_mhd_wave::VectorPotential<0>(
                            x1_center - 0.25 * dx1, x2_face, x3_face, wave,
                            wave_geometry) +
                        relativistic_mhd_wave::VectorPotential<0>(
                            x1_center + 0.25 * dx1, x2_face, x3_face, wave,
                            wave_geometry));

          const bool a2_correct =
              ((a2_finer_mask & (1 << 1)) && i == interior_ib.s) ||
              ((a2_finer_mask & (1 << 7)) && i == interior_ib.e + 1) ||
              ((a2_finer_mask & (1 << 3)) && k == interior_kb.s) ||
              ((a2_finer_mask & (1 << 5)) && k == interior_kb.e + 1) ||
              ((a2_finer_mask & (1 << 0)) && i == interior_ib.s &&
               k == interior_kb.s) ||
              ((a2_finer_mask & (1 << 2)) && i == interior_ib.s &&
               k == interior_kb.e + 1) ||
              ((a2_finer_mask & (1 << 6)) && i == interior_ib.e + 1 &&
               k == interior_kb.s) ||
              ((a2_finer_mask & (1 << 8)) && i == interior_ib.e + 1 &&
               k == interior_kb.e + 1);
          if (a2_correct)
            a2 = 0.5 * (relativistic_mhd_wave::VectorPotential<1>(
                            x1_face, x2_center - 0.25 * dx2, x3_face, wave,
                            wave_geometry) +
                        relativistic_mhd_wave::VectorPotential<1>(
                            x1_face, x2_center + 0.25 * dx2, x3_face, wave,
                            wave_geometry));

          const bool a3_correct =
              ((a3_finer_mask & (1 << 1)) && i == interior_ib.s) ||
              ((a3_finer_mask & (1 << 7)) && i == interior_ib.e + 1) ||
              ((a3_finer_mask & (1 << 3)) && j == interior_jb.s) ||
              ((a3_finer_mask & (1 << 5)) && j == interior_jb.e + 1) ||
              ((a3_finer_mask & (1 << 0)) && i == interior_ib.s &&
               j == interior_jb.s) ||
              ((a3_finer_mask & (1 << 2)) && i == interior_ib.s &&
               j == interior_jb.e + 1) ||
              ((a3_finer_mask & (1 << 6)) && i == interior_ib.e + 1 &&
               j == interior_jb.s) ||
              ((a3_finer_mask & (1 << 8)) && i == interior_ib.e + 1 &&
               j == interior_jb.e + 1);
          if (a3_correct)
            a3 = 0.5 * (relativistic_mhd_wave::VectorPotential<2>(
                            x1_face, x2_face, x3_center - 0.25 * dx3, wave,
                            wave_geometry) +
                        relativistic_mhd_wave::VectorPotential<2>(
                            x1_face, x2_face, x3_center + 0.25 * dx3, wave,
                            wave_geometry));

          potential(0, kk, jj, ii) = a1;
          potential(1, kk, jj, ii) = a2;
          potential(2, kk, jj, ii) = a3;
        });
    block->par_for(
        "PANGU sync GRMHD diagonal B1 initialization", kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const int ii = i - ib.s;
          const int jj = j - jb.s;
          const int kk = k - kb.s;
          bface(0, 0, 0, 0, k, j, i) =
              (potential(2, kk, jj + 1, ii) - potential(2, kk, jj, ii)) /
                  coordinates.Dxc<X2DIR>(k, j, i) -
              (potential(1, kk + 1, jj, ii) - potential(1, kk, jj, ii)) /
                  coordinates.Dxc<X3DIR>(k, j, i);
        });
    block->par_for(
        "PANGU sync GRMHD diagonal B2 initialization", kb.s, kb.e, jb.s,
        jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const int ii = i - ib.s;
          const int jj = j - jb.s;
          const int kk = k - kb.s;
          bface(1, 0, 0, 0, k, j, i) =
              (potential(0, kk + 1, jj, ii) - potential(0, kk, jj, ii)) /
                  coordinates.Dxc<X3DIR>(k, j, i) -
              (potential(2, kk, jj, ii + 1) - potential(2, kk, jj, ii)) /
                  coordinates.Dxc<X1DIR>(k, j, i);
        });
    block->par_for(
        "PANGU sync GRMHD diagonal B3 initialization", kb.s, kb.e + 1, jb.s,
        jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const int ii = i - ib.s;
          const int jj = j - jb.s;
          const int kk = k - kb.s;
          bface(2, 0, 0, 0, k, j, i) =
              (potential(1, kk, jj, ii + 1) - potential(1, kk, jj, ii)) /
                  coordinates.Dxc<X1DIR>(k, j, i) -
              (potential(0, kk, jj + 1, ii) - potential(0, kk, jj, ii)) /
                  coordinates.Dxc<X2DIR>(k, j, i);
        });
  }
  block->par_for(
      "PANGU sync GRMHD cell magnetic initialization", kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        bcell(0, k, j, i) =
            0.5 * (bface(0, 0, 0, 0, k, j, i) + bface(0, 0, 0, 0, k, j, i + 1));
        bcell(1, k, j, i) = 0.5 * (bface(1, 0, 0, 0, k, j, i) +
                                   bface(1, 0, 0, 0, k, j + (ndim >= 2), i));
        bcell(2, k, j, i) = 0.5 * (bface(2, 0, 0, 0, k, j, i) +
                                   bface(2, 0, 0, 0, k + (ndim >= 3), j, i));
      });
  nr::SyncGRMHDPrimitiveToConservedBlock(data.get());
  nr::BuildStressEnergyBlockTask(data.get(), 0.0);
  nr::Z4cToADMBlockTask(data.get());
}

void SyncGRHDTOV(MeshBlock *block, ParameterInput *pin) {
  const auto nr_package = block->packages.Get("numerical_relativity");
  const auto hydro_package = block->packages.Get("hydro");
  PARTHENON_REQUIRE(
      nr_package->Param<std::string>("matter_source_name") == "grhd",
      "sync_grhd_tov requires numerical_relativity/matter_source=grhd");
  PARTHENON_REQUIRE(hydro_package->Param<int>("physics_mode") ==
                        static_cast<int>(relativity::HydroMode::gr),
                    "sync_grhd_tov requires hydro/physics=gr");
  const int order = nr_package->Param<int>("finite_difference_order");
  if (order == 2)
    InitializeSyncGRHDTOV<2>(block, pin);
  else if (order == 4)
    InitializeSyncGRHDTOV<4>(block, pin);
  else
    InitializeSyncGRHDTOV<6>(block, pin);
}

void SyncGRMHDTOV(MeshBlock *block, ParameterInput *pin) {
  const auto nr_package = block->packages.Get("numerical_relativity");
  const auto mhd_package = block->packages.Get("mhd");
  PARTHENON_REQUIRE(
      nr_package->Param<std::string>("matter_source_name") == "grmhd",
      "sync_grmhd_tov requires numerical_relativity/matter_source=grmhd");
  PARTHENON_REQUIRE(mhd_package->Param<int>("physics_mode") ==
                        static_cast<int>(relativity::HydroMode::gr),
                    "sync_grmhd_tov requires mhd/physics=gr");
  const int order = nr_package->Param<int>("finite_difference_order");
  if (order == 2)
    InitializeSyncGRMHDTOV<2>(block, pin);
  else if (order == 4)
    InitializeSyncGRMHDTOV<4>(block, pin);
  else
    InitializeSyncGRMHDTOV<6>(block, pin);
}

void NumericalRelativitySchwarzschild(MeshBlock *block, ParameterInput *pin) {
  const Real mask_radius =
      pin->GetOrAddReal("problem", "constraint_mask_radius", 0.0);
  PARTHENON_REQUIRE(
      mask_radius >= 0.0,
      "NR Schwarzschild constraint_mask_radius must be nonnegative");
  InitializeConstraintMask(block, mask_radius);
  auto data = block->meshblock_data.Get();
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coordinates = block->coords;
  const Real mass = pin->GetOrAddReal("problem", "mass", 1.0);
  const auto options = block->packages.Get("numerical_relativity")
                           ->Param<nr::Z4cOptions>("z4c_options");
  PARTHENON_REQUIRE(mass >= 0.0, "NR Schwarzschild mass must be nonnegative");
  block->par_for(
      "PANGU NR Schwarzschild isotropic initial data", kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        nr::SchwarzschildIsotropicData initial{};
        const bool valid = nr::BuildSchwarzschildIsotropic(
            coordinates.Xc<X1DIR>(k, j, i), coordinates.Xc<X2DIR>(k, j, i),
            coordinates.Xc<X3DIR>(k, j, i), mass, options.chi_psi_power,
            initial);
        for (int component = 0; component < nr::kZ4cComponents; ++component)
          z4c(component, k, j, i) = 0.0;
        if (valid) {
          const Real chi = pow(nr::rhs::SpatialDeterminant(initial.adm.metric),
                               options.chi_psi_power / 12.0);
          z4c(nr::Index(nr::Z4cComponent::chi), k, j, i) = chi;
          z4c(nr::Index(nr::Z4cComponent::alpha), k, j, i) = initial.adm.lapse;
          for (int component = 0; component < 6; ++component)
            z4c(nr::Index(nr::Z4cComponent::gxx) + component, k, j, i) =
                chi * initial.adm.metric[component];
        } else {
          z4c(nr::Index(nr::Z4cComponent::chi), k, j, i) = 1.0;
          z4c(nr::Index(nr::Z4cComponent::gxx), k, j, i) = 1.0;
          z4c(nr::Index(nr::Z4cComponent::gyy), k, j, i) = 1.0;
          z4c(nr::Index(nr::Z4cComponent::gzz), k, j, i) = 1.0;
          z4c(nr::Index(nr::Z4cComponent::alpha), k, j, i) = 1.0;
        }
      });

  const int order = block->packages.Get("numerical_relativity")
                        ->Param<int>("finite_difference_order");
  if (order == 2)
    InitializeGaugeWaveGamma<2>(block, z4c);
  else if (order == 4)
    InitializeGaugeWaveGamma<4>(block, z4c);
  else
    InitializeGaugeWaveGamma<6>(block, z4c);
  FinalizeNumericalRelativityInitialData(data.get());
}

void NumericalRelativitySchwarzschildKerrSchild(MeshBlock *block,
                                                ParameterInput *pin) {
  const Real mask_radius =
      pin->GetOrAddReal("problem", "constraint_mask_radius", 0.0);
  PARTHENON_REQUIRE(mask_radius >= 0.0,
                    "NR Schwarzschild Kerr-Schild constraint_mask_radius must "
                    "be nonnegative");
  InitializeConstraintMask(block, mask_radius);
  auto data = block->meshblock_data.Get();
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coordinates = block->coords;
  const Real mass = pin->GetOrAddReal("problem", "mass", 1.0);
  const auto options = block->packages.Get("numerical_relativity")
                           ->Param<nr::Z4cOptions>("z4c_options");
  PARTHENON_REQUIRE(mass >= 0.0,
                    "NR Schwarzschild Kerr-Schild mass must be nonnegative");
  block->par_for(
      "PANGU NR Schwarzschild Kerr-Schild initial data", kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        nr::SchwarzschildKerrSchildData initial{};
        const bool valid = nr::BuildSchwarzschildKerrSchild(
            coordinates.Xc<X1DIR>(k, j, i), coordinates.Xc<X2DIR>(k, j, i),
            coordinates.Xc<X3DIR>(k, j, i), mass, initial);
        nr::Z4cState converted{};
        const bool converted_ok =
            valid && nr::ConvertADMToZ4c(initial.adm, initial.derivatives,
                                         options.chi_psi_power, converted);
        for (int component = 0; component < nr::kZ4cComponents; ++component)
          z4c(component, k, j, i) =
              converted_ok ? converted.values[component] : 0.0;
        if (!converted_ok) {
          z4c(nr::Index(nr::Z4cComponent::chi), k, j, i) = 1.0;
          z4c(nr::Index(nr::Z4cComponent::gxx), k, j, i) = 1.0;
          z4c(nr::Index(nr::Z4cComponent::gyy), k, j, i) = 1.0;
          z4c(nr::Index(nr::Z4cComponent::gzz), k, j, i) = 1.0;
          z4c(nr::Index(nr::Z4cComponent::alpha), k, j, i) = 1.0;
        }
      });
  FinalizeNumericalRelativityInitialData(data.get());
}

namespace {

void InitializeNumericalRelativityBoostedPuncture(MeshBlock *block,
                                                  ParameterInput *pin,
                                                  const bool finalize_matter) {
  const Real mask_radius =
      pin->GetOrAddReal("problem", "constraint_mask_radius", 0.5);
  PARTHENON_REQUIRE(
      mask_radius >= 0.0,
      "NR boosted puncture constraint_mask_radius must be nonnegative");
  const Real mass = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.0);
  const Real center_x = pin->GetOrAddReal("problem", "punc_center_x1", 0.0);
  const Real center_y = pin->GetOrAddReal("problem", "punc_center_x2", 0.0);
  const Real center_z = pin->GetOrAddReal("problem", "punc_center_x3", 0.0);
  const Real velocity_x = pin->GetOrAddReal("problem", "punc_velocity_x1", 0.0);
  const Real velocity_y = pin->GetOrAddReal("problem", "punc_velocity_x2", 0.0);
  const Real velocity_z = pin->GetOrAddReal("problem", "punc_velocity_x3", 0.0);
  PARTHENON_REQUIRE(mass > 0.0,
                    "NR boosted puncture punc_ADM_mass must be positive");
  PARTHENON_REQUIRE(std::fabs(velocity_y) <= 1.0e-14 &&
                        std::fabs(velocity_z) <= 1.0e-14,
                    "NR boosted puncture currently supports only an x1 boost");
  PARTHENON_REQUIRE(
      velocity_x * velocity_x < 1.0,
      "NR boosted puncture velocity must have magnitude below one");
  InitializeConstraintMask(block, mask_radius);
  auto data = block->meshblock_data.Get();
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coordinates = block->coords;
  const auto options = block->packages.Get("numerical_relativity")
                           ->Param<nr::Z4cOptions>("z4c_options");
  block->par_for(
      "PANGU NR boosted-puncture initial data", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        nr::BoostedPunctureData initial{};
        const bool valid =
            nr::BuildBoostedPuncture(coordinates.Xc<X1DIR>(k, j, i) - center_x,
                                     coordinates.Xc<X2DIR>(k, j, i) - center_y,
                                     coordinates.Xc<X3DIR>(k, j, i) - center_z,
                                     mass, velocity_x, initial);
        for (int component = 0; component < nr::kZ4cComponents; ++component)
          z4c(component, k, j, i) = 0.0;
        if (!valid) {
          z4c(nr::Index(nr::Z4cComponent::chi), k, j, i) = 1.0;
          z4c(nr::Index(nr::Z4cComponent::gxx), k, j, i) = 1.0;
          z4c(nr::Index(nr::Z4cComponent::gyy), k, j, i) = 1.0;
          z4c(nr::Index(nr::Z4cComponent::gzz), k, j, i) = 1.0;
          z4c(nr::Index(nr::Z4cComponent::alpha), k, j, i) = 1.0;
          return;
        }

        const Real determinant =
            nr::rhs::SpatialDeterminant(initial.adm.metric);
        const Real chi = pow(determinant, options.chi_psi_power / 12.0);
        Real inverse_metric[6]{};
        nr::rhs::SpatialInverse(1.0 / determinant, initial.adm.metric,
                                inverse_metric);
        Real trace_k = 0.0;
        for (int first = 0; first < 3; ++first)
          for (int second = 0; second < 3; ++second)
            trace_k +=
                nr::rhs::SymmetricValue(inverse_metric, first, second) *
                nr::rhs::SymmetricValue(initial.adm.extrinsic, first, second);

        z4c(nr::Index(nr::Z4cComponent::chi), k, j, i) = chi;
        z4c(nr::Index(nr::Z4cComponent::khat), k, j, i) = trace_k;
        z4c(nr::Index(nr::Z4cComponent::theta), k, j, i) = initial.adm.theta;
        // AthenaK applies GaugePreCollapsedLapse after ADMToZ4c/Z4cToADM,
        // therefore the lapse is based on the conformal factor inferred from
        // the *full boosted spatial metric*.  Using the isotropic lapse stored
        // by BuildBoostedPuncture would omit the longitudinal boost factor.
        // For psi4 = chi^(4/chi_psi_power), this is alpha = psi4^(-1/2).
        z4c(nr::Index(nr::Z4cComponent::alpha), k, j, i) =
            pow(chi, -2.0 / options.chi_psi_power);
        for (int axis = 0; axis < 3; ++axis)
          z4c(nr::Index(nr::Z4cComponent::betax) + axis, k, j, i) =
              initial.adm.shift[axis];
        for (int component = 0; component < 6; ++component) {
          const Real conformal_metric = chi * initial.adm.metric[component];
          z4c(nr::Index(nr::Z4cComponent::gxx) + component, k, j, i) =
              conformal_metric;
          z4c(nr::Index(nr::Z4cComponent::axx) + component, k, j, i) =
              chi * (initial.adm.extrinsic[component] -
                     initial.adm.metric[component] * trace_k / 3.0);
        }
      });

  const int order = block->packages.Get("numerical_relativity")
                        ->Param<int>("finite_difference_order");
  if (order == 2)
    InitializeGaugeWaveGamma<2>(block, z4c);
  else if (order == 4)
    InitializeGaugeWaveGamma<4>(block, z4c);
  else
    InitializeGaugeWaveGamma<6>(block, z4c);
  nr::EnforceAlgebraicConstraintsBlockTask(data.get());
  // A coupled GRMHD problem must first expose the ADM fields so that its
  // primitive-to-conserved conversion can use the puncture geometry.  The
  // matter constraints cannot be evaluated until mhd.cons and nr.tmunu have
  // subsequently been initialized by the coupled problem generator.
  if (finalize_matter)
    FinalizeNumericalRelativityInitialData(data.get());
  else
    nr::Z4cToADMFieldsBlockTask(data.get());
}

struct NativePunctureData {
  std::vector<nr::puncture::Puncture> punctures;
  nr::puncture::HamiltonianSolution solution;
  std::unique_ptr<nr::puncture::SpectralInterpolator> interpolator;
};

NativePunctureData &GetNativePunctureData(ParameterInput *pin) {
  static NativePunctureData data;
  if (data.interpolator != nullptr)
    return data;

  const bool legacy = pin->DoesParameterExist("problem", "par_m_plus");
  const int count =
      legacy ? 2 : pin->GetOrAddInteger("problem", "puncture_count", 1);
  PARTHENON_REQUIRE(
      count >= 1 && count <= 2,
      "native puncture initial data currently supports one or two punctures");
  data.punctures.resize(count);
  bool tune_target_masses = false;
  std::vector<double> target_masses;
  Real legacy_adm_tolerance = 1.0e-10;
  if (legacy) {
    const Real separation = pin->GetOrAddReal("problem", "par_b", 3.257);
    const Real offset[3] = {
        pin->GetOrAddReal("problem", "center_offset1", 0.0),
        pin->GetOrAddReal("problem", "center_offset2", 0.0),
        pin->GetOrAddReal("problem", "center_offset3", 0.0)};
    const bool swap_xz = pin->GetOrAddBoolean("problem", "swap_xz", false);
    const char *labels[2] = {"plus", "minus"};
    for (int puncture = 0; puncture < 2; ++puncture) {
      auto &value = data.punctures[puncture];
      const std::string label = labels[puncture];
      value.mass = pin->GetOrAddReal("problem", "par_m_" + label, 0.483);
      value.center[0] =
          offset[0] +
          (swap_xz ? 0.0 : (puncture == 0 ? separation : -separation));
      value.center[1] = offset[1];
      value.center[2] =
          offset[2] +
          (swap_xz ? (puncture == 0 ? separation : -separation) : 0.0);
      for (int axis = 0; axis < 3; ++axis) {
        const std::string suffix = std::to_string(axis + 1);
        value.momentum[axis] =
            pin->GetOrAddReal("problem", "par_P_" + label + suffix, 0.0);
        value.spin[axis] =
            pin->GetOrAddReal("problem", "par_S_" + label + suffix, 0.0);
      }
    }
    tune_target_masses =
        !pin->GetOrAddBoolean("problem", "give_bare_mass", true);
    target_masses = {pin->GetOrAddReal("problem", "target_M_plus", 0.505),
                     pin->GetOrAddReal("problem", "target_M_minus", 0.505)};
    legacy_adm_tolerance = pin->GetOrAddReal("problem", "adm_tol", 1.0e-10);
    // Read no-longer-needed TwoPunctures grid/debug knobs so legacy decks are
    // warning-free. The native Cartesian compactification has no separate B
    // or phi resolution and needs no puncture regularization constants.
    pin->GetOrAddReal("problem", "TP_epsilon", 0.0);
    pin->GetOrAddReal("problem", "TP_Tiny", 0.0);
    pin->GetOrAddReal("problem", "TP_Extend_Radius", 0.0);
    pin->GetOrAddBoolean("problem", "do_residuum_debug_output", false);
    pin->GetOrAddInteger("problem", "grid_setup_method", 0);
    pin->GetOrAddBoolean("problem", "solve_momentum_constraint", false);
    if (data.punctures[1].mass == 0.0 && target_masses[1] == 0.0 &&
        data.punctures[1].momentum[0] == 0.0 &&
        data.punctures[1].momentum[1] == 0.0 &&
        data.punctures[1].momentum[2] == 0.0 &&
        data.punctures[1].spin[0] == 0.0 &&
        data.punctures[1].spin[1] == 0.0 &&
        data.punctures[1].spin[2] == 0.0) {
      data.punctures.resize(1);
      target_masses.resize(1);
    }
  } else {
    for (int puncture = 0; puncture < count; ++puncture) {
      auto &value = data.punctures[puncture];
      const std::string prefix = "puncture_" + std::to_string(puncture) + "_";
      value.mass = pin->GetOrAddReal("problem", prefix + "mass", 1.0);
      const char *axes[3] = {"x", "y", "z"};
      for (int axis = 0; axis < 3; ++axis) {
        value.center[axis] =
            pin->GetOrAddReal("problem", prefix + "center_" + axes[axis], 0.0);
        value.momentum[axis] = pin->GetOrAddReal(
            "problem", prefix + "momentum_" + axes[axis], 0.0);
        value.spin[axis] =
            pin->GetOrAddReal("problem", prefix + "spin_" + axes[axis], 0.0);
      }
    }
  }
  for (const auto &puncture : data.punctures)
    PARTHENON_REQUIRE(puncture.mass > 0.0,
                      "native puncture bare masses must be positive");

  nr::puncture::HamiltonianSolveOptions options{};
  if (legacy) {
    const int points_a = pin->GetOrAddInteger("problem", "npoints_A", 30);
    const int points_b = pin->GetOrAddInteger("problem", "npoints_B", 30);
    pin->GetOrAddInteger("problem", "npoints_phi", 16);
    options.grid.points = std::max(points_a, points_b);
  } else {
    options.grid.points =
        pin->GetOrAddInteger("problem", "puncture_spectral_points", 24);
  }
  options.grid.scale =
      pin->GetOrAddReal("problem", "puncture_spectral_scale", 2.0);
  options.nonlinear_tolerance =
      legacy ? pin->GetOrAddReal("problem", "Newton_tol", 1.0e-10)
             : pin->GetOrAddReal("problem", "puncture_nonlinear_tolerance",
                                 1.0e-10);
  options.linear_tolerance =
      legacy
          ? std::min(1.0e-11, 0.1 * options.nonlinear_tolerance)
          : pin->GetOrAddReal("problem", "puncture_linear_tolerance", 1.0e-8);
  options.maximum_newton_iterations =
      legacy ? pin->GetOrAddInteger("problem", "Newton_maxit", 5)
             : pin->GetOrAddInteger("problem",
                                    "puncture_maximum_newton_iterations", 12);
  options.maximum_linear_iterations =
      legacy ? 2000
             : pin->GetOrAddInteger("problem",
                                    "puncture_maximum_linear_iterations", 800);
  if (tune_target_masses) {
    data.solution = nr::puncture::SolveTargetADMMasses(
        data.punctures, target_masses, options, legacy_adm_tolerance, 24);
  } else {
    data.solution =
        nr::puncture::SolveHamiltonianConstraint(data.punctures, options);
  }
  PARTHENON_REQUIRE(
      data.solution.converged,
      "native puncture Hamiltonian-constraint solve did not converge");
  data.interpolator =
      std::make_unique<nr::puncture::SpectralInterpolator>(data.solution);
  if (Globals::my_rank == 0) {
    std::cout << "PANGU native puncture data ready: residual "
              << data.solution.initial_residual << " -> "
              << data.solution.final_residual << ", Newton "
              << data.solution.newton_iterations << ", linear "
              << data.solution.linear_iterations << ".\n";
  }
  return data;
}

void InitializeNativePunctureMask(
    MeshBlock *block, ParameterInput *pin,
    const std::vector<nr::puncture::Puncture> &punctures) {
  const Real radius =
      pin->GetOrAddReal("problem", "constraint_mask_radius", 0.75);
  PARTHENON_REQUIRE(radius >= 0.0,
                    "NR puncture constraint_mask_radius must be nonnegative");
  const int count = static_cast<int>(punctures.size());
  const Real x0 = punctures[0].center[0], y0 = punctures[0].center[1],
             z0 = punctures[0].center[2];
  const Real x1 = count > 1 ? punctures[1].center[0] : x0;
  const Real y1 = count > 1 ? punctures[1].center[1] : y0;
  const Real z1 = count > 1 ? punctures[1].center[2] : z0;
  auto data = block->meshblock_data.Get();
  const auto mask =
      data->PackVariables(std::vector<std::string>{"nr.constraint_mask"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coordinates = block->coords;
  const Real radius_squared = radius * radius;
  block->par_for(
      "PANGU NR native puncture constraint mask", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coordinates.Xc<X1DIR>(k, j, i);
        const Real y = coordinates.Xc<X2DIR>(k, j, i);
        const Real z = coordinates.Xc<X3DIR>(k, j, i);
        const Real d0 =
            (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0);
        const Real d1 =
            (x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1);
        mask(0, k, j, i) = radius > 0.0 && (d0 <= radius_squared ||
                                            (count > 1 && d1 <= radius_squared))
                               ? 0.0
                               : 1.0;
      });
}

} // namespace

void NumericalRelativityBoostedPuncture(MeshBlock *block, ParameterInput *pin) {
  InitializeNumericalRelativityBoostedPuncture(block, pin, true);
}

void NumericalRelativityPuncture(MeshBlock *block, ParameterInput *pin) {
  auto &puncture_data = GetNativePunctureData(pin);
  InitializeNativePunctureMask(block, pin, puncture_data.punctures);
  auto data = block->meshblock_data.Get();
  auto z4c_data = data->Get("nr.z4c").data;
  auto host_z4c = z4c_data.GetHostMirrorAndCopy();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coordinates = block->coords;
  const auto options = block->packages.Get("numerical_relativity")
                           ->Param<nr::Z4cOptions>("z4c_options");
  const Real lapse_exponent =
      pin->GetOrAddReal("problem", "initial_lapse_psi_exponent", -2.0);
  bool valid = true;
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        nr::ADMState adm{};
        const bool cell_valid = nr::puncture::BuildPunctureADM(
            coordinates.Xc<X1DIR>(k, j, i), coordinates.Xc<X2DIR>(k, j, i),
            coordinates.Xc<X3DIR>(k, j, i), puncture_data.punctures,
            *puncture_data.interpolator, adm);
        valid = valid && cell_valid;
        if (!cell_valid)
          continue;
        const Real determinant = nr::rhs::SpatialDeterminant(adm.metric);
        const Real chi = pow(determinant, options.chi_psi_power / 12.0);
        Real inverse_metric[6]{};
        nr::rhs::SpatialInverse(1.0 / determinant, adm.metric, inverse_metric);
        Real trace_k = 0.0;
        for (int first = 0; first < 3; ++first)
          for (int second = 0; second < 3; ++second)
            trace_k += nr::rhs::SymmetricValue(inverse_metric, first, second) *
                       nr::rhs::SymmetricValue(adm.extrinsic, first, second);
        for (int component = 0; component < nr::kZ4cComponents; ++component)
          host_z4c(component, k, j, i) = 0.0;
        host_z4c(nr::Index(nr::Z4cComponent::chi), k, j, i) = chi;
        host_z4c(nr::Index(nr::Z4cComponent::khat), k, j, i) = trace_k;
        host_z4c(nr::Index(nr::Z4cComponent::alpha), k, j, i) =
            pow(determinant, lapse_exponent / 12.0);
        for (int component = 0; component < 6; ++component) {
          host_z4c(nr::Index(nr::Z4cComponent::gxx) + component, k, j, i) =
              chi * adm.metric[component];
          host_z4c(nr::Index(nr::Z4cComponent::axx) + component, k, j, i) =
              chi * (adm.extrinsic[component] -
                     adm.metric[component] * trace_k / 3.0);
        }
      }
    }
  }
  PARTHENON_REQUIRE(valid,
                    "native puncture interpolation produced invalid ADM data");
  Kokkos::deep_copy(z4c_data, host_z4c);
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const int order = block->packages.Get("numerical_relativity")
                        ->Param<int>("finite_difference_order");
  if (order == 2)
    InitializeGaugeWaveGamma<2>(block, z4c);
  else if (order == 4)
    InitializeGaugeWaveGamma<4>(block, z4c);
  else
    InitializeGaugeWaveGamma<6>(block, z4c);
  nr::EnforceAlgebraicConstraintsBlockTask(data.get());
  FinalizeNumericalRelativityInitialData(data.get());
}

void SyncGRMHDBoostedPuncture(MeshBlock *block, ParameterInput *pin) {
  const auto nr_package = block->packages.Get("numerical_relativity");
  const auto mhd_package = block->packages.Get("mhd");
  PARTHENON_REQUIRE(nr_package->Param<std::string>("matter_source_name") ==
                        "grmhd",
                    "sync_grmhd_boosted_puncture requires matter_source=grmhd");
  PARTHENON_REQUIRE(mhd_package->Param<int>("physics_mode") ==
                        static_cast<int>(relativity::HydroMode::gr),
                    "sync_grmhd_boosted_puncture requires mhd/physics=gr");
  InitializeNumericalRelativityBoostedPuncture(block, pin, false);

  auto data = block->meshblock_data.Get();
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto divb = data->PackVariables(std::vector<std::string>{"mhd.divb"});
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto recovery = data->PackVariables(std::vector<std::string>{"mhd.recovery"});
  auto bface = data->Get("mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const int ndim = block->pmy_mesh->ndim;
  const auto eos = pangu::eos::ReadRelativistic(*mhd_package);
  const Real density =
      pin->GetOrAddReal("problem", "atmosphere_density",
                        mhd_package->Param<Real>("puncture_density"));
  const Real pressure =
      pin->GetOrAddReal("problem", "atmosphere_pressure",
                        mhd_package->Param<Real>("puncture_pressure"));
  const Real b1 = pin->GetOrAddReal("problem", "btilde1", 1.0e-8);
  const Real b2 = pin->GetOrAddReal("problem", "btilde2", 0.0);
  const Real b3 = pin->GetOrAddReal("problem", "btilde3", 0.0);
  PARTHENON_REQUIRE(density >= eos.density_floor &&
                        pressure >= eos.pressure_floor,
                    "boosted-puncture atmosphere must respect MHD floors");
  block->par_for(
      "PANGU sync GRMHD boosted-puncture atmosphere", kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        primitive(mhd::IDN, k, j, i) = density;
        primitive(mhd::IV1, k, j, i) = 0.0;
        primitive(mhd::IV2, k, j, i) = 0.0;
        primitive(mhd::IV3, k, j, i) = 0.0;
        primitive(mhd::IPR, k, j, i) = eos.InternalEnergyDensity(pressure);
        bcell(0, k, j, i) = b1;
        bcell(1, k, j, i) = b2;
        bcell(2, k, j, i) = b3;
        divb(0, k, j, i) = 0.0;
        flags(0, k, j, i) = 0.0;
        for (int component = 0; component < 3; ++component)
          recovery(component, k, j, i) = 0.0;
      });
  block->par_for(
      "PANGU sync GRMHD boosted-puncture B1", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        bface(0, 0, 0, 0, k, j, i) = b1;
      });
  block->par_for(
      "PANGU sync GRMHD boosted-puncture B2", kb.s, kb.e, jb.s,
      jb.e + (ndim >= 2), ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        bface(1, 0, 0, 0, k, j, i) = b2;
      });
  block->par_for(
      "PANGU sync GRMHD boosted-puncture B3", kb.s, kb.e + (ndim >= 3), jb.s,
      jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        bface(2, 0, 0, 0, k, j, i) = b3;
      });
  nr::SyncGRMHDPrimitiveToConservedBlock(data.get());
  nr::BuildStressEnergyBlockTask(data.get(), 0.0);
  nr::Z4cToADMBlockTask(data.get());
}

void NumericalRelativityLinearWave(MeshBlock *block, ParameterInput *pin) {
  InitializeConstraintMask(block);
  auto data = block->meshblock_data.Get();
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto &mesh_size = block->pmy_mesh->mesh_size;
  const Real x1_length = mesh_size.xmax(X1DIR) - mesh_size.xmin(X1DIR);
  const Real x2_length = mesh_size.xmax(X2DIR) - mesh_size.xmin(X2DIR);
  const Real x3_length = mesh_size.xmax(X3DIR) - mesh_size.xmin(X3DIR);
  const Real amplitude = pin->GetOrAddReal("problem", "amplitude", 1.0e-8);
  const Real kx1 = pin->GetOrAddReal("problem", "kx1", 1.0 / x1_length);
  const Real kx2 = pin->GetOrAddReal("problem", "kx2", 1.0 / x2_length);
  const Real kx3 = pin->GetOrAddReal("problem", "kx3", 1.0 / x3_length);
  const Real wave_number = sqrt(kx1 * kx1 + kx2 * kx2 + kx3 * kx3);
  PARTHENON_REQUIRE(wave_number > 0.0,
                    "NR linear wave requires a nonzero wavevector");
  const Real theta = atan2(sqrt(kx2 * kx2 + kx1 * kx1), kx3);
  const Real phi = atan2(kx1, kx2);
  const Real weights[6] = {-cos(theta) * cos(theta) * cos(2.0 * phi) -
                               cos(phi) * cos(phi) * sin(theta) * sin(theta),
                           -0.25 * (3.0 + cos(2.0 * theta)) * sin(2.0 * phi),
                           -cos(theta) * sin(theta) * sin(phi),
                           cos(theta) * cos(theta) * cos(2.0 * phi) -
                               sin(theta) * sin(theta) * sin(phi) * sin(phi),
                           cos(theta) * sin(theta) * cos(phi),
                           sin(theta) * sin(theta)};
  const auto coordinates = block->coords;
  block->par_for(
      "PANGU NR linear-wave initial data", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        for (int component = 0; component < nr::kZ4cComponents; ++component)
          z4c(component, k, j, i) = 0.0;
        const Real phase = 2.0 * M_PI *
                           (kx1 * coordinates.Xc<X1DIR>(k, j, i) +
                            kx2 * coordinates.Xc<X2DIR>(k, j, i) +
                            kx3 * coordinates.Xc<X3DIR>(k, j, i));
        const Real sine = sin(phase);
        const Real cosine = wave_number * M_PI * cos(phase);
        for (int component = 0; component < 6; ++component) {
          const bool diagonal =
              component == nr::SpatialSymmetricComponent(0, 0) ||
              component == nr::SpatialSymmetricComponent(1, 1) ||
              component == nr::SpatialSymmetricComponent(2, 2);
          z4c(nr::Index(nr::Z4cComponent::gxx) + component, k, j, i) =
              (diagonal ? 1.0 : 0.0) + weights[component] * amplitude * sine;
          z4c(nr::Index(nr::Z4cComponent::axx) + component, k, j, i) =
              weights[component] * amplitude * cosine;
        }
        z4c(nr::Index(nr::Z4cComponent::chi), k, j, i) = 1.0;
        z4c(nr::Index(nr::Z4cComponent::alpha), k, j, i) = 1.0;
      });
  FinalizeNumericalRelativityInitialData(data.get());
}

void NumericalRelativityGaugeWave(MeshBlock *block, ParameterInput *pin) {
  InitializeConstraintMask(block);
  auto data = block->meshblock_data.Get();
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coordinates = block->coords;
  const auto &mesh_size = block->pmy_mesh->mesh_size;
  const Real x1_min = mesh_size.xmin(X1DIR);
  const Real wavelength = mesh_size.xmax(X1DIR) - x1_min;
  const Real amplitude = pin->GetOrAddReal("problem", "amplitude", 1.0e-2);
  const auto options = block->packages.Get("numerical_relativity")
                           ->Param<nr::Z4cOptions>("z4c_options");
  PARTHENON_REQUIRE(amplitude > 0.0 && amplitude < 1.0,
                    "NR gauge-wave amplitude must be in (0,1)");
  block->par_for(
      "PANGU NR gauge-wave initial data", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        for (int component = 0; component < nr::kZ4cComponents; ++component)
          z4c(component, k, j, i) = 0.0;
        const Real x = coordinates.Xc<X1DIR>(k, j, i);
        const Real phase = 2.0 * M_PI * (x - x1_min) / wavelength;
        const Real h = amplitude * sin(phase);
        const Real physical_gxx = 1.0 - h;
        const Real dh_dt = -amplitude * (2.0 * M_PI / wavelength) * cos(phase);
        const Real alpha = sqrt(physical_gxx);
        const Real physical_kxx = 0.5 * dh_dt / alpha;
        const Real conformal_scale = pow(physical_gxx, -1.0 / 3.0);
        const Real khat = physical_kxx / physical_gxx;

        z4c(nr::Index(nr::Z4cComponent::chi), k, j, i) =
            pow(physical_gxx, options.chi_psi_power / 12.0);
        z4c(nr::Index(nr::Z4cComponent::gxx), k, j, i) =
            conformal_scale * physical_gxx;
        z4c(nr::Index(nr::Z4cComponent::gyy), k, j, i) = conformal_scale;
        z4c(nr::Index(nr::Z4cComponent::gzz), k, j, i) = conformal_scale;
        z4c(nr::Index(nr::Z4cComponent::khat), k, j, i) = khat;
        z4c(nr::Index(nr::Z4cComponent::axx), k, j, i) =
            conformal_scale * physical_kxx -
            khat * conformal_scale * physical_gxx / 3.0;
        z4c(nr::Index(nr::Z4cComponent::ayy), k, j, i) =
            -khat * conformal_scale / 3.0;
        z4c(nr::Index(nr::Z4cComponent::azz), k, j, i) =
            -khat * conformal_scale / 3.0;
        z4c(nr::Index(nr::Z4cComponent::alpha), k, j, i) = alpha;
      });

  const int order = block->packages.Get("numerical_relativity")
                        ->Param<int>("finite_difference_order");
  if (order == 2)
    InitializeGaugeWaveGamma<2>(block, z4c);
  else if (order == 4)
    InitializeGaugeWaveGamma<4>(block, z4c);
  else
    InitializeGaugeWaveGamma<6>(block, z4c);
  nr::EnforceAlgebraicConstraintsBlockTask(data.get());
  FinalizeNumericalRelativityInitialData(data.get());
}

void NumericalRelativityRobustStability(MeshBlock *block, ParameterInput *pin) {
  InitializeConstraintMask(block);
  auto data = block->meshblock_data.Get();
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto interior_k = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const int global_nx3 = block->pmy_mesh->mesh_size.nx(X3DIR);
  const Real rho = pin->GetOrAddReal("problem", "rho", 1.0);
  const Real amplitude = 1.0e-10 / (rho * rho);
  const auto options = block->packages.Get("numerical_relativity")
                           ->Param<nr::Z4cOptions>("z4c_options");
  PARTHENON_REQUIRE(rho > 0.0, "NR robust-stability rho must be positive");
  block->par_for(
      "PANGU NR deterministic robust-stability initial data", kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        for (int component = 0; component < nr::kZ4cComponents; ++component)
          z4c(component, k, j, i) = 0.0;
        int wrapped_plane = (k - interior_k.s) % global_nx3;
        if (wrapped_plane < 0)
          wrapped_plane += global_nx3;
        Real physical_metric[6]{};
        Real physical_k[6]{};
        for (int component = 0; component < 6; ++component) {
          const bool diagonal =
              component == nr::SpatialSymmetricComponent(0, 0) ||
              component == nr::SpatialSymmetricComponent(1, 1) ||
              component == nr::SpatialSymmetricComponent(2, 2);
          physical_metric[component] =
              (diagonal ? 1.0 : 0.0) +
              amplitude * DeterministicNoise(wrapped_plane, component);
          physical_k[component] =
              amplitude * DeterministicNoise(wrapped_plane, component + 6);
        }
        const Real determinant = nr::rhs::SpatialDeterminant(physical_metric);
        const Real conformal_scale = pow(determinant, -1.0 / 3.0);
        Real conformal_metric[6]{};
        Real conformal_k[6]{};
        for (int component = 0; component < 6; ++component) {
          conformal_metric[component] =
              conformal_scale * physical_metric[component];
          conformal_k[component] = conformal_scale * physical_k[component];
        }
        Real inverse[6]{};
        const Real conformal_determinant =
            nr::rhs::SpatialDeterminant(conformal_metric);
        nr::rhs::SpatialInverse(1.0 / conformal_determinant, conformal_metric,
                                inverse);
        Real khat = 0.0;
        for (int first = 0; first < 3; ++first) {
          for (int second = 0; second < 3; ++second) {
            const int component = nr::SpatialSymmetricComponent(first, second);
            khat += inverse[component] * conformal_k[component];
          }
        }
        z4c(nr::Index(nr::Z4cComponent::chi), k, j, i) =
            pow(determinant, options.chi_psi_power / 12.0);
        z4c(nr::Index(nr::Z4cComponent::khat), k, j, i) = khat;
        z4c(nr::Index(nr::Z4cComponent::alpha), k, j, i) = 1.0;
        for (int component = 0; component < 6; ++component) {
          z4c(nr::Index(nr::Z4cComponent::gxx) + component, k, j, i) =
              conformal_metric[component];
          z4c(nr::Index(nr::Z4cComponent::axx) + component, k, j, i) =
              conformal_k[component] - khat * conformal_metric[component] / 3.0;
        }
      });

  const int order = block->packages.Get("numerical_relativity")
                        ->Param<int>("finite_difference_order");
  if (order == 2)
    InitializeGaugeWaveGamma<2>(block, z4c);
  else if (order == 4)
    InitializeGaugeWaveGamma<4>(block, z4c);
  else
    InitializeGaugeWaveGamma<6>(block, z4c);
  nr::EnforceAlgebraicConstraintsBlockTask(data.get());
  FinalizeNumericalRelativityInitialData(data.get());
}

void RegisterNumericalRelativityReflectingBoundaries(
    ApplicationInput *app_input) {
  app_input->RegisterBoundaryCondition(
      BoundaryFace::inner_x1, "nr_reflecting",
      NumericalRelativityReflectingBoundary<0, true>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::outer_x1, "nr_reflecting",
      NumericalRelativityReflectingBoundary<0, false>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::inner_x2, "nr_reflecting",
      NumericalRelativityReflectingBoundary<1, true>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::outer_x2, "nr_reflecting",
      NumericalRelativityReflectingBoundary<1, false>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::inner_x3, "nr_reflecting",
      NumericalRelativityReflectingBoundary<2, true>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::outer_x3, "nr_reflecting",
      NumericalRelativityReflectingBoundary<2, false>);
}

void RegisterNumericalRelativityExtrapolationBoundaries(
    ApplicationInput *app_input) {
  app_input->RegisterBoundaryCondition(
      BoundaryFace::inner_x1, "nr_extrapolate",
      NumericalRelativityExtrapolationBoundary<0, true>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::outer_x1, "nr_extrapolate",
      NumericalRelativityExtrapolationBoundary<0, false>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::inner_x2, "nr_extrapolate",
      NumericalRelativityExtrapolationBoundary<1, true>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::outer_x2, "nr_extrapolate",
      NumericalRelativityExtrapolationBoundary<1, false>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::inner_x3, "nr_extrapolate",
      NumericalRelativityExtrapolationBoundary<2, true>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::outer_x3, "nr_extrapolate",
      NumericalRelativityExtrapolationBoundary<2, false>);
}

void RegisterNumericalRelativityOutflowBoundaries(ApplicationInput *app_input) {
  // AthenaK's Z4c `outflow` boundary uses linear extrapolation in the ghost
  // zones before applying its Sommerfeld RHS on the outermost active cells.
  // Keep that behavior under an explicit NR name so the generic Parthenon
  // outflow callback remains unchanged for hydrodynamics and MHD.
  app_input->RegisterBoundaryCondition(
      BoundaryFace::inner_x1, "nr_outflow",
      NumericalRelativityExtrapolationBoundary<0, true>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::outer_x1, "nr_outflow",
      NumericalRelativityExtrapolationBoundary<0, false>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::inner_x2, "nr_outflow",
      NumericalRelativityExtrapolationBoundary<1, true>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::outer_x2, "nr_outflow",
      NumericalRelativityExtrapolationBoundary<1, false>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::inner_x3, "nr_outflow",
      NumericalRelativityExtrapolationBoundary<2, true>);
  app_input->RegisterBoundaryCondition(
      BoundaryFace::outer_x3, "nr_outflow",
      NumericalRelativityExtrapolationBoundary<2, false>);
}

void NumericalRelativityAfterLoop(Mesh *mesh, ParameterInput *, SimTime &time) {
  const bool vacuum =
      mesh->packages.Get("numerical_relativity")->Param<bool>("vacuum");
  Real maximum = 0.0;
  for (const auto &block : mesh->block_list) {
    const auto data = block->meshblock_data.Get();
    const auto z4c = data->Get("nr.z4c").data.GetHostMirrorAndCopy();
    const auto adm = data->Get("nr.adm").data.GetHostMirrorAndCopy();
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          for (int component = 0; component < nr::kZ4cComponents; ++component) {
            const bool unit = component == nr::Index(nr::Z4cComponent::chi) ||
                              component == nr::Index(nr::Z4cComponent::gxx) ||
                              component == nr::Index(nr::Z4cComponent::gyy) ||
                              component == nr::Index(nr::Z4cComponent::gzz) ||
                              component == nr::Index(nr::Z4cComponent::alpha);
            maximum = std::max(maximum, std::abs(z4c(component, k, j, i) -
                                                 (unit ? 1.0 : 0.0)));
          }
          for (int component = 0; component < nr::kADMComponents; ++component) {
            const bool unit = component == nr::Index(nr::ADMComponent::gxx) ||
                              component == nr::Index(nr::ADMComponent::gyy) ||
                              component == nr::Index(nr::ADMComponent::gzz) ||
                              component == nr::Index(nr::ADMComponent::psi4);
            maximum = std::max(maximum, std::abs(adm(component, k, j, i) -
                                                 (unit ? 1.0 : 0.0)));
          }
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  Real global_maximum = maximum;
  PARTHENON_MPI_CHECK(MPI_Allreduce(&maximum, &global_maximum, 1,
                                    MPI_PARTHENON_REAL, MPI_MAX,
                                    MPI_COMM_WORLD));
  maximum = global_maximum;
#endif
  if (!vacuum) {
    PARTHENON_REQUIRE(
        std::isfinite(maximum),
        "SYNC-0 matter-source evolution produced a non-finite state");
    if (Globals::my_rank == 0) {
      std::cout << "SYNC-0 matter-source evolution PASS: cycles=" << time.ncycle
                << " finite state maximum=" << maximum << '\n';
    }
    return;
  }
  const Real tolerance = 128.0 * std::numeric_limits<Real>::epsilon();
  PARTHENON_REQUIRE(std::isfinite(maximum) && maximum <= tolerance,
                    "NR-0 Minkowski stage/boundary/ADM preservation failed");
  if (Globals::my_rank == 0) {
    std::cout << "NR-0 Minkowski evolution PASS: cycles=" << time.ncycle
              << " max error=" << maximum << '\n';
  }
}

void NumericalRelativityLinearWaveAfterLoop(Mesh *mesh, ParameterInput *pin,
                                            SimTime &time) {
  const Real amplitude = pin->GetReal("problem", "amplitude");
  const Real x1_length =
      mesh->mesh_size.xmax(X1DIR) - mesh->mesh_size.xmin(X1DIR);
  const Real x2_length =
      mesh->mesh_size.xmax(X2DIR) - mesh->mesh_size.xmin(X2DIR);
  const Real x3_length =
      mesh->mesh_size.xmax(X3DIR) - mesh->mesh_size.xmin(X3DIR);
  const Real kx1 = pin->GetOrAddReal("problem", "kx1", 1.0 / x1_length);
  const Real kx2 = pin->GetOrAddReal("problem", "kx2", 1.0 / x2_length);
  const Real kx3 = pin->GetOrAddReal("problem", "kx3", 1.0 / x3_length);
  const Real wave_number = sqrt(kx1 * kx1 + kx2 * kx2 + kx3 * kx3);
  const Real theta = atan2(sqrt(kx2 * kx2 + kx1 * kx1), kx3);
  const Real phi = atan2(kx1, kx2);
  const Real weights[6] = {-cos(theta) * cos(theta) * cos(2.0 * phi) -
                               cos(phi) * cos(phi) * sin(theta) * sin(theta),
                           -0.25 * (3.0 + cos(2.0 * theta)) * sin(2.0 * phi),
                           -cos(theta) * sin(theta) * sin(phi),
                           cos(theta) * cos(theta) * cos(2.0 * phi) -
                               sin(theta) * sin(theta) * sin(phi) * sin(phi),
                           cos(theta) * sin(theta) * cos(phi),
                           sin(theta) * sin(theta)};
  Real error_sum[6]{};
  Real maximum = 0.0;
  std::uint64_t cells = 0;
  for (const auto &block : mesh->block_list) {
    const auto data = block->meshblock_data.Get();
    const auto z4c = data->Get("nr.z4c").data.GetHostMirrorAndCopy();
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          const Real phase = 2.0 * M_PI *
                             (kx1 * block->coords.Xc<X1DIR>(k, j, i) +
                              kx2 * block->coords.Xc<X2DIR>(k, j, i) +
                              kx3 * block->coords.Xc<X3DIR>(k, j, i) -
                              wave_number * time.time);
          for (int component = 0; component < 6; ++component) {
            const bool diagonal =
                component == nr::SpatialSymmetricComponent(0, 0) ||
                component == nr::SpatialSymmetricComponent(1, 1) ||
                component == nr::SpatialSymmetricComponent(2, 2);
            const Real expected = (diagonal ? 1.0 : 0.0) +
                                  weights[component] * amplitude * sin(phase);
            const Real error = std::abs(
                z4c(nr::Index(nr::Z4cComponent::gxx) + component, k, j, i) -
                expected);
            error_sum[component] += error;
            maximum = std::max(maximum, error);
          }
          ++cells;
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  Real global_sum[6]{};
  Real global_maximum = maximum;
  std::uint64_t global_cells = cells;
  PARTHENON_MPI_CHECK(MPI_Allreduce(
      error_sum, global_sum, 6, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(&maximum, &global_maximum, 1,
                                    MPI_PARTHENON_REAL, MPI_MAX,
                                    MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(&cells, &global_cells, 1, MPI_UINT64_T,
                                    MPI_SUM, MPI_COMM_WORLD));
  std::copy(std::begin(global_sum), std::end(global_sum),
            std::begin(error_sum));
  maximum = global_maximum;
  cells = global_cells;
#endif
  Real rms = 0.0;
  for (const Real sum : error_sum) {
    const Real l1 = sum / static_cast<Real>(cells);
    rms += l1 * l1;
  }
  rms = sqrt(rms);
  PARTHENON_REQUIRE(std::isfinite(rms) && std::isfinite(maximum),
                    "NR linear-wave evolution produced a non-finite error");
  if (Globals::my_rank == 0)
    std::cout << "NR-2 linear wave: cycles=" << time.ncycle << " L1_RMS=" << rms
              << " Linf=" << maximum << '\n';
}

void NumericalRelativityGaugeWaveAfterLoop(Mesh *mesh, ParameterInput *pin,
                                           SimTime &time) {
  const Real amplitude = pin->GetReal("problem", "amplitude");
  const Real x1_min = mesh->mesh_size.xmin(X1DIR);
  const Real wavelength = mesh->mesh_size.xmax(X1DIR) - x1_min;
  Real error_sum = 0.0;
  Real maximum = 0.0;
  std::uint64_t values = 0;
  for (const auto &block : mesh->block_list) {
    const auto data = block->meshblock_data.Get();
    const auto z4c = data->Get("nr.z4c").data.GetHostMirrorAndCopy();
    const auto adm = data->Get("nr.adm").data.GetHostMirrorAndCopy();
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          const Real x = block->coords.Xc<X1DIR>(k, j, i);
          const Real phase =
              2.0 * M_PI * ((x - x1_min) / wavelength - time.time / wavelength);
          const Real h = amplitude * sin(phase);
          const Real physical_gxx = 1.0 - h;
          const Real expected_alpha = sqrt(physical_gxx);
          const Real dh_dt =
              -amplitude * (2.0 * M_PI / wavelength) * cos(phase);
          const Real expected_kxx = 0.5 * dh_dt / expected_alpha;
          const Real errors[3] = {
              std::abs(adm(nr::Index(nr::ADMComponent::gxx), k, j, i) -
                       physical_gxx),
              std::abs(adm(nr::Index(nr::ADMComponent::kxx), k, j, i) -
                       expected_kxx),
              std::abs(z4c(nr::Index(nr::Z4cComponent::alpha), k, j, i) -
                       expected_alpha)};
          for (const Real error : errors) {
            error_sum += error;
            maximum = std::max(maximum, error);
            ++values;
          }
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  Real global_sum = error_sum;
  Real global_maximum = maximum;
  std::uint64_t global_values = values;
  PARTHENON_MPI_CHECK(MPI_Allreduce(
      &error_sum, &global_sum, 1, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(&maximum, &global_maximum, 1,
                                    MPI_PARTHENON_REAL, MPI_MAX,
                                    MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(&values, &global_values, 1, MPI_UINT64_T,
                                    MPI_SUM, MPI_COMM_WORLD));
  error_sum = global_sum;
  maximum = global_maximum;
  values = global_values;
#endif
  const Real l1 = error_sum / static_cast<Real>(values);
  PARTHENON_REQUIRE(std::isfinite(l1) && std::isfinite(maximum),
                    "NR gauge-wave evolution produced a non-finite error");
  if (Globals::my_rank == 0)
    std::cout << "NR-2 gauge wave: cycles=" << time.ncycle << " L1=" << l1
              << " Linf=" << maximum << '\n';
}

void NumericalRelativityRobustStabilityAfterLoop(Mesh *mesh, ParameterInput *,
                                                 SimTime &time) {
  Real squared_sum = 0.0;
  Real maximum = 0.0;
  std::uint64_t values = 0;
  for (const auto &block : mesh->block_list) {
    const auto data = block->meshblock_data.Get();
    const auto z4c = data->Get("nr.z4c").data.GetHostMirrorAndCopy();
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          for (int component = 0; component < nr::kZ4cComponents; ++component) {
            const bool unit = component == nr::Index(nr::Z4cComponent::chi) ||
                              component == nr::Index(nr::Z4cComponent::gxx) ||
                              component == nr::Index(nr::Z4cComponent::gyy) ||
                              component == nr::Index(nr::Z4cComponent::gzz) ||
                              component == nr::Index(nr::Z4cComponent::alpha);
            const Real error = z4c(component, k, j, i) - (unit ? 1.0 : 0.0);
            squared_sum += error * error;
            maximum = std::max(maximum, std::abs(error));
            ++values;
          }
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  Real global_squared_sum = squared_sum;
  Real global_maximum = maximum;
  std::uint64_t global_values = values;
  PARTHENON_MPI_CHECK(MPI_Allreduce(&squared_sum, &global_squared_sum, 1,
                                    MPI_PARTHENON_REAL, MPI_SUM,
                                    MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(&maximum, &global_maximum, 1,
                                    MPI_PARTHENON_REAL, MPI_MAX,
                                    MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(&values, &global_values, 1, MPI_UINT64_T,
                                    MPI_SUM, MPI_COMM_WORLD));
  squared_sum = global_squared_sum;
  maximum = global_maximum;
  values = global_values;
#endif
  const Real rms = sqrt(squared_sum / static_cast<Real>(values));
  PARTHENON_REQUIRE(
      std::isfinite(rms) && std::isfinite(maximum),
      "NR robust-stability evolution produced a non-finite state");
  if (Globals::my_rank == 0)
    std::cout << "NR-2 robust stability: cycles=" << time.ncycle
              << " RMS=" << rms << " Linf=" << maximum << '\n';
}

} // namespace pangu::pgen
