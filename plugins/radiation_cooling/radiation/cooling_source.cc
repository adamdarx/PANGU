#include "radiation/cooling_source.h"

#include <string>
#include <vector>

#include "geometry_assembly.h"
#include "mhd/mhd_types.h"

namespace pangu::radiation {
using namespace parthenon::package::prelude;

TaskStatus ApplyCoolingMeshTask(MeshData<Real>* data, const Real time, const Real dt,
                                const bool record_diagnostics) {
  const auto& packages = data->GetParentPointer()->packages;
  const auto radiation = packages.Get("radiation");
  const auto mhd_package = packages.Get("mhd");
  const auto geometry_package = packages.Get("geometry");
  const Real h_target = radiation->Param<Real>("h_target");
  const Real beta_cool = radiation->Param<Real>("beta_cool");
  const Real start_time = radiation->Param<Real>("start_time");
  const Real ramp_time = radiation->Param<Real>("ramp_time");
  const Real rho_min = radiation->Param<Real>("rho_min");
  const Real radiation_sigma_max = radiation->Param<Real>("sigma_max");
  const bool bound_only = radiation->Param<bool>("bound_only");
  const bool track_energy = radiation->Param<bool>("track_energy");
  const auto eos = pangu::eos::ReadMagnetised(*mhd_package);
  const Real gamma_max = mhd_package->Param<Real>("gamma_max");
  const Real c2p_sigma_max = mhd_package->Param<Real>("sigma_max");
  const Real spin = geometry_package->Param<Real>("bh_spin");
  const bool use_excision = geometry_package->Param<bool>("excision");
  const Real excision_radius = geometry_package->Param<Real>("excision_radius");
  const Real ramp = CoolingRamp(time, start_time, ramp_time);

  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto magnetic = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto cooling_rate = data->PackVariables(std::vector<std::string>{"radiation.cooling_rate"});
  auto target = data->PackVariables(std::vector<std::string>{"radiation.target_internal_energy"});
  auto mask = data->PackVariables(std::vector<std::string>{"radiation.cooling_mask"});
  auto fraction = data->PackVariables(std::vector<std::string>{"radiation.cooling_fraction"});
  auto cooling_time = data->PackVariables(std::vector<std::string>{"radiation.cooling_time"});
  auto accumulated =
      data->PackVariables(std::vector<std::string>{"radiation.cumulative_removed_energy"});
  const auto spacetime = geometry::GetGeometry(packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed covariant target-thickness cooling",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        if (record_diagnostics) {
          cooling_rate(b, 0, k, j, i) = 0.0;
          target(b, 0, k, j, i) = 0.0;
          mask(b, 0, k, j, i) = 0.0;
          fraction(b, 0, k, j, i) = 0.0;
          cooling_time(b, 0, k, j, i) = 0.0;
        }
        if (!(ramp > 0.0)) {
          if (record_diagnostics)
            mask(b, 0, k, j, i) = static_cast<Real>(CoolingMask::before_start);
          return;
        }

        const auto& coordinates = primitive.GetCoords(b);
        const Real x1 = coordinates.Xc<1>(i);
        const Real x2 = coordinates.Xc<2>(j);
        const Real x3 = coordinates.Xc<3>(k);
        const int geometry_block = geometry_block_offset + b;
        if (use_excision && spacetime.IsExcised(x1, x2, x3, excision_radius)) {
          if (record_diagnostics)
            mask(b, 0, k, j, i) = static_cast<Real>(CoolingMask::excised);
          return;
        }

        relativity::MHDPrimitiveState state{};
        state.fluid.density = primitive(b, mhd::IDN, k, j, i);
        state.fluid.u[0] = primitive(b, mhd::IV1, k, j, i);
        state.fluid.u[1] = primitive(b, mhd::IV2, k, j, i);
        state.fluid.u[2] = primitive(b, mhd::IV3, k, j, i);
        const Real internal_energy = primitive(b, mhd::IPR, k, j, i);
        state.fluid.pressure = eos.PressureFromInternalEnergyDensity(internal_energy);
        for (int axis = 0; axis < 3; ++axis)
          state.magnetic[axis] = magnetic(b, axis, k, j, i);
        if (!(state.fluid.density >= rho_min)) {
          if (record_diagnostics)
            mask(b, 0, k, j, i) = static_cast<Real>(CoolingMask::low_density);
          return;
        }

        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k,
                                               j, i, coordinates);
        const Real magnetic_squared =
            relativity::ComputeComovingMagneticFieldSquared(state, metric);
        if (!(magnetic_squared / state.fluid.density <= radiation_sigma_max)) {
          if (record_diagnostics)
            mask(b, 0, k, j, i) = static_cast<Real>(CoolingMask::high_magnetization);
          return;
        }
        if (bound_only && !IsBoundFluid(state.fluid, internal_energy, eos, metric)) {
          if (record_diagnostics)
            mask(b, 0, k, j, i) = static_cast<Real>(CoolingMask::unbound);
          return;
        }

        const auto four_velocity = BuildFourVelocity(state.fluid, metric);
        const Real radius = spacetime.SphericalCoordinates(x1, x2, x3).radius;
        const auto cooling = EvaluateTargetThicknessCooling(
            state.fluid.density, internal_energy, eos, radius, spin, h_target, beta_cool, dt,
            four_velocity.upper[0], ramp);
        if (record_diagnostics) {
          target(b, 0, k, j, i) = cooling.target_internal_energy;
          cooling_time(b, 0, k, j, i) = cooling.cooling_time;
        }
        if (!(cooling.removed_internal_energy > 0.0)) {
          if (record_diagnostics)
            mask(b, 0, k, j, i) = static_cast<Real>(CoolingMask::below_target);
          return;
        }

        relativity::HydroConservedState base{conserved(b, mhd::IDN, k, j, i),
                                             {conserved(b, mhd::IM1, k, j, i),
                                              conserved(b, mhd::IM2, k, j, i),
                                              conserved(b, mhd::IM3, k, j, i)},
                                             conserved(b, mhd::IEN, k, j, i)};
        const auto increment =
            BuildCoolingIncrement(state.fluid, metric, cooling.removed_internal_energy);
        bool base_valid = true;
        const Real applied_scale = LimitCoolingIncrement(base, increment, state.magnetic, eos,
                                                        gamma_max, c2p_sigma_max, metric,
                                                        base_valid);
        if (!base_valid) {
          if (record_diagnostics)
            mask(b, 0, k, j, i) = static_cast<Real>(CoolingMask::base_state_invalid);
          return;
        }
        if (!(applied_scale > 0.0)) {
          if (record_diagnostics)
            mask(b, 0, k, j, i) = static_cast<Real>(CoolingMask::c2p_limited);
          return;
        }

        const auto cooled_state = AddCoolingIncrement(base, increment, applied_scale);
        conserved(b, mhd::IM1, k, j, i) = cooled_state.momentum[0];
        conserved(b, mhd::IM2, k, j, i) = cooled_state.momentum[1];
        conserved(b, mhd::IM3, k, j, i) = cooled_state.momentum[2];
        conserved(b, mhd::IEN, k, j, i) = cooled_state.energy;
        if (record_diagnostics) {
          const Real applied_removed = applied_scale * cooling.removed_internal_energy;
          const Real proper_dt = dt / four_velocity.upper[0];
          cooling_rate(b, 0, k, j, i) = applied_removed / proper_dt;
          fraction(b, 0, k, j, i) = applied_scale * cooling.removed_fraction;
          int cell_mask = static_cast<int>(CoolingMask::cooled);
          if (applied_scale < 1.0)
            cell_mask |= static_cast<int>(CoolingMask::c2p_limited);
          mask(b, 0, k, j, i) = static_cast<Real>(cell_mask);
        }
        if (track_energy)
          accumulated(b, 0, k, j, i) += applied_scale * increment.energy;
      });
  return TaskStatus::complete;
}

} // namespace pangu::radiation
