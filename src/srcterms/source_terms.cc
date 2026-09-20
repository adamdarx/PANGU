#include "srcterms/source_terms.h"

#include <limits>
#include <string>
#include <vector>

#include "eos/ideal_gas.h"
#include "hydro/hydro_types.h"
#include "pangu.h"
#include "srcterms/ism_cooling.h"
#include "units/units.h"
#include "utils/error_checking.hpp"

namespace pangu::srcterms {
using namespace parthenon::package::prelude;

namespace {

template <bool IsMHD>
TaskStatus ApplyBlock(std::shared_ptr<MeshBlockData<Real>>& data, const Real dt) {
  auto block = data->GetBlockPointer();
  const auto source = block->packages.Get("source_terms");
  const Real cooling = source->Param<Real>("cooling_rate");
  const Real heating = source->Param<Real>("heating_rate");
  const Real energy_floor = source->Param<Real>("energy_floor");
  const Real point_mass = source->Param<Real>("point_mass");
  const Real softening = source->Param<Real>("softening");
  const bool constant_acceleration = source->Param<bool>("constant_acceleration");
  const Real acceleration_value = source->Param<Real>("acceleration_value");
  const int acceleration_direction = source->Param<int>("acceleration_direction");
  const bool ism_cooling = source->Param<bool>("ism_cooling");
  const Real ism_heating_rate = source->Param<Real>("ism_heating_rate");
  const bool relativistic_cooling = source->Param<bool>("relativistic_cooling");
  const Real relativistic_rate = source->Param<Real>("relativistic_rate");
  const Real relativistic_power = source->Param<Real>("relativistic_power");
  if (cooling == 0.0 && heating == 0.0 && point_mass == 0.0 && !constant_acceleration &&
      !ism_cooling && !relativistic_cooling)
    return TaskStatus::complete;
  const std::string prefix = IsMHD ? "mhd" : "hydro";
  const auto fluid = block->packages.Get(prefix);
  auto conserved = data->PackVariables(std::vector<std::string>{prefix + ".cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{prefix + ".prim"});
  const bool ideal = fluid->Param<int>("eos_mode") == 0;
  const eos::IdealGas ideal_gas{fluid->Param<Real>("gamma")};
  // AthenaK stores relativistic-MHD primitive energy as internal-energy
  // density, while PANGU's Newtonian fluids and relativistic Hydro currently
  // expose pressure.  Normalize once here so every source uses the same
  // thermodynamic quantities as the AthenaK implementation.
  const bool primitive_energy_is_internal = IsMHD && fluid->Param<int>("physics_mode") != 0;
  const auto code_units = units::FromPackage(block->packages.Get("units"));
  const Real number_density_unit = code_units.NumberDensity();
  const Real cooling_unit =
      code_units.Pressure() / code_units.time / (number_density_unit * number_density_unit);
  const Real heating_unit = code_units.Pressure() / code_units.time / number_density_unit;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coords = block->coords;
  block->par_for(
      "PANGU generic sources", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real density = primitive(hydro::IDN, k, j, i);
        // Match AthenaK's ConstantAccel discretization: both sources use the
        // primitive state copied from the beginning of this RK stage.
        if (constant_acceleration) {
          const int momentum = hydro::IM1 + acceleration_direction;
          const Real source_momentum = density * acceleration_value;
          if (ideal)
            conserved(hydro::IEN, k, j, i) +=
                dt * source_momentum * primitive(hydro::IV1 + acceleration_direction, k, j, i);
          conserved(momentum, k, j, i) += dt * source_momentum;
        }
        if (point_mass != 0.0) {
          const Real x = coords.Xc<1>(i);
          const Real y = coords.Xc<2>(j);
          const Real z = coords.Xc<3>(k);
          const Real r2 = x * x + y * y + z * z + softening * softening;
          const Real inverse_r3 = 1.0 / (r2 * sqrt(r2));
          const Real acceleration[3]{-point_mass * x * inverse_r3, -point_mass * y * inverse_r3,
                                     -point_mass * z * inverse_r3};
          Real work = 0.0;
          Real a2 = 0.0;
          for (int axis = 0; axis < 3; ++axis) {
            work += conserved(hydro::IM1 + axis, k, j, i) * acceleration[axis];
            a2 += acceleration[axis] * acceleration[axis];
            conserved(hydro::IM1 + axis, k, j, i) += dt * density * acceleration[axis];
          }
          if (ideal)
            conserved(hydro::IEN, k, j, i) += dt * work + 0.5 * dt * dt * density * a2;
        }
        if (ideal) {
          const Real stored_energy = primitive(hydro::IPR, k, j, i);
          const Real internal_energy = primitive_energy_is_internal
                                           ? stored_energy
                                           : ideal_gas.InternalEnergyDensity(stored_energy);
          const Real pressure = ideal_gas.PressureFromInternalEnergyDensity(internal_energy);
          const Real loss = cooling * density * density;
          conserved(hydro::IEN, k, j, i) =
              fmax(energy_floor, conserved(hydro::IEN, k, j, i) + dt * (heating * density - loss));
          if (ism_cooling) {
            const Real temperature = code_units.Temperature() * pressure / density;
            const Real lambda_cooling = ISMCoolingFunction(temperature) / cooling_unit;
            const Real gamma_heating = ism_heating_rate / heating_unit;
            conserved(hydro::IEN, k, j, i) -=
                dt * density * (density * lambda_cooling - gamma_heating);
          }
          if (relativistic_cooling) {
            const Real temperature = pressure / density;
            const Real ux = primitive(hydro::IV1, k, j, i);
            const Real uy = primitive(hydro::IV2, k, j, i);
            const Real uz = primitive(hydro::IV3, k, j, i);
            const Real ut = sqrt(1.0 + ux * ux + uy * uy + uz * uz);
            const Real rate = pow(temperature * relativistic_rate, relativistic_power);
            conserved(hydro::IEN, k, j, i) -= dt * density * ut * rate;
            conserved(hydro::IM1, k, j, i) -= dt * density * ux * rate;
            conserved(hydro::IM2, k, j, i) -= dt * density * uy * rate;
            conserved(hydro::IM3, k, j, i) -= dt * density * uz * rate;
          }
        }
      });
  return TaskStatus::complete;
}

} // namespace

std::shared_ptr<StateDescriptor> Initialize(ParameterInput* pin) {
  auto package = std::make_shared<StateDescriptor>("source_terms");
  const Real cooling = pin->GetOrAddReal("source_terms", "cooling_rate", 0.0);
  const Real heating = pin->GetOrAddReal("source_terms", "heating_rate", 0.0);
  const Real floor = pin->GetOrAddReal("source_terms", "energy_floor", 0.0);
  const Real point_mass = pin->GetOrAddReal("source_terms", "point_mass", 0.0);
  const Real softening = pin->GetOrAddReal("source_terms", "softening", 0.0);
  const bool constant_acceleration =
      pin->GetOrAddBoolean("source_terms", "constant_acceleration", false);
  const Real acceleration_value = pin->GetOrAddReal("source_terms", "acceleration_value", 0.0);
  const int acceleration_direction =
      pin->GetOrAddInteger("source_terms", "acceleration_direction", 1) - 1;
  const bool ism_cooling = pin->GetOrAddBoolean("source_terms", "ism_cooling", false);
  const Real ism_heating_rate = pin->GetOrAddReal("source_terms", "ism_heating_rate", 0.0);
  const bool relativistic_cooling =
      pin->GetOrAddBoolean("source_terms", "relativistic_cooling", false);
  const Real relativistic_rate = pin->GetOrAddReal("source_terms", "relativistic_rate", 0.0);
  const Real relativistic_power = pin->GetOrAddReal("source_terms", "relativistic_power", 1.0);
  PARTHENON_REQUIRE(cooling >= 0.0 && heating >= 0.0 && floor >= 0.0 && point_mass >= 0.0 &&
                        softening >= 0.0 && ism_heating_rate >= 0.0 && relativistic_rate >= 0.0 &&
                        relativistic_power > 0.0,
                    "source term rates, floors, point mass and softening must be non-negative");
  PARTHENON_REQUIRE(acceleration_direction >= 0 && acceleration_direction < 3,
                    "source_terms/acceleration_direction must be 1, 2, or 3");
  package->AddParam<Real>("cooling_rate", cooling);
  package->AddParam<Real>("heating_rate", heating);
  package->AddParam<Real>("energy_floor", floor);
  package->AddParam<Real>("point_mass", point_mass);
  package->AddParam<Real>("softening", softening);
  package->AddParam<bool>("constant_acceleration", constant_acceleration);
  package->AddParam<Real>("acceleration_value", acceleration_value);
  package->AddParam<int>("acceleration_direction", acceleration_direction);
  package->AddParam<bool>("ism_cooling", ism_cooling);
  package->AddParam<Real>("ism_heating_rate", ism_heating_rate);
  package->AddParam<bool>("relativistic_cooling", relativistic_cooling);
  package->AddParam<Real>("relativistic_rate", relativistic_rate);
  package->AddParam<Real>("relativistic_power", relativistic_power);
  package->EstimateTimestepBlock = EstimateTimestepBlock;
  return package;
}

TaskStatus ApplyHydroBlockTask(std::shared_ptr<MeshBlockData<Real>>& data, const Real dt) {
  return ApplyBlock<false>(data, dt);
}

TaskStatus ApplyMHDBlockTask(std::shared_ptr<MeshBlockData<Real>>& data, const Real dt) {
  return ApplyBlock<true>(data, dt);
}

Real EstimateTimestepBlock(MeshBlockData<Real>* data) {
  auto block = data->GetBlockPointer();
  const auto source = block->packages.Get("source_terms");
  const bool ism_cooling = source->Param<bool>("ism_cooling");
  const bool relativistic_cooling = source->Param<bool>("relativistic_cooling");
  if (!ism_cooling && !relativistic_cooling)
    return std::numeric_limits<Real>::max();

  const bool is_mhd = block->packages.AllPackages().count("mhd") != 0;
  const std::string prefix = is_mhd ? "mhd" : "hydro";
  const auto fluid = block->packages.Get(prefix);
  if (fluid->Param<int>("eos_mode") != 0)
    return std::numeric_limits<Real>::max();
  const eos::IdealGas ideal_gas{fluid->Param<Real>("gamma")};
  const Real ism_heating_rate = source->Param<Real>("ism_heating_rate");
  const Real relativistic_rate = source->Param<Real>("relativistic_rate");
  const Real relativistic_power = source->Param<Real>("relativistic_power");
  const auto code_units = units::FromPackage(block->packages.Get("units"));
  const Real number_density_unit = code_units.NumberDensity();
  const Real cooling_unit =
      code_units.Pressure() / code_units.time / (number_density_unit * number_density_unit);
  const Real heating_unit = code_units.Pressure() / code_units.time / number_density_unit;
  const auto primitive = data->PackVariables(std::vector<std::string>{prefix + ".prim"});
  const bool primitive_energy_is_internal = is_mhd && fluid->Param<int>("physics_mode") != 0;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  Real minimum = std::numeric_limits<Real>::max();
  ParReduce(
      "PANGU source-term timestep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
        const Real density = primitive(hydro::IDN, k, j, i);
        const Real stored_energy = primitive(hydro::IPR, k, j, i);
        const Real internal_energy = primitive_energy_is_internal
                                         ? stored_energy
                                         : ideal_gas.InternalEnergyDensity(stored_energy);
        const Real pressure = ideal_gas.PressureFromInternalEnergyDensity(internal_energy);
        if (ism_cooling) {
          const Real temperature = code_units.Temperature() * pressure / density;
          const Real lambda_cooling = ISMCoolingFunction(temperature) / cooling_unit;
          const Real gamma_heating = ism_heating_rate / heating_unit;
          const Real rate = static_cast<Real>(std::numeric_limits<float>::min()) +
                            fabs(density * (density * lambda_cooling - gamma_heating));
          local = fmin(local, internal_energy / rate);
        }
        if (relativistic_cooling) {
          const Real ux = primitive(hydro::IV1, k, j, i);
          const Real uy = primitive(hydro::IV2, k, j, i);
          const Real uz = primitive(hydro::IV3, k, j, i);
          const Real ut = sqrt(1.0 + ux * ux + uy * uy + uz * uz);
          const Real rate =
              static_cast<Real>(std::numeric_limits<float>::min()) +
              fabs(density * ut * pow(pressure / density * relativistic_rate, relativistic_power));
          local = fmin(local, internal_energy / rate);
        }
      },
      Kokkos::Min<Real>(minimum));
  return minimum;
}

bool HasActiveSources(const parthenon::Packages_t& packages) {
  const auto source = packages.Get("source_terms");
  return source->Param<Real>("cooling_rate") != 0.0 || source->Param<Real>("heating_rate") != 0.0 ||
         source->Param<Real>("point_mass") != 0.0 || source->Param<bool>("constant_acceleration") ||
         source->Param<bool>("ism_cooling") || source->Param<bool>("relativistic_cooling");
}

} // namespace pangu::srcterms
