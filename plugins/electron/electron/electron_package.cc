#include "electron/electron_package.h"
#include "driver/stage_extension.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "electron/model/registry.h"
#include "eos/ideal_gas.h"
#include "geometry_assembly.h"
#include "mhd/mhd_types.h"
#include "mhd/passive_transport.h"
#include "relativity/relativistic_mhd.h"
#include "utils/error_checking.hpp"

namespace pangu::electron {
using namespace parthenon::package::prelude;

namespace {

constexpr int ktot_component = 0;
constexpr int diagnostic_raw_dissipation = 0;
constexpr int diagnostic_applied_dissipation = 1;
constexpr int diagnostic_heating_fraction = 2;
constexpr int diagnostic_temperature_ratio = 3;
constexpr int diagnostic_flags = 4;

void RejectUnsupportedEnergySources(const parthenon::Packages_t& packages) {
  const auto sources = packages.Get("source_terms");
  const bool external_source =
      sources->Param<Real>("cooling_rate") != 0.0 || sources->Param<Real>("heating_rate") != 0.0 ||
      sources->Param<Real>("point_mass") != 0.0 || sources->Param<bool>("constant_acceleration") ||
      sources->Param<bool>("ism_cooling") || sources->Param<bool>("relativistic_cooling");
  PARTHENON_REQUIRE(!external_source,
                    "electron heating does not yet support external fluid-energy sources");

}

template <int Direction, bool Inner>
void EntropyBoundaryBlock(std::shared_ptr<MeshBlockData<Real>>& data, const bool coarse) {
  if (coarse)
    return;
  auto* block = data->GetBlockPointer();
  auto electron = data->PackVariables(std::vector<std::string>{"electrons.cons"});
  const auto fluid = data->PackVariables(std::vector<std::string>{"mhd.cons"});
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
  const int components = electron.GetDim(4);
  block->par_for(
      "PANGU electron entropy boundary", 0, components - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        const int si = Direction == 0 ? source : i;
        const int sj = Direction == 1 ? source : j;
        const int sk = Direction == 2 ? source : k;
        const Real entropy = electron(n, sk, sj, si) / fluid(mhd::IDN, sk, sj, si);
        electron(n, k, j, i) = fluid(mhd::IDN, k, j, i) * entropy;
      });
}

void InitializeStateMesh(Mesh*, ParameterInput*, MeshData<Real>* data) {
  const auto package = data->GetParentPointer()->packages.Get("electrons");
  const auto mhd_package = data->GetParentPointer()->packages.Get("mhd");
  const int components = package->Param<int>("components");
  const Real gamma = mhd_package->Param<Real>("gamma");
  const Real gamma_e = package->Param<Real>("gamma_e");
  const Real gamma_p = package->Param<Real>("gamma_p");
  const eos::IdealGas ideal_gas{gamma};
  const Real fel_0 = package->Param<Real>("fel_0");
  const Real fel_constant = package->Param<Real>("fel_constant");
  const bool write_diagnostics = package->Param<bool>("write_diagnostics");
  const int diagnostic_component = package->Param<int>("diagnostic_component");
  const int diagnostic_model = package->Param<int>("diagnostic_model");
  auto electron_primitive = data->PackVariables(std::vector<std::string>{"electrons.prim"});
  auto electron_conserved = data->PackVariables(std::vector<std::string>{"electrons.cons"});
  const auto fluid_primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  const auto fluid_conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto magnetic = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU initialize electron entropies", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, 0, components - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        const Real density = fluid_primitive(b, mhd::IDN, k, j, i);
        const Real internal = fluid_primitive(b, mhd::IPR, k, j, i);
        const Real entropy = n == 0 ? (gamma - 1.0) * internal * pow(density, -gamma)
                                    : (gamma_e - 1.0) * fel_0 * internal * pow(density, -gamma_e);
        electron_primitive(b, n, k, j, i) = entropy;
        electron_conserved(b, n, k, j, i) = fluid_conserved(b, mhd::IDN, k, j, i) * entropy;
      });
  if (write_diagnostics) {
    auto diagnostics = data->PackVariables(std::vector<std::string>{"electrons.diagnostics"});
    const auto spacetime = geometry::GetGeometry(data->GetParentPointer()->packages);
    const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU initialize electron diagnostics", parthenon::DevExecSpace(), 0,
        data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const Real density = fluid_primitive(b, mhd::IDN, k, j, i);
          const Real internal = fluid_primitive(b, mhd::IPR, k, j, i);
          const auto& coordinates = fluid_primitive.GetCoords(b);
          const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                                 geometry_block_offset + b, k, j, i, coordinates);
          const relativity::MHDPrimitiveState state{
              {density,
               {fluid_primitive(b, mhd::IV1, k, j, i), fluid_primitive(b, mhd::IV2, k, j, i),
                fluid_primitive(b, mhd::IV3, k, j, i)},
               ideal_gas.PressureFromInternalEnergyDensity(internal)},
              {magnetic(b, 0, k, j, i), magnetic(b, 1, k, j, i), magnetic(b, 2, k, j, i)}};
          const Real magnetic_squared =
              relativity::ComputeComovingMagneticFieldSquared(state, metric);
          const Real fraction =
              HeatingFraction(diagnostic_model, density, internal, magnetic_squared,
                              electron_primitive(b, diagnostic_component, k, j, i), gamma, gamma_e,
                              gamma_p, fel_constant);
          diagnostics(b, diagnostic_raw_dissipation, k, j, i) = 0.0;
          diagnostics(b, diagnostic_applied_dissipation, k, j, i) = 0.0;
          diagnostics(b, diagnostic_heating_fraction, k, j, i) = fraction;
          diagnostics(b, diagnostic_temperature_ratio, k, j, i) = TemperatureRatio(
              density, internal, electron_primitive(b, diagnostic_component, k, j, i), gamma_e,
              gamma_p);
          diagnostics(b, diagnostic_flags, k, j, i) = 0.0;
        });
  }
}

TaskStatus ApplyHeatingMeshImpl(MeshData<Real>* old_data, MeshData<Real>* new_data) {
  auto* mesh = new_data->GetParentPointer();
  const auto package = mesh->packages.Get("electrons");
  const auto mhd_package = mesh->packages.Get("mhd");
  const Real gamma = mhd_package->Param<Real>("gamma");
  const Real gamma_e = package->Param<Real>("gamma_e");
  const Real gamma_p = package->Param<Real>("gamma_p");
  const eos::IdealGas ideal_gas{gamma};
  const Real fel_constant = package->Param<Real>("fel_constant");
  const Real sigma_heat_cutoff = package->Param<Real>("sigma_heat_cutoff");
  const Real tp_over_te_min = package->Param<Real>("tp_over_te_min");
  const Real tp_over_te_max = package->Param<Real>("tp_over_te_max");
  const bool enforce_positive = package->Param<bool>("enforce_positive_dissipation");
  const bool suppress_highb = package->Param<bool>("suppress_highb_heat");
  const bool limit_kel = package->Param<bool>("limit_kel");
  const bool write_diagnostics = package->Param<bool>("write_diagnostics");
  const auto component_indices =
      package->Param<std::array<int, heating_model_count>>("model_components");
  const int diagnostic_component = package->Param<int>("diagnostic_component");
  bool needs_magnetic_field = suppress_highb;
  for (int model_index = 0; model_index < heating_model_count; ++model_index)
    needs_magnetic_field = needs_magnetic_field ||
                           (component_indices[model_index] >= 0 &&
                            model::RegisteredModels::magnetic_requirements[model_index]);

  const auto old_fluid = old_data->PackVariables(std::vector<std::string>{"mhd.prim"});
  const auto old_magnetic = old_data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  const auto old_electron = old_data->PackVariables(std::vector<std::string>{"electrons.prim"});
  const auto new_fluid = new_data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto new_electron = new_data->PackVariables(std::vector<std::string>{"electrons.prim"});
  auto diagnostics = new_data->PackVariables(
      std::vector<std::string>{write_diagnostics ? "electrons.diagnostics" : "electrons.prim"});
  const auto ib = new_data->GetBoundsI(IndexDomain::interior);
  const auto jb = new_data->GetBoundsJ(IndexDomain::interior);
  const auto kb = new_data->GetBoundsK(IndexDomain::interior);
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const int geometry_block_offset = new_data->GetBlockData(0)->GetBlockPointer()->lid;
  const int blocks = new_data->NumBlocks();
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU apply electron heating models", parthenon::DevExecSpace(), 0,
      blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real magnetic_squared = 0.0;
        if (needs_magnetic_field) {
          const auto& coordinates = new_fluid.GetCoords(b);
          const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                                 geometry_block_offset + b, k, j, i, coordinates);
          const relativity::MHDPrimitiveState state{
              {old_fluid(b, mhd::IDN, k, j, i),
               {old_fluid(b, mhd::IV1, k, j, i), old_fluid(b, mhd::IV2, k, j, i),
                old_fluid(b, mhd::IV3, k, j, i)},
               ideal_gas.PressureFromInternalEnergyDensity(
                   old_fluid(b, mhd::IPR, k, j, i))},
              {old_magnetic(b, 0, k, j, i), old_magnetic(b, 1, k, j, i),
               old_magnetic(b, 2, k, j, i)}};
          magnetic_squared = relativity::ComputeComovingMagneticFieldSquared(state, metric);
        }
        const Real old_density = old_fluid(b, mhd::IDN, k, j, i);
        const Real old_internal = old_fluid(b, mhd::IPR, k, j, i);
        const Real new_density = new_fluid(b, mhd::IDN, k, j, i);
        const Real new_internal = new_fluid(b, mhd::IPR, k, j, i);
        const Real old_total_entropy = old_electron(b, ktot_component, k, j, i);
        const Real total_entropy = EnergyConservingEntropy(new_density, new_internal, gamma);
        const Real raw_dissipation = DissipativeElectronEntropy(
            old_density, total_entropy, new_electron(b, ktot_component, k, j, i), gamma, gamma_e);
        Real applied_dissipation = raw_dissipation;
        int common_flags = heating_none;
        const Real magnetization = magnetic_squared / old_density;
        if (suppress_highb && magnetization > sigma_heat_cutoff) {
          applied_dissipation = 0.0;
          common_flags |= heating_high_magnetization_suppressed;
        } else if (enforce_positive && applied_dissipation < 0.0) {
          applied_dissipation = 0.0;
          common_flags |= heating_negative_dissipation_clipped;
        }
        const auto bounds = ElectronEntropyBounds(old_density, old_total_entropy, gamma, gamma_e,
                                                  gamma_p, tp_over_te_min, tp_over_te_max);
        for (int model_index = 0; model_index < heating_model_count; ++model_index) {
          const int component = component_indices[model_index];
          if (component < 0)
            continue;
          Real fraction = HeatingFraction(model_index, old_density, old_internal, magnetic_squared,
                                          old_electron(b, component, k, j, i), gamma, gamma_e,
                                          gamma_p, fel_constant);
          if ((common_flags & heating_high_magnetization_suppressed) != 0)
            fraction = 0.0;
          const auto update = ApplyElectronEntropyUpdate(
              new_electron(b, component, k, j, i), fraction, applied_dissipation, bounds, limit_kel,
              new_density, new_internal, gamma_e, gamma_p, common_flags);
          new_electron(b, component, k, j, i) = update.electron_entropy;
          if (write_diagnostics && component == diagnostic_component) {
            diagnostics(b, diagnostic_heating_fraction, k, j, i) = update.heating_fraction;
            diagnostics(b, diagnostic_temperature_ratio, k, j, i) = update.temperature_ratio;
            diagnostics(b, diagnostic_flags, k, j, i) = static_cast<Real>(update.flags);
          }
        }
        new_electron(b, ktot_component, k, j, i) = total_entropy;
        if (write_diagnostics) {
          diagnostics(b, diagnostic_raw_dissipation, k, j, i) = raw_dissipation;
          diagnostics(b, diagnostic_applied_dissipation, k, j, i) = applied_dissipation;
        }
      });
  return TaskStatus::complete;
}

template <int Direction, hydro::Reconstruction Method>
void FinalizeDirectionBlock(const std::shared_ptr<MeshBlockData<Real>>& data) {
  auto* block = data->GetBlockPointer();
  const auto primitive = data->PackVariables(std::vector<std::string>{"electrons.prim"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"electrons.cons"});
  const auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  int il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  if constexpr (Direction == 0)
    ++iu;
  if constexpr (Direction == 1)
    ++ju;
  if constexpr (Direction == 2)
    ++ku;
  const int components = primitive.GetDim(4);
  block->par_for(
      "PANGU finalize electron entropy flux", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int lk = k - (Direction == 2);
        const int lj = j - (Direction == 1);
        const int li = i - (Direction == 0);
        const bool first_order = flags(0, lk, lj, li) > 0.5 || flags(0, k, j, i) > 0.5;
        const Real coefficient_left = conserved.flux(Direction + 1, 0, k, j, i);
        const Real coefficient_right = conserved.flux(Direction + 1, 1, k, j, i);
        for (int n = 0; n < components; ++n) {
          Real left = 0.0, right = 0.0;
          if (first_order) {
            left = primitive(n, lk, lj, li);
            right = primitive(n, k, j, i);
          } else {
            mhd::ReconstructPassive<Direction, Method>(primitive, n, k, j, i, left, right);
          }
          conserved.flux(Direction + 1, n, k, j, i) =
              coefficient_left * left + coefficient_right * right;
        }
      });
}

template <int Direction, hydro::Reconstruction Method>
void FinalizeDirectionMesh(MeshData<Real>* data) {
  const auto primitive = data->PackVariables(std::vector<std::string>{"electrons.prim"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"electrons.cons"});
  const auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  int il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  if constexpr (Direction == 0)
    ++iu;
  if constexpr (Direction == 1)
    ++ju;
  if constexpr (Direction == 2)
    ++ku;
  const int components = primitive.GetDim(4);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed finalize electron entropy flux",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const int lk = k - (Direction == 2);
        const int lj = j - (Direction == 1);
        const int li = i - (Direction == 0);
        const bool first_order = flags(b, 0, lk, lj, li) > 0.5 || flags(b, 0, k, j, i) > 0.5;
        auto block_conserved = conserved(b);
        const Real coefficient_left = block_conserved.flux(Direction + 1, 0, k, j, i);
        const Real coefficient_right = block_conserved.flux(Direction + 1, 1, k, j, i);
        for (int n = 0; n < components; ++n) {
          Real left = 0.0, right = 0.0;
          if (first_order) {
            left = primitive(b, n, lk, lj, li);
            right = primitive(b, n, k, j, i);
          } else {
            mhd::ReconstructPassiveMesh<Direction, Method>(primitive, b, n, k, j, i, left, right);
          }
          block_conserved.flux(Direction + 1, n, k, j, i) =
              coefficient_left * left + coefficient_right * right;
        }
      });
}

template <hydro::Reconstruction Method>
TaskStatus FinalizeFluxesBlock(const std::shared_ptr<MeshBlockData<Real>>& data) {
  FinalizeDirectionBlock<0, Method>(data);
  if (data->GetBlockPointer()->pmy_mesh->ndim >= 2)
    FinalizeDirectionBlock<1, Method>(data);
  if (data->GetBlockPointer()->pmy_mesh->ndim >= 3)
    FinalizeDirectionBlock<2, Method>(data);
  return TaskStatus::complete;
}

template <hydro::Reconstruction Method> TaskStatus FinalizeFluxesMesh(MeshData<Real>* data) {
  FinalizeDirectionMesh<0, Method>(data);
  if (data->GetParentPointer()->ndim >= 2)
    FinalizeDirectionMesh<1, Method>(data);
  if (data->GetParentPointer()->ndim >= 3)
    FinalizeDirectionMesh<2, Method>(data);
  return TaskStatus::complete;
}

} // namespace

bool Enabled(ParameterInput* pin) {
  const bool has_enabled = pin->DoesParameterExist("electrons", "enabled");
  const bool has_on = pin->DoesParameterExist("electrons", "on");
  if (!has_enabled && !has_on)
    return false;
  const bool enabled =
      has_enabled ? pin->GetBoolean("electrons", "enabled") : pin->GetBoolean("electrons", "on");
  if (has_enabled && has_on)
    PARTHENON_REQUIRE(enabled == pin->GetBoolean("electrons", "on"),
                      "electrons/enabled and electrons/on disagree");
  return enabled;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput* pin,
                                            const parthenon::Packages_t& packages) {
  PARTHENON_REQUIRE(packages.AllPackages().count("mhd") != 0, "electrons require an MHD package");
  const auto mhd_package = packages.Get("mhd");
  const auto physics = static_cast<relativity::HydroMode>(mhd_package->Param<int>("physics_mode"));
  PARTHENON_REQUIRE(physics != relativity::HydroMode::newtonian,
                    "electrons require SRMHD or GRMHD");
  PARTHENON_REQUIRE(mhd_package->Param<int>("nscalars") == 0,
                    "electrons cannot be combined with mhd/nscalars");

  auto package = std::make_shared<StateDescriptor>("electrons");
  const bool heating = pin->GetOrAddBoolean("electrons", "heating", true);
  package->AddParam<driver::PassiveStageExtension>(
      std::string(driver::passive_stage_extension_key),
      driver::PassiveStageExtension{
          "electrons.cons", heating, CalculateFluxesMeshTask,
          relativity::ApplyMHDFluxCorrectionWithPassiveMeshTask,
          UpdateConservedMeshTask, ConservedToPrimitiveMeshTask,
          ApplyHeatingMeshTask, PrimitiveToConservedMeshTask,
          ApplyPhysicalBoundariesMeshTask,
          relativity::CalculateMHDFluxesWithPassiveBlockTask,
          relativity::ApplyMHDFluxCorrectionWithPassiveBlockTask, CalculateFluxesBlockTask});
  const Real gamma_e = pin->GetOrAddReal("electrons", "gamma_e", 4.0 / 3.0);
  const Real gamma_p = pin->GetOrAddReal("electrons", "gamma_p", 5.0 / 3.0);
  const Real fel_0 = pin->GetOrAddReal("electrons", "fel_0", 0.01);
  const Real fel_constant = pin->GetOrAddReal("electrons", "fel_constant", 0.1);
  const bool enforce_positive =
      pin->GetOrAddBoolean("electrons", "enforce_positive_dissipation", false);
  const bool suppress_highb = pin->GetOrAddBoolean("electrons", "suppress_highb_heat", false);
  const Real sigma_heat_cutoff = pin->GetOrAddReal("electrons", "sigma_heat_cutoff", 1.0);
  const bool limit_kel = pin->GetOrAddBoolean("electrons", "limit_kel", true);
  const Real tp_over_te_min = pin->GetOrAddReal("electrons", "tp_over_te_min", 1.0e-3);
  const Real tp_over_te_max = pin->GetOrAddReal("electrons", "tp_over_te_max", 1.0e3);
  const bool reinitialize = pin->GetOrAddBoolean("electrons", "reinitialize", false);
  const bool write_diagnostics = pin->GetOrAddBoolean("electrons", "write_diagnostics", heating);
  const bool init_to_fel_0 = pin->GetOrAddBoolean("electrons", "init_to_fel_0", true);
  PARTHENON_REQUIRE(gamma_e > 1.0, "electrons/gamma_e must be greater than one");
  PARTHENON_REQUIRE(gamma_p > 1.0, "electrons/gamma_p must be greater than one");
  PARTHENON_REQUIRE(fel_0 >= 0.0 && fel_0 <= 1.0, "electrons/fel_0 must be in [0,1]");
  PARTHENON_REQUIRE(fel_constant >= 0.0 && fel_constant <= 1.0,
                    "electrons/fel_constant must be in [0,1]");
  PARTHENON_REQUIRE(sigma_heat_cutoff > 0.0, "electrons/sigma_heat_cutoff must be positive");
  PARTHENON_REQUIRE(tp_over_te_min > 0.0 && tp_over_te_max > tp_over_te_min,
                    "electron temperature-ratio limits must satisfy 0 < min < max");
  PARTHENON_REQUIRE(init_to_fel_0, "EH-2 supports initialization through electrons/fel_0 only");
  PARTHENON_REQUIRE(!reinitialize, "electron restart reinitialization is not implemented yet");
  const auto reconstruction = static_cast<hydro::Reconstruction>(
      packages.Get("mhd")->Param<int>("reconstruction"));
  PARTHENON_REQUIRE(reconstruct::Describe(reconstruction).passive_transport,
                    "selected reconstruction does not support electron transport");

  std::vector<std::string> labels{"Ktot"};
  std::array<int, heating_model_count> model_components{};
  model_components.fill(-1);
  int diagnostic_component = -1;
  int diagnostic_model = -1;
  std::array<int, heating_model_count> ordered_models{};
  for (int model_index = 0; model_index < heating_model_count; ++model_index)
    ordered_models[model_index] = model_index;
  std::sort(ordered_models.begin(), ordered_models.end(), [](const int left, const int right) {
    return std::string_view(model::RegisteredModels::keys[left]) <
           std::string_view(model::RegisteredModels::keys[right]);
  });
  for (const int model_index : ordered_models) {
    if (pin->GetOrAddBoolean("electrons", model::RegisteredModels::keys[model_index], false)) {
      labels.emplace_back(model::RegisteredModels::labels[model_index]);
      model_components[model_index] = static_cast<int>(labels.size()) - 1;
      if (diagnostic_component < 0) {
        diagnostic_component = model_components[model_index];
        diagnostic_model = model_index;
      }
    }
  }
  PARTHENON_REQUIRE(labels.size() > 1, "electrons require at least one enabled heating model");
  if (heating)
    RejectUnsupportedEnergySources(packages);
  const int components = static_cast<int>(labels.size());
  package->AddParam<int>("components", components);
  package->AddParam<bool>("heating", heating);
  package->AddParam<Real>("gamma_e", gamma_e);
  package->AddParam<Real>("gamma_p", gamma_p);
  package->AddParam<Real>("fel_0", fel_0);
  package->AddParam<Real>("fel_constant", fel_constant);
  package->AddParam<bool>("enforce_positive_dissipation", enforce_positive);
  package->AddParam<bool>("suppress_highb_heat", suppress_highb);
  package->AddParam<Real>("sigma_heat_cutoff", sigma_heat_cutoff);
  package->AddParam<bool>("limit_kel", limit_kel);
  package->AddParam<Real>("tp_over_te_min", tp_over_te_min);
  package->AddParam<Real>("tp_over_te_max", tp_over_te_max);
  package->AddParam<bool>("write_diagnostics", write_diagnostics);
  package->AddParam<std::array<int, heating_model_count>>("model_components", model_components);
  package->AddParam<int>("diagnostic_component", diagnostic_component);
  package->AddParam<int>("diagnostic_model", diagnostic_model);
  package->AddParam<std::vector<std::string>>("component_labels", labels);

  Metadata conserved({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
                      Metadata::FillGhost, Metadata::Restart},
                     std::vector<int>{components}, labels);
  conserved.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                                  parthenon::refinement_ops::RestrictAverage>();
  package->AddField("electrons.cons", conserved);
  package->AddField("electrons.prim", Metadata({Metadata::Cell, Metadata::Derived},
                                               std::vector<int>{components}, labels));
  if (write_diagnostics) {
    package->AddField(
        "electrons.diagnostics",
        Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, std::vector<int>{5},
                 std::vector<std::string>{"raw_dissipation", "applied_dissipation",
                                          "heating_fraction", "tp_over_te", "flags"}));
  }
  package->PostProblemGeneratorMesh = InitializeStateMesh;
  package->PostFillDerivedBlock = ConservedToPrimitiveBlock;
  package->UserBoundaryFunctions[parthenon::BoundaryFace::inner_x1].push_back(
      EntropyBoundaryBlock<0, true>);
  package->UserBoundaryFunctions[parthenon::BoundaryFace::outer_x1].push_back(
      EntropyBoundaryBlock<0, false>);
  package->UserBoundaryFunctions[parthenon::BoundaryFace::inner_x2].push_back(
      EntropyBoundaryBlock<1, true>);
  package->UserBoundaryFunctions[parthenon::BoundaryFace::outer_x2].push_back(
      EntropyBoundaryBlock<1, false>);
  package->UserBoundaryFunctions[parthenon::BoundaryFace::inner_x3].push_back(
      EntropyBoundaryBlock<2, true>);
  package->UserBoundaryFunctions[parthenon::BoundaryFace::outer_x3].push_back(
      EntropyBoundaryBlock<2, false>);
  return package;
}

void ConservedToPrimitiveBlock(MeshBlockData<Real>* data) {
  auto* block = data->GetBlockPointer();
  auto primitive = data->PackVariables(std::vector<std::string>{"electrons.prim"});
  const auto conserved = data->PackVariables(std::vector<std::string>{"electrons.cons"});
  const auto fluid = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const int components = primitive.GetDim(4);
  block->par_for(
      "PANGU electron conserved to primitive", 0, components - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        primitive(n, k, j, i) = conserved(n, k, j, i) / fluid(mhd::IDN, k, j, i);
      });
}

void ConservedToPrimitiveMesh(MeshData<Real>* data) {
  auto primitive = data->PackVariables(std::vector<std::string>{"electrons.prim"});
  const auto conserved = data->PackVariables(std::vector<std::string>{"electrons.cons"});
  const auto fluid = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto ib = data->GetBoundsI(IndexDomain::entire);
  const auto jb = data->GetBoundsJ(IndexDomain::entire);
  const auto kb = data->GetBoundsK(IndexDomain::entire);
  const int components = primitive.GetDim(4);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed electron conserved to primitive",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, 0, components - 1, kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        primitive(b, n, k, j, i) = conserved(b, n, k, j, i) / fluid(b, mhd::IDN, k, j, i);
      });
}

TaskStatus ConservedToPrimitiveMeshTask(MeshData<Real>* data) {
  ConservedToPrimitiveMesh(data);
  return TaskStatus::complete;
}

TaskStatus ApplyPhysicalBoundariesMeshTask(MeshData<Real>* data) {
  const int ndim = data->GetParentPointer()->ndim;
  for (int b = 0; b < data->NumBlocks(); ++b) {
    auto block_data = data->GetBlockData(b);
    const auto* block = block_data->GetBlockPointer();
    const auto physical = [&](const parthenon::BoundaryFace face) {
      if (block->boundary_flag[face] != parthenon::BoundaryFlag::user)
        return false;
      if (ndim < 3 && (face == parthenon::inner_x3 || face == parthenon::outer_x3))
        return false;
      if (ndim < 2 && (face == parthenon::inner_x2 || face == parthenon::outer_x2))
        return false;
      return true;
    };
    if (physical(parthenon::inner_x1))
      EntropyBoundaryBlock<0, true>(block_data, false);
    if (physical(parthenon::outer_x1))
      EntropyBoundaryBlock<0, false>(block_data, false);
    if (physical(parthenon::inner_x2))
      EntropyBoundaryBlock<1, true>(block_data, false);
    if (physical(parthenon::outer_x2))
      EntropyBoundaryBlock<1, false>(block_data, false);
    if (physical(parthenon::inner_x3))
      EntropyBoundaryBlock<2, true>(block_data, false);
    if (physical(parthenon::outer_x3))
      EntropyBoundaryBlock<2, false>(block_data, false);
  }
  return TaskStatus::complete;
}

TaskStatus ApplyHeatingMeshTask(MeshData<Real>* old_data, MeshData<Real>* new_data) {
  return ApplyHeatingMeshImpl(old_data, new_data);
}

void PrimitiveToConservedBlock(MeshBlockData<Real>* data, const IndexDomain domain) {
  auto* block = data->GetBlockPointer();
  const auto primitive = data->PackVariables(std::vector<std::string>{"electrons.prim"});
  auto conserved = data->PackVariables(std::vector<std::string>{"electrons.cons"});
  const auto fluid = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto ib = block->cellbounds.GetBoundsI(domain);
  const auto jb = block->cellbounds.GetBoundsJ(domain);
  const auto kb = block->cellbounds.GetBoundsK(domain);
  const int components = primitive.GetDim(4);
  block->par_for(
      "PANGU electron primitive to conserved", 0, components - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        conserved(n, k, j, i) = fluid(mhd::IDN, k, j, i) * primitive(n, k, j, i);
      });
}

void PrimitiveToConservedMesh(MeshData<Real>* data, const IndexDomain domain) {
  const auto primitive = data->PackVariables(std::vector<std::string>{"electrons.prim"});
  auto conserved = data->PackVariables(std::vector<std::string>{"electrons.cons"});
  const auto fluid = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto ib = data->GetBoundsI(domain);
  const auto jb = data->GetBoundsJ(domain);
  const auto kb = data->GetBoundsK(domain);
  const int components = primitive.GetDim(4);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed electron primitive to conserved",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, 0, components - 1, kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        conserved(b, n, k, j, i) = fluid(b, mhd::IDN, k, j, i) * primitive(b, n, k, j, i);
      });
}

TaskStatus PrimitiveToConservedBlockTask(std::shared_ptr<MeshBlockData<Real>>& data) {
  PrimitiveToConservedBlock(data.get(), IndexDomain::entire);
  return TaskStatus::complete;
}

TaskStatus PrimitiveToConservedMeshTask(MeshData<Real>* data) {
  PrimitiveToConservedMesh(data, IndexDomain::entire);
  return TaskStatus::complete;
}

TaskStatus CalculateFluxesBlockTask(std::shared_ptr<MeshBlockData<Real>>& data) {
  const auto reconstruction = static_cast<hydro::Reconstruction>(
      data->GetBlockPointer()->packages.Get("mhd")->Param<int>("reconstruction"));
  return reconstruct::VisitPassiveTransport(reconstruction, [&]<hydro::Reconstruction Method>() {
    return FinalizeFluxesBlock<Method>(data);
  });
}

TaskStatus CalculateFluxesMeshTask(MeshData<Real>* data) {
  const auto reconstruction = static_cast<hydro::Reconstruction>(
      data->GetParentPointer()->packages.Get("mhd")->Param<int>("reconstruction"));
  return reconstruct::VisitPassiveTransport(reconstruction, [&]<hydro::Reconstruction Method>() {
    return FinalizeFluxesMesh<Method>(data);
  });
}

TaskStatus UpdateConservedMeshTask(MeshData<Real>* current, MeshData<Real>* base, const Real gam0,
                                   const Real gam1, const Real beta_dt, MeshData<Real>* next) {
  const auto state = current->PackVariablesAndFluxes(std::vector<std::string>{"electrons.cons"});
  const auto initial = base->PackVariables(std::vector<std::string>{"electrons.cons"});
  auto output = next->PackVariables(std::vector<std::string>{"electrons.cons"});
  const auto ib = current->GetBoundsI(IndexDomain::interior);
  const auto jb = current->GetBoundsJ(IndexDomain::interior);
  const auto kb = current->GetBoundsK(IndexDomain::interior);
  const int ndim = current->GetNDim();
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU electron conservative RK update", parthenon::DevExecSpace(), 0,
      state.GetDim(5) - 1, 0, state.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        const auto& coords = state.GetCoords(b);
        const auto& block_state = state(b);
        Real divergence =
            (block_state.flux(X1DIR, n, k, j, i + 1) - block_state.flux(X1DIR, n, k, j, i)) /
            coords.Dxc<X1DIR>(k, j, i);
        if (ndim >= 2)
          divergence +=
              (block_state.flux(X2DIR, n, k, j + 1, i) - block_state.flux(X2DIR, n, k, j, i)) /
              coords.Dxc<X2DIR>(k, j, i);
        if (ndim >= 3)
          divergence +=
              (block_state.flux(X3DIR, n, k + 1, j, i) - block_state.flux(X3DIR, n, k, j, i)) /
              coords.Dxc<X3DIR>(k, j, i);
        output(b, n, k, j, i) =
            gam0 * state(b, n, k, j, i) + gam1 * initial(b, n, k, j, i) - beta_dt * divergence;
      });
  return TaskStatus::complete;
}

} // namespace pangu::electron
