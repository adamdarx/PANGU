#include "radiation/radiation_package.h"

#include <string>
#include <vector>

#include "driver/stage_extension.h"
#include "driver/task_assembly.h"
#include "mhd/mhd_types.h"
#include "pangu.h"
#include "radiation/cooling_source.h"

namespace pangu::radiation {
using namespace parthenon::package::prelude;
using parthenon::TaskID;
using parthenon::TaskList;

namespace {

TaskID AddCoolingStageTask(TaskID dependency, TaskID face_update, TaskList& tasks,
                           MeshData<Real>* next, const Real time, const Real dt,
                           const bool record_diagnostics) {
  return tasks.AddTask(dependency | face_update, ApplyCoolingMeshTask, next, time, dt,
                       record_diagnostics);
}

// Conserved RK update fused with the radiation energy ledgers.
TaskStatus UpdateConservedWithLedgerMeshTask(MeshData<Real>* current, MeshData<Real>* base,
                                             const Real gam0, const Real gam1,
                                             const Real beta_dt, MeshData<Real>* next) {
  auto* mesh = current->GetParentPointer();
  const auto variables = driver::CellConservedVariables(mesh);
  const auto state = current->PackVariablesAndFluxes(variables);
  const auto initial = base->PackVariables(variables);
  const auto output = next->PackVariables(variables);
  const auto ib = current->GetBoundsI(IndexDomain::interior);
  const auto jb = current->GetBoundsJ(IndexDomain::interior);
  const auto kb = current->GetBoundsK(IndexDomain::interior);
  const int ndim = state.GetNdim();
  // The radiation ledgers obey the same low-storage RK algebra as mhd.cons.
  // Advance them in the energy-component thread of this existing kernel so
  // enabling cooling does not add a second full-grid RK-update pass.
  const std::vector<std::string> ledger_fields{"radiation.cumulative_removed_energy",
                                               "radiation.cumulative_boundary_energy"};
  const auto radiation_current = current->PackVariables(ledger_fields);
  const auto radiation_base = base->PackVariables(ledger_fields);
  const auto radiation_next = next->PackVariables(ledger_fields);
  const bool physical_inner_x1 =
      mesh->mesh_bcs[parthenon::BoundaryFace::inner_x1] == parthenon::BoundaryFlag::user;
  const bool physical_outer_x1 =
      mesh->mesh_bcs[parthenon::BoundaryFace::outer_x1] == parthenon::BoundaryFlag::user;
  const bool physical_inner_x2 =
      mesh->mesh_bcs[parthenon::BoundaryFace::inner_x2] == parthenon::BoundaryFlag::user;
  const bool physical_outer_x2 =
      mesh->mesh_bcs[parthenon::BoundaryFace::outer_x2] == parthenon::BoundaryFlag::user;
  const bool physical_inner_x3 =
      mesh->mesh_bcs[parthenon::BoundaryFace::inner_x3] == parthenon::BoundaryFlag::user;
  const bool physical_outer_x3 =
      mesh->mesh_bcs[parthenon::BoundaryFace::outer_x3] == parthenon::BoundaryFlag::user;
  const Real domain_min_x1 = mesh->mesh_size.xmin(parthenon::X1DIR);
  const Real domain_max_x1 = mesh->mesh_size.xmax(parthenon::X1DIR);
  const Real domain_min_x2 = mesh->mesh_size.xmin(parthenon::X2DIR);
  const Real domain_max_x2 = mesh->mesh_size.xmax(parthenon::X2DIR);
  const Real domain_min_x3 = mesh->mesh_size.xmin(parthenon::X3DIR);
  const Real domain_max_x3 = mesh->mesh_size.xmax(parthenon::X3DIR);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU radiation-fused cell update",
      parthenon::DevExecSpace(), 0, state.GetDim(5) - 1, 0, state.GetDim(4) - 1, kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
        if (state.IsAllocated(m, n) && initial.IsAllocated(m, n) && output.IsAllocated(m, n)) {
          const auto& coords = state.GetCoords(m);
          const auto& block_state = state(m);
          Real divergence = (block_state.flux(parthenon::X1DIR, n, k, j, i + 1) -
                             block_state.flux(parthenon::X1DIR, n, k, j, i)) /
                            coords.Dxc<parthenon::X1DIR>(k, j, i);
          if (ndim >= 2)
            divergence += (block_state.flux(parthenon::X2DIR, n, k, j + 1, i) -
                           block_state.flux(parthenon::X2DIR, n, k, j, i)) /
                          coords.Dxc<parthenon::X2DIR>(k, j, i);
          if (ndim >= 3)
            divergence += (block_state.flux(parthenon::X3DIR, n, k + 1, j, i) -
                           block_state.flux(parthenon::X3DIR, n, k, j, i)) /
                          coords.Dxc<parthenon::X3DIR>(k, j, i);
          output(m, n, k, j, i) =
              gam0 * state(m, n, k, j, i) + gam1 * initial(m, n, k, j, i) - beta_dt * divergence;
          if (n == mhd::IEN) {
            radiation_next(m, 0, k, j, i) =
                gam0 * radiation_current(m, 0, k, j, i) + gam1 * radiation_base(m, 0, k, j, i);
            Real outward_divergence = 0.0;
            if (i == ib.s && physical_inner_x1 &&
                fabs(coords.Xf<parthenon::X1DIR>(k, j, ib.s) - domain_min_x1) <
                    0.25 * coords.Dxc<parthenon::X1DIR>(k, j, i))
              outward_divergence -= state(m).flux(parthenon::X1DIR, mhd::IEN, k, j, i) /
                                    coords.Dxc<parthenon::X1DIR>(k, j, i);
            if (i == ib.e && physical_outer_x1 &&
                fabs(coords.Xf<parthenon::X1DIR>(k, j, ib.e + 1) - domain_max_x1) <
                    0.25 * coords.Dxc<parthenon::X1DIR>(k, j, i))
              outward_divergence += state(m).flux(parthenon::X1DIR, mhd::IEN, k, j, i + 1) /
                                    coords.Dxc<parthenon::X1DIR>(k, j, i);
            if (ndim >= 2 && j == jb.s && physical_inner_x2 &&
                fabs(coords.Xf<parthenon::X2DIR>(k, jb.s, i) - domain_min_x2) <
                    0.25 * coords.Dxc<parthenon::X2DIR>(k, j, i))
              outward_divergence -= state(m).flux(parthenon::X2DIR, mhd::IEN, k, j, i) /
                                    coords.Dxc<parthenon::X2DIR>(k, j, i);
            if (ndim >= 2 && j == jb.e && physical_outer_x2 &&
                fabs(coords.Xf<parthenon::X2DIR>(k, jb.e + 1, i) - domain_max_x2) <
                    0.25 * coords.Dxc<parthenon::X2DIR>(k, j, i))
              outward_divergence += state(m).flux(parthenon::X2DIR, mhd::IEN, k, j + 1, i) /
                                    coords.Dxc<parthenon::X2DIR>(k, j, i);
            if (ndim >= 3 && k == kb.s && physical_inner_x3 &&
                fabs(coords.Xf<parthenon::X3DIR>(kb.s, j, i) - domain_min_x3) <
                    0.25 * coords.Dxc<parthenon::X3DIR>(k, j, i))
              outward_divergence -= state(m).flux(parthenon::X3DIR, mhd::IEN, k, j, i) /
                                    coords.Dxc<parthenon::X3DIR>(k, j, i);
            if (ndim >= 3 && k == kb.e && physical_outer_x3 &&
                fabs(coords.Xf<parthenon::X3DIR>(kb.e + 1, j, i) - domain_max_x3) <
                    0.25 * coords.Dxc<parthenon::X3DIR>(k, j, i))
              outward_divergence += state(m).flux(parthenon::X3DIR, mhd::IEN, k + 1, j, i) /
                                    coords.Dxc<parthenon::X3DIR>(k, j, i);
            radiation_next(m, 1, k, j, i) = gam0 * radiation_current(m, 1, k, j, i) +
                                            gam1 * radiation_base(m, 1, k, j, i) +
                                            beta_dt * outward_divergence;
          }
        }
      });
  return TaskStatus::complete;
}

void InitializeRadiationMesh(Mesh*, ParameterInput*, MeshData<Real>* data) {
  auto fields = data->PackVariables(std::vector<std::string>{
      "radiation.cooling_rate", "radiation.target_internal_energy", "radiation.cooling_mask",
      "radiation.cooling_fraction", "radiation.cooling_time", "radiation.cumulative_removed_energy",
      "radiation.cumulative_boundary_energy"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU initialize radiation diagnostics", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, 0, fields.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        fields(b, n, k, j, i) = 0.0;
      });
}

Real IntegratedField(MeshData<Real>* data, const std::string& name) {
  const auto field = data->PackVariables(std::vector<std::string>{name});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  Real total = 0.0;
  ParReduce(
      "PANGU integrated radiation diagnostic", 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        local += field(b, 0, k, j, i) * field.GetCoords(b).CellVolume(k, j, i);
      },
      Kokkos::Sum<Real>(total));
  return total;
}

Real IntegratedCoolingRate(MeshData<Real>* data) {
  return IntegratedField(data, "radiation.cooling_rate");
}

Real IntegratedRemovedEnergy(MeshData<Real>* data) {
  return IntegratedField(data, "radiation.cumulative_removed_energy");
}

Real MaximumCoolingFraction(MeshData<Real>* data) {
  const auto fraction = data->PackVariables(std::vector<std::string>{"radiation.cooling_fraction"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  Real maximum = 0.0;
  ParReduce(
      "PANGU maximum radiation cooling fraction", 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        local = fmax(local, fraction(b, 0, k, j, i));
      },
      Kokkos::Max<Real>(maximum));
  return maximum;
}

} // namespace

std::shared_ptr<StateDescriptor> InitializeCooling(ParameterInput* pin,
                                            const parthenon::Packages_t& packages) {
  PARTHENON_REQUIRE(packages.AllPackages().count("mhd") != 0,
                    "radiation_cooling requires a GRMHD problem with an mhd package");
  PARTHENON_REQUIRE(packages.Get("mhd")->Param<int>("physics_mode") == 2,
                    "radiation_cooling currently supports GRMHD only");

  auto package = std::make_shared<StateDescriptor>("radiation");
  package->AddParam<driver::StageSourceExtension>(
      std::string(driver::stage_source_extension_key),
      driver::StageSourceExtension{"target_thickness_cooling", AddCoolingStageTask});
  package->AddParam<driver::ConservedUpdateExtension>(
      std::string(driver::conserved_update_extension_key),
      driver::ConservedUpdateExtension{"radiation_energy_ledger", UpdateConservedWithLedgerMeshTask});
  package->AddParam<std::string>("backend", std::string("cooling"));
  const std::string model = pin->GetOrAddString("radiation", "model", "target_thickness");
  const Real h_target = pin->GetOrAddReal("radiation", "h_target", 0.1);
  const Real beta_cool = pin->GetOrAddReal("radiation", "beta_cool", 0.6283185307179586);
  const Real start_time = pin->GetOrAddReal("radiation", "start_time", 0.0);
  const Real ramp_time = pin->GetOrAddReal("radiation", "ramp_time", 100.0);
  const Real rho_min = pin->GetOrAddReal("radiation", "rho_min", 1.0e-6);
  const Real sigma_max = pin->GetOrAddReal("radiation", "sigma_max", 10.0);
  PARTHENON_REQUIRE(model == "target_thickness",
                    "radiation_cooling supports model=target_thickness only");
  PARTHENON_REQUIRE(h_target > 0.0 && h_target < 1.0,
                    "radiation/h_target must lie strictly between zero and one");
  PARTHENON_REQUIRE(beta_cool > 0.0, "radiation/beta_cool must be positive");
  PARTHENON_REQUIRE(
      start_time >= 0.0 && ramp_time >= 0.0 && rho_min >= 0.0 && sigma_max > 0.0,
      "radiation times and density threshold must be non-negative and sigma_max positive");
  const auto source_terms = packages.Get("source_terms");
  PARTHENON_REQUIRE(!source_terms->Param<bool>("relativistic_cooling") &&
                        !source_terms->Param<bool>("ism_cooling") &&
                        source_terms->Param<Real>("cooling_rate") == 0.0,
                    "the radiation package cannot be combined with legacy source_terms cooling");
  package->AddParam<std::string>("model", model);
  package->AddParam<Real>("h_target", h_target);
  package->AddParam<Real>("beta_cool", beta_cool);
  package->AddParam<Real>("start_time", start_time);
  package->AddParam<Real>("ramp_time", ramp_time);
  package->AddParam<Real>("rho_min", rho_min);
  package->AddParam<Real>("sigma_max", sigma_max);
  package->AddParam<bool>("bound_only", pin->GetOrAddBoolean("radiation", "bound_only", true));
  package->AddParam<bool>("track_energy", pin->GetOrAddBoolean("radiation", "track_energy", true));
  const Metadata diagnostics({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  package->AddField("radiation.cooling_rate", diagnostics);
  package->AddField("radiation.target_internal_energy", diagnostics);
  package->AddField("radiation.cooling_mask", diagnostics);
  package->AddField("radiation.cooling_fraction", diagnostics);
  package->AddField("radiation.cooling_time", diagnostics);
  package->AddField("radiation.cumulative_removed_energy",
                    Metadata({Metadata::Cell, Metadata::Independent, Metadata::Restart}));
  package->AddField("radiation.cumulative_boundary_energy",
                    Metadata({Metadata::Cell, Metadata::Independent, Metadata::Restart}));
  parthenon::HstVar_list history;
  history.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::sum, IntegratedCoolingRate, "radiation_cooling_power"));
  history.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::sum, IntegratedRemovedEnergy, "radiation_removed_energy"));
  history.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::sum,
      [](MeshData<Real>* data) {
        return IntegratedField(data, "radiation.cumulative_boundary_energy");
      },
      "radiation_boundary_energy"));
  history.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max, MaximumCoolingFraction, "radiation_max_fraction"));
  package->AddParam<>(parthenon::hist_param_key, history);
  package->PostProblemGeneratorMesh = InitializeRadiationMesh;
  return package;
}

} // namespace pangu::radiation
