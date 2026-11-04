#include "plugin.h"
#include "bns_initial_data.h"

#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "geometry_assembly.h"
#include "mhd/mhd_package.h"
#include "mhd/mhd_types.h"
#include "pangu/plugin_api/initial_data.h"
#include "pgen/numerical_relativity.h"
#include "z4c/coupling/sync_grmhd.h"
#include "z4c/evolution/package.h"

namespace pangu::plugins::compact_objects {
namespace {
using namespace parthenon::package::prelude;
constexpr const char *label = "plugin.compact_objects";

struct Plasma {
  Real density = 1, pressure = .2;
  Real btilde[3] = {0, 0, .1};
};

void InitializePlasma(MeshBlock *block, ParameterInput *input) {
  PARTHENON_REQUIRE(block->pmy_mesh->ndim == 3,
                    "puncture_plasma requires a three-dimensional mesh");
  // The native puncture solver supplies the metric, including its constraint
  // solve and gauge conversion. The plasma is initialized after ADM exists.
  pgen::NumericalRelativityPuncture(block, input);
  auto data = block->meshblock_data.Get();
  const auto model = block->packages.Get(label)->Param<Plasma>("plasma");
  const auto eos = pangu::eos::ReadRelativistic(*block->packages.Get("mhd"));
  auto prim = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto divb = data->PackVariables(std::vector<std::string>{"mhd.divb"});
  auto fofc = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto recovery = data->PackVariables(std::vector<std::string>{"mhd.recovery"});
  auto bface = data->Get("mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  block->par_for("compact-object initial plasma", kb.s, kb.e, jb.s, jb.e,
                 ib.s, ib.e, KOKKOS_LAMBDA(int k, int j, int i) {
    prim(mhd::IDN, k, j, i) = model.density;
    prim(mhd::IPR, k, j, i) = eos.InternalEnergyDensity(model.pressure);
    for (int axis = 0; axis < 3; ++axis) {
      prim(mhd::IV1 + axis, k, j, i) = 0;
      bcell(axis, k, j, i) = model.btilde[axis];
      recovery(axis, k, j, i) = 0;
    }
    divb(0, k, j, i) = 0;
    fofc(0, k, j, i) = 0;
  });
  // Constant densitized B is the curl of a linear vector potential. It gives
  // exactly zero discrete divergence, including through AMR prolongation.
  // Physical Eulerian B equals Btilde/sqrt(gamma), not Btilde near a hole.
  block->par_for("compact-object CT B1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
                 KOKKOS_LAMBDA(int k, int j, int i) {
    bface(0, 0, 0, 0, k, j, i) = model.btilde[0];
  });
  block->par_for("compact-object CT B2", kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
                 KOKKOS_LAMBDA(int k, int j, int i) {
    bface(1, 0, 0, 0, k, j, i) = model.btilde[1];
  });
  block->par_for("compact-object CT B3", kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(int k, int j, int i) {
    bface(2, 0, 0, 0, k, j, i) = model.btilde[2];
  });
  nr::SyncGRMHDPrimitiveToConservedBlock(data.get());
  nr::Z4cToADMBlockTask(data.get());
}
} // namespace

std::shared_ptr<parthenon::StateDescriptor> Plugin::Initialize(plugin::InitContext &context) {
  auto &input = context.input;
  const std::string section = "plugin/compact_objects";
  if (input.GetOrAddString(section, "model", "puncture_plasma") == "bns_time_symmetric")
    return InitializeBNS(context);
  const std::set<std::string> keys{"model", "density", "pressure", "btilde1", "btilde2", "btilde3"};
  for (const auto &key : input.GetParameterNames(section))
    PARTHENON_REQUIRE(keys.count(key) != 0, "unknown compact_objects parameter: " + key);
  PARTHENON_REQUIRE(input.GetString("parthenon/job", "problem_id") == "sync_plugin_grmhd",
                    "compact_objects requires problem_id=sync_plugin_grmhd");
  PARTHENON_REQUIRE(geometry::ConfiguredGeometryIsSynchronized(),
                    "compact_objects requires METRIC=z4c MODE=sync");
  PARTHENON_REQUIRE(input.GetOrAddString(section, "model", "puncture_plasma") == "puncture_plasma",
                    "compact_objects currently implements puncture_plasma");
  PARTHENON_REQUIRE(context.dependencies.AllPackages().contains("numerical_relativity") &&
                        context.dependencies.AllPackages().contains("mhd"),
                    "compact_objects requires Z4c and MHD packages");
  // Kelly et al. arXiv:1710.02132 neglect the plasma contribution to Einstein's
  // equations. Their independent rho0=1 normalization cannot be used as a
  // gravitating density in the M_total=1 geometric units.
  PARTHENON_REQUIRE(context.dependencies.Get("numerical_relativity")
                        ->Param<std::string>("matter_source_name") == "zero",
                    "puncture_plasma benchmark requires matter_source=zero (negligible plasma self-gravity)");
  const auto mhd_package = context.dependencies.Get("mhd");
  const auto eos = pangu::eos::ReadRelativistic(*mhd_package);
  PARTHENON_REQUIRE(mhd_package->Param<int>("physics_mode") ==
                        static_cast<int>(relativity::HydroMode::gr),
                    "puncture_plasma requires mhd/physics=gr");
  Plasma p;
  p.density = input.GetOrAddReal(section, "density", 1);
  p.pressure = input.GetOrAddReal(section, "pressure", .2);
  for (int axis = 0; axis < 3; ++axis)
    p.btilde[axis] = input.GetOrAddReal(section, "btilde" + std::to_string(axis + 1),
                                     axis == 2 ? .1 : 0);
  PARTHENON_REQUIRE(std::isfinite(p.density) && std::isfinite(p.pressure) &&
                        p.density >= eos.density_floor && p.pressure >= eos.pressure_floor,
                    "plasma density/pressure must be finite and respect the evolution floors");
  for (Real value : p.btilde)
    PARTHENON_REQUIRE(std::isfinite(value), "nonfinite plasma magnetic field");
  auto package = std::make_shared<StateDescriptor>(label);
  package->AddParam<Plasma>("plasma", p);
  package->AddParam<plugin_api::BlockInitialData>(plugin_api::block_initial_data_key,
                                                InitializePlasma);
  return package;
}
} // namespace pangu::plugins::compact_objects
