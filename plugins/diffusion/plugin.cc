#include "plugin.h"

#include <limits>
#include <string>
#include <vector>

#include "hydro/hydro_types.h"
#include "mhd/mhd_types.h"
#include "pangu/plugin_api/diffusion.h"

namespace pangu::plugins::diffusion {
using namespace parthenon::package::prelude;

namespace {

template <int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION Real Offset(const Pack& values, const int component, const int k,
                                   const int j, const int i, const int offset) {
  if constexpr (Direction == 0)
    return values(component, k, j, i + offset);
  if constexpr (Direction == 1)
    return values(component, k, j + offset, i);
  return values(component, k + offset, j, i);
}

template <int Direction>
void AddConductionDirection(const std::shared_ptr<MeshBlockData<Real>>& data,
                            const std::string& prefix, const int energy_component,
                            const Real diffusivity) {
  if (diffusivity == 0.0 || energy_component < 0)
    return;
  auto block = data->GetBlockPointer();
  auto primitive = data->PackVariables(std::vector<std::string>{prefix + ".prim"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{prefix + ".cons"});
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
  const auto coordinates = block->coords;
  block->par_for(
      "PANGU conduction flux", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real density_left = Offset<Direction>(primitive, hydro::IDN, k, j, i, -1);
        const Real density_right = Offset<Direction>(primitive, hydro::IDN, k, j, i, 0);
        const Real density_face = 0.5 * (density_left + density_right);
        const Real temperature_left =
            Offset<Direction>(primitive, hydro::IPR, k, j, i, -1) / density_left;
        const Real temperature_right =
            Offset<Direction>(primitive, hydro::IPR, k, j, i, 0) / density_right;
        conserved.flux(Direction + 1, energy_component, k, j, i) -=
            diffusivity * density_face * (temperature_right - temperature_left) /
            coordinates.Dxc<Direction + 1>(k, j, i);
      });
}

TaskStatus AddFluxes(plugin_api::DiffusionOperator::BlockData& data) {
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("plugin.diffusion");
  const Real diffusivity = package->Param<Real>("alpha");
  if (diffusivity == 0.0)
    return TaskStatus::complete;
  const bool is_mhd = block->packages.AllPackages().contains("mhd");
  const auto fluid = block->packages.Get(is_mhd ? "mhd" : "hydro");
  const bool ideal = fluid->Param<int>("eos_mode") == 0;
  if (!ideal)
    return TaskStatus::complete;
  const std::string prefix = is_mhd ? "mhd" : "hydro";
  const int energy = is_mhd ? mhd::IEN : hydro::IEN;
  AddConductionDirection<0>(data, prefix, energy, diffusivity);
  if (block->pmy_mesh->ndim >= 2)
    AddConductionDirection<1>(data, prefix, energy, diffusivity);
  if (block->pmy_mesh->ndim >= 3)
    AddConductionDirection<2>(data, prefix, energy, diffusivity);
  return TaskStatus::complete;
}

Real EstimateTimestep(MeshBlockData<Real>* data) {
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("plugin.diffusion");
  const Real diffusivity = package->Param<Real>("alpha");
  if (diffusivity == 0.0)
    return std::numeric_limits<Real>::max();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  const int dimensions = block->pmy_mesh->ndim;
  Real timestep = std::numeric_limits<Real>::max();
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "PANGU conduction timestep",
      parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
        Real inverse_spacing_squared =
            1.0 / (coordinates.Dxc<X1DIR>(k, j, i) * coordinates.Dxc<X1DIR>(k, j, i));
        if (dimensions >= 2)
          inverse_spacing_squared +=
              1.0 / (coordinates.Dxc<X2DIR>(k, j, i) * coordinates.Dxc<X2DIR>(k, j, i));
        if (dimensions >= 3)
          inverse_spacing_squared +=
              1.0 / (coordinates.Dxc<X3DIR>(k, j, i) * coordinates.Dxc<X3DIR>(k, j, i));
        local = fmin(local, 0.5 / (diffusivity * inverse_spacing_squared));
      }, Kokkos::Min<Real>(timestep));
  return package->Param<Real>("cfl") * timestep;
}

} // namespace

std::shared_ptr<StateDescriptor> Plugin::Initialize(plugin::InitContext& context) {
  auto package = std::make_shared<StateDescriptor>("plugin.diffusion");
  const auto integrator = context.input.GetOrAddString("plugin/diffusion", "integrator", "unsplit");
  const Real diffusivity = context.input.GetOrAddReal("plugin/diffusion", "alpha", 0.0);
  const Real cfl = context.input.GetOrAddReal("plugin/diffusion", "cfl", 0.5);
  PARTHENON_REQUIRE(integrator == "unsplit",
                    "plugin/diffusion/integrator supports only unsplit");
  PARTHENON_REQUIRE(diffusivity >= 0.0, "plugin/diffusion/alpha must be non-negative");
  PARTHENON_REQUIRE(cfl > 0.0 && cfl <= 1.0, "plugin/diffusion/cfl must be in (0,1]");
  package->AddParam<std::string>("integrator", integrator);
  package->AddParam<Real>("alpha", diffusivity);
  package->AddParam<Real>("cfl", cfl);
  package->AddParam<plugin_api::DiffusionOperator>(
      std::string(plugin_api::diffusion_operator_key),
      plugin_api::DiffusionOperator{"conduction", AddFluxes});
  package->EstimateTimestepBlock = EstimateTimestep;
  return package;
}

} // namespace pangu::plugins::diffusion
