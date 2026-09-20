#include "plugin.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "hydro/hydro_types.h"
#include "mhd/mhd_types.h"
#include "pangu/plugin_api/diffusion.h"

namespace pangu::plugins::viscosity {
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

template <int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION Real TransverseDerivative(const Pack& values, const int component,
                                                 const int transverse, const int k, const int j,
                                                 const int i, const Real spacing) {
  int kp = k, km = k, jp = j, jm = j, ip = i, im = i;
  if (transverse == 0) {
    ++ip;
    --im;
  } else if (transverse == 1) {
    ++jp;
    --jm;
  } else {
    ++kp;
    --km;
  }
  int lkp = kp, lkm = km, ljp = jp, ljm = jm, lip = ip, lim = im;
  if constexpr (Direction == 0) {
    --lip;
    --lim;
  } else if constexpr (Direction == 1) {
    --ljp;
    --ljm;
  } else {
    --lkp;
    --lkm;
  }
  return (values(component, kp, jp, ip) + values(component, lkp, ljp, lip) -
          values(component, km, jm, im) - values(component, lkm, ljm, lim)) /
         (4.0 * spacing);
}

template <int Direction>
void AddDirection(const std::shared_ptr<MeshBlockData<Real>>& data, const std::string& prefix,
                  const int energy_component, const Real viscosity) {
  auto block = data->GetBlockPointer();
  auto primitive = data->PackVariables(std::vector<std::string>{prefix + ".prim"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{prefix + ".cons"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const int dimensions = block->pmy_mesh->ndim;
  int il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  if constexpr (Direction == 0)
    ++iu;
  if constexpr (Direction == 1)
    ++ju;
  if constexpr (Direction == 2)
    ++ku;
  const auto coordinates = block->coords;
  block->par_for(
      "PANGU viscosity flux", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real normal_spacing = coordinates.Dxc<Direction + 1>(k, j, i);
        const Real density = 0.5 * (Offset<Direction>(primitive, hydro::IDN, k, j, i, -1) +
                                    Offset<Direction>(primitive, hydro::IDN, k, j, i, 0));
        Real stress[3]{};
        Real divergence = 0.0;
        for (int axis = 0; axis < dimensions; ++axis) {
          const Real spacing = axis == 0   ? coordinates.Dxc<X1DIR>(k, j, i)
                               : axis == 1 ? coordinates.Dxc<X2DIR>(k, j, i)
                                           : coordinates.Dxc<X3DIR>(k, j, i);
          divergence += axis == Direction
                            ? (Offset<Direction>(primitive, hydro::IV1 + axis, k, j, i, 0) -
                               Offset<Direction>(primitive, hydro::IV1 + axis, k, j, i, -1)) /
                                  normal_spacing
                            : TransverseDerivative<Direction>(primitive, hydro::IV1 + axis, axis,
                                                              k, j, i, spacing);
        }
        for (int component = 0; component < 3; ++component) {
          const Real normal =
              (Offset<Direction>(primitive, hydro::IV1 + component, k, j, i, 0) -
               Offset<Direction>(primitive, hydro::IV1 + component, k, j, i, -1)) /
              normal_spacing;
          Real transpose = normal;
          if (component != Direction) {
            transpose = 0.0;
            if (component < dimensions) {
              const Real spacing = component == 0   ? coordinates.Dxc<X1DIR>(k, j, i)
                                   : component == 1 ? coordinates.Dxc<X2DIR>(k, j, i)
                                                    : coordinates.Dxc<X3DIR>(k, j, i);
              transpose = TransverseDerivative<Direction>(
                  primitive, hydro::IV1 + Direction, component, k, j, i, spacing);
            }
          }
          stress[component] = normal + transpose;
          if (component == Direction)
            stress[component] -= (2.0 / 3.0) * divergence;
          conserved.flux(Direction + 1, hydro::IM1 + component, k, j, i) -=
              viscosity * density * stress[component];
        }
        if (energy_component >= 0) {
          Real work = 0.0;
          for (int component = 0; component < 3; ++component) {
            const Real velocity =
                0.5 * (Offset<Direction>(primitive, hydro::IV1 + component, k, j, i, -1) +
                       Offset<Direction>(primitive, hydro::IV1 + component, k, j, i, 0));
            work += velocity * stress[component];
          }
          conserved.flux(Direction + 1, energy_component, k, j, i) -=
              viscosity * density * work;
        }
      });
}

TaskStatus AddFluxes(plugin_api::DiffusionOperator::BlockData& data) {
  auto block = data->GetBlockPointer();
  const auto viscosity = block->packages.Get("plugin.viscosity")->Param<Real>("nu");
  const bool is_mhd = block->packages.AllPackages().contains("mhd");
  const auto fluid = block->packages.Get(is_mhd ? "mhd" : "hydro");
  const bool ideal = fluid->Param<int>("eos_mode") == 0;
  const std::string prefix = is_mhd ? "mhd" : "hydro";
  const int energy = ideal ? (is_mhd ? mhd::IEN : hydro::IEN) : -1;
  AddDirection<0>(data, prefix, energy, viscosity);
  if (block->pmy_mesh->ndim >= 2)
    AddDirection<1>(data, prefix, energy, viscosity);
  if (block->pmy_mesh->ndim >= 3)
    AddDirection<2>(data, prefix, energy, viscosity);
  return TaskStatus::complete;
}

Real EstimateTimestep(MeshBlockData<Real>* data) {
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("plugin.viscosity");
  const Real viscosity = package->Param<Real>("nu");
  if (viscosity == 0.0)
    return std::numeric_limits<Real>::max();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  const int dimensions = block->pmy_mesh->ndim;
  Real timestep = std::numeric_limits<Real>::max();
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "PANGU viscosity timestep",
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
        local = fmin(local, 0.5 / (viscosity * inverse_spacing_squared));
      },
      Kokkos::Min<Real>(timestep));
  return package->Param<Real>("cfl") * timestep;
}

} // namespace

std::shared_ptr<StateDescriptor> Plugin::Initialize(plugin::InitContext& context) {
  auto package = std::make_shared<StateDescriptor>("plugin.viscosity");
  const Real viscosity = context.input.GetOrAddReal("plugin/viscosity", "nu", 0.0);
  const Real cfl = context.input.GetOrAddReal("plugin/viscosity", "cfl", 0.5);
  PARTHENON_REQUIRE(viscosity >= 0.0, "plugin/viscosity/nu must be non-negative");
  PARTHENON_REQUIRE(cfl > 0.0 && cfl <= 1.0, "plugin/viscosity/cfl must be in (0,1]");
  package->AddParam<Real>("nu", viscosity);
  package->AddParam<Real>("cfl", cfl);
  package->AddParam<plugin_api::DiffusionOperator>(
      std::string(plugin_api::diffusion_operator_key),
      plugin_api::DiffusionOperator{"viscosity", AddFluxes});
  package->EstimateTimestepBlock = EstimateTimestep;
  return package;
}

} // namespace pangu::plugins::viscosity
