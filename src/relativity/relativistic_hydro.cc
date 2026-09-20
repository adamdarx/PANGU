#include "relativity/relativistic_hydro.h"

// Relativistic conserved variables, wave speeds, and the Galeazzi/Kastaun
// inversion follow AthenaK's BSD-3-Clause fixed-background Hydro path.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "estimator/estimator.h"
#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "pangu.h"
#include "reconstruct/hydro_reconstruction.h"
#include "riemann/registry.h"

namespace pangu::relativity {
using namespace parthenon::package::prelude;
using Estimator = estimator::Selected;

namespace {

// First-order flux correction replaces flagged faces with the registered fallback solver.
using FirstOrderSolver =
    riemann::Implementation<riemann::Physics::relativistic_hydro, riemann::first_order_fallback>;

struct RHDFluxState {
  Real conserved[5]{};
  Real flux[5]{};
  Real lambda_plus = 0.0;
  Real lambda_minus = 0.0;
};

template <int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION Real ReadBlock(const Pack& pack, const int component, const int k,
                                      const int j, const int i, const int offset) {
  if constexpr (Direction == 0)
    return pack(component, k, j, i + offset);
  if constexpr (Direction == 1)
    return pack(component, k, j + offset, i);
  return pack(component, k + offset, j, i);
}

template <hydro::Reconstruction Method, int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION void
ReconstructFaceStatesBlock(const Pack& primitive, const int k, const int j, const int i,
                           const eos::RelativisticEOS& eos, HydroPrimitiveState& left,
                           HydroPrimitiveState& right) {
  Real values_left[5];
  Real values_right[5];
  for (int n = 0; n < 5; ++n) {
    reconstruct::ReconstructFaceStencil<Method>(
        [&](const int offset) { return ReadBlock<Direction>(primitive, n, k, j, i, offset); },
        values_left[n], values_right[n]);
  }
  left = {fmax(values_left[hydro::IDN], eos.density_floor),
          {values_left[hydro::IV1], values_left[hydro::IV2], values_left[hydro::IV3]},
          fmax(values_left[hydro::IPR], eos.pressure_floor)};
  right = {fmax(values_right[hydro::IDN], eos.density_floor),
           {values_right[hydro::IV1], values_right[hydro::IV2], values_right[hydro::IV3]},
           fmax(values_right[hydro::IPR], eos.pressure_floor)};
}

template <int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION Real ReadMesh(const Pack& pack, const int block, const int component,
                                     const int k, const int j, const int i, const int offset) {
  if constexpr (Direction == 0)
    return pack(block, component, k, j, i + offset);
  if constexpr (Direction == 1)
    return pack(block, component, k, j + offset, i);
  return pack(block, component, k + offset, j, i);
}

template <hydro::Reconstruction Method, int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION void
ReconstructFaceStatesMesh(const Pack& primitive, const int block, const int k, const int j,
                          const int i, const eos::RelativisticEOS& eos, HydroPrimitiveState& left,
                          HydroPrimitiveState& right) {
  Real values_left[5];
  Real values_right[5];
  for (int n = 0; n < 5; ++n) {
    reconstruct::ReconstructFaceStencil<Method>(
        [&](const int offset) {
          return ReadMesh<Direction>(primitive, block, n, k, j, i, offset);
        },
        values_left[n], values_right[n]);
  }
  left = {fmax(values_left[hydro::IDN], eos.density_floor),
          {values_left[hydro::IV1], values_left[hydro::IV2], values_left[hydro::IV3]},
          fmax(values_left[hydro::IPR], eos.pressure_floor)};
  right = {fmax(values_right[hydro::IDN], eos.density_floor),
           {values_right[hydro::IV1], values_right[hydro::IV2], values_right[hydro::IV3]},
           fmax(values_right[hydro::IPR], eos.pressure_floor)};
}

KOKKOS_INLINE_FUNCTION void SRSoundSpeeds(const HydroPrimitiveState& primitive,
                                          const eos::RelativisticEOS& eos, const int direction,
                                          Real& plus, Real& minus) {
  const Real lorentz = sqrt(1.0 + primitive.u[0] * primitive.u[0] +
                            primitive.u[1] * primitive.u[1] + primitive.u[2] * primitive.u[2]);
  const Real cs2 = eos.SoundSpeedSquared(primitive.density, primitive.pressure);
  const Real velocity2 = 1.0 - 1.0 / (lorentz * lorentz);
  const Real normal_velocity = primitive.u[direction] / lorentz;
  const Real p1 = normal_velocity * (1.0 - cs2);
  const Real root =
      sqrt(fmax(0.0, cs2 * ((1.0 - velocity2 * cs2) - p1 * normal_velocity))) / lorentz;
  const Real inverse = 1.0 / (1.0 - velocity2 * cs2);
  plus = (p1 + root) * inverse;
  minus = (p1 - root) * inverse;
}

KOKKOS_INLINE_FUNCTION RHDFluxState BuildSRHDFluxState(const HydroPrimitiveState& primitive,
                                                       const eos::RelativisticEOS& eos,
                                                       const int direction) {
  RHDFluxState state{};
  const auto conserved = ConvertSRHDP2C(primitive, eos);
  state.conserved[hydro::IDN] = conserved.density;
  state.conserved[hydro::IM1] = conserved.momentum[0];
  state.conserved[hydro::IM2] = conserved.momentum[1];
  state.conserved[hydro::IM3] = conserved.momentum[2];
  state.conserved[hydro::IEN] = conserved.energy;
  const Real lorentz = sqrt(1.0 + primitive.u[0] * primitive.u[0] +
                            primitive.u[1] * primitive.u[1] + primitive.u[2] * primitive.u[2]);
  const Real enthalpy_un =
      eos.EnthalpyDensity(primitive.density, primitive.pressure) * primitive.u[direction];
  state.flux[hydro::IDN] = primitive.density * primitive.u[direction];
  for (int axis = 0; axis < 3; ++axis)
    state.flux[hydro::IM1 + axis] =
        enthalpy_un * primitive.u[axis] + (axis == direction ? primitive.pressure : 0.0);
  state.flux[hydro::IEN] = enthalpy_un * lorentz - state.flux[hydro::IDN];
  SRSoundSpeeds(primitive, eos, direction, state.lambda_plus, state.lambda_minus);
  return state;
}

KOKKOS_INLINE_FUNCTION RHDFluxState BuildGRHDFluxState(const HydroPrimitiveState& primitive,
                                                       const eos::RelativisticEOS& eos,
                                                       const int direction,
                                                       const geometry::MetricPoint& metric) {
  RHDFluxState state{};
  Real spatial_u2 = 0.0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      spatial_u2 += metric.lower[a + 1][b + 1] * primitive.u[a] * primitive.u[b];
  const Real lorentz = sqrt(1.0 + spatial_u2);
  Real upper[4]{lorentz / metric.lapse, 0.0, 0.0, 0.0};
  for (int axis = 0; axis < 3; ++axis)
    upper[axis + 1] = primitive.u[axis] - metric.lapse * lorentz * metric.upper[0][axis + 1];
  Real lower[4]{};
  for (int mu = 0; mu < 4; ++mu)
    for (int nu = 0; nu < 4; ++nu)
      lower[mu] += metric.lower[mu][nu] * upper[nu];
  const Real enthalpy = eos.EnthalpyDensity(primitive.density, primitive.pressure);
  state.conserved[hydro::IDN] = primitive.density * upper[0];
  for (int axis = 0; axis < 3; ++axis)
    state.conserved[hydro::IM1 + axis] = enthalpy * upper[0] * lower[axis + 1];
  state.conserved[hydro::IEN] =
      enthalpy * upper[0] * lower[0] + primitive.pressure + state.conserved[hydro::IDN];
  const int normal = direction + 1;
  state.flux[hydro::IDN] = primitive.density * upper[normal];
  for (int axis = 0; axis < 3; ++axis)
    state.flux[hydro::IM1 + axis] =
        enthalpy * upper[normal] * lower[axis + 1] + (axis == direction ? primitive.pressure : 0.0);
  state.flux[hydro::IEN] = enthalpy * upper[normal] * lower[0] + state.flux[hydro::IDN];
  for (int n = 0; n < 5; ++n) {
    state.conserved[n] *= metric.gdet;
    state.flux[n] *= metric.gdet;
  }
  const Real cs2 = eos.SoundSpeedSquared(primitive.density, primitive.pressure);
  const Real a = upper[0] * upper[0] - (metric.upper[0][0] + upper[0] * upper[0]) * cs2;
  const Real b = -2.0 * (upper[0] * upper[normal] -
                         (metric.upper[0][normal] + upper[0] * upper[normal]) * cs2);
  const Real c = upper[normal] * upper[normal] -
                 (metric.upper[normal][normal] + upper[normal] * upper[normal]) * cs2;
  const Real discriminant = fmax(0.0, b * b - 4.0 * a * c);
  const Real root1 = (-b + sqrt(discriminant)) / (2.0 * a);
  const Real root2 = (-b - sqrt(discriminant)) / (2.0 * a);
  state.lambda_plus = fmax(root1, root2);
  state.lambda_minus = fmin(root1, root2);
  return state;
}

template <int Direction, typename PrimitivePack, typename ConservedPack, typename FlagPack>
void ReplaceFlaggedFacesBlock(MeshBlock* block, const PrimitivePack& primitive,
                              ConservedPack& conserved, const FlagPack& flags, const HydroMode mode,
                              const eos::RelativisticEOS& eos, const IndexRange& flag_i,
                              const IndexRange& flag_j, const IndexRange& flag_k) {
  int il = flag_i.s, iu = flag_i.e, jl = flag_j.s, ju = flag_j.e;
  int kl = flag_k.s, ku = flag_k.e;
  if constexpr (Direction == 0)
    ++iu;
  if constexpr (Direction == 1)
    ++ju;
  if constexpr (Direction == 2)
    ++ku;
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  block->par_for(
      "PANGU relativistic Hydro FOFC face", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int lk = k - (Direction == 2);
        const int lj = j - (Direction == 1);
        const int li = i - (Direction == 0);
        const bool has_left = Direction == 0   ? i > flag_i.s
                              : Direction == 1 ? j > flag_j.s
                                               : k > flag_k.s;
        const bool has_right = Direction == 0   ? i <= flag_i.e
                               : Direction == 1 ? j <= flag_j.e
                                                : k <= flag_k.e;
        if (!((has_left && flags(0, lk, lj, li) > 0.5) || (has_right && flags(0, k, j, i) > 0.5)))
          return;
        const HydroPrimitiveState left{primitive(hydro::IDN, lk, lj, li),
                                       {primitive(hydro::IV1, lk, lj, li),
                                        primitive(hydro::IV2, lk, lj, li),
                                        primitive(hydro::IV3, lk, lj, li)},
                                       primitive(hydro::IPR, lk, lj, li)};
        const HydroPrimitiveState right{primitive(hydro::IDN, k, j, i),
                                        {primitive(hydro::IV1, k, j, i),
                                         primitive(hydro::IV2, k, j, i),
                                         primitive(hydro::IV3, k, j, i)},
                                        primitive(hydro::IPR, k, j, i)};
        Real flux[5]{};
        if (mode == HydroMode::sr) {
          FirstOrderSolver::Solve(BuildSRHDFluxState(left, eos, Direction),
                                  BuildSRHDFluxState(right, eos, Direction), flux);
        } else {
          const auto metric = spacetime.MetricAt(geometry::FaceLocation(Direction), geometry_block,
                                                 k, j, i, coords);
          FirstOrderSolver::Solve(BuildGRHDFluxState(left, eos, Direction, metric),
                                  BuildGRHDFluxState(right, eos, Direction, metric), flux);
        }
        for (int n = 0; n < 5; ++n)
          conserved.flux(Direction + 1, n, k, j, i) = flux[n];
      });
}

template <int Direction, typename PrimitivePack, typename ConservedPack, typename FlagPack>
void ReplaceFlaggedFacesMesh(MeshData<Real>* data, const PrimitivePack& primitive,
                             ConservedPack& conserved, const FlagPack& flags, const HydroMode mode,
                             const eos::RelativisticEOS& eos, const IndexRange& flag_i,
                             const IndexRange& flag_j, const IndexRange& flag_k) {
  int il = flag_i.s, iu = flag_i.e, jl = flag_j.s, ju = flag_j.e;
  int kl = flag_k.s, ku = flag_k.e;
  if constexpr (Direction == 0)
    ++iu;
  if constexpr (Direction == 1)
    ++ju;
  if constexpr (Direction == 2)
    ++ku;
  const auto spacetime = geometry::GetGeometry(data->GetParentPointer()->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed relativistic Hydro FOFC face", parthenon::DevExecSpace(),
      0, data->NumBlocks() - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const int lk = k - (Direction == 2);
        const int lj = j - (Direction == 1);
        const int li = i - (Direction == 0);
        const bool has_left = Direction == 0   ? i > flag_i.s
                              : Direction == 1 ? j > flag_j.s
                                               : k > flag_k.s;
        const bool has_right = Direction == 0   ? i <= flag_i.e
                               : Direction == 1 ? j <= flag_j.e
                                                : k <= flag_k.e;
        if (!((has_left && flags(b, 0, lk, lj, li) > 0.5) ||
              (has_right && flags(b, 0, k, j, i) > 0.5)))
          return;
        const HydroPrimitiveState left{primitive(b, hydro::IDN, lk, lj, li),
                                       {primitive(b, hydro::IV1, lk, lj, li),
                                        primitive(b, hydro::IV2, lk, lj, li),
                                        primitive(b, hydro::IV3, lk, lj, li)},
                                       primitive(b, hydro::IPR, lk, lj, li)};
        const HydroPrimitiveState right{primitive(b, hydro::IDN, k, j, i),
                                        {primitive(b, hydro::IV1, k, j, i),
                                         primitive(b, hydro::IV2, k, j, i),
                                         primitive(b, hydro::IV3, k, j, i)},
                                        primitive(b, hydro::IPR, k, j, i)};
        Real flux[5]{};
        if (mode == HydroMode::sr) {
          FirstOrderSolver::Solve(BuildSRHDFluxState(left, eos, Direction),
                                  BuildSRHDFluxState(right, eos, Direction), flux);
        } else {
          const auto& coords = primitive.GetCoords(b);
          const auto metric = spacetime.MetricAt(geometry::FaceLocation(Direction),
                                                 geometry_block_offset + b, k, j, i, coords);
          FirstOrderSolver::Solve(BuildGRHDFluxState(left, eos, Direction, metric),
                                  BuildGRHDFluxState(right, eos, Direction, metric), flux);
        }
        auto block_conserved = conserved(b);
        for (int n = 0; n < 5; ++n)
          block_conserved.flux(Direction + 1, n, k, j, i) = flux[n];
      });
}

template <hydro::Reconstruction Method, riemann::Solver Solver, int Direction>
void CalculateDirectionBlock(const std::shared_ptr<MeshBlockData<Real>>& data, const HydroMode mode,
                             const eos::RelativisticEOS& eos) {
  auto block = data->GetBlockPointer();
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"hydro.cons"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  const int ndim = block->pmy_mesh->ndim;
  const bool replace_first_order = block->packages.Get("hydro")->Param<bool>("fofc");
  int il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  if constexpr (Direction == 0) {
    ++iu;
    if (replace_first_order) {
      --il;
      ++iu;
    }
    if (replace_first_order) {
      if (ndim >= 2) {
        --jl;
        ++ju;
      }
      if (ndim >= 3) {
        --kl;
        ++ku;
      }
    }
  }
  if constexpr (Direction == 1) {
    ++ju;
    if (replace_first_order) {
      --jl;
      ++ju;
    }
    if (replace_first_order) {
      --il;
      ++iu;
      if (ndim >= 3) {
        --kl;
        ++ku;
      }
    }
  }
  if constexpr (Direction == 2) {
    ++ku;
    if (replace_first_order) {
      --kl;
      ++ku;
    }
    if (replace_first_order) {
      --jl;
      ++ju;
      --il;
      ++iu;
    }
  }
  block->par_for(
      "PANGU relativistic Hydro flux", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        HydroPrimitiveState left{}, right{};
        ReconstructFaceStatesBlock<Method, Direction>(primitive, k, j, i, eos, left, right);
        Real flux[5]{};
        if (mode == HydroMode::sr) {
          riemann::Implementation<riemann::Physics::relativistic_hydro, Solver>::Solve(
              BuildSRHDFluxState(left, eos, Direction),
              BuildSRHDFluxState(right, eos, Direction), flux);
        } else {
          const auto metric = spacetime.MetricAt(geometry::FaceLocation(Direction), geometry_block,
                                                 k, j, i, coords);
          riemann::Implementation<riemann::Physics::relativistic_hydro, Solver>::Solve(
              BuildGRHDFluxState(left, eos, Direction, metric),
              BuildGRHDFluxState(right, eos, Direction, metric), flux);
        }
        for (int n = 0; n < 5; ++n)
          conserved.flux(Direction + 1, n, k, j, i) = flux[n];
      });
}

template <hydro::Reconstruction Method, riemann::Solver Solver, int Direction>
void CalculateDirectionMesh(MeshData<Real>* data, const HydroMode mode,
                            const eos::RelativisticEOS& eos) {
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"hydro.cons"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int ndim = data->GetParentPointer()->ndim;
  const auto spacetime = geometry::GetGeometry(data->GetParentPointer()->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  const bool replace_first_order =
      data->GetParentPointer()->packages.Get("hydro")->Param<bool>("fofc");
  int il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  if constexpr (Direction == 0) {
    ++iu;
    if (replace_first_order) {
      --il;
      ++iu;
    }
    if (replace_first_order) {
      if (ndim >= 2) {
        --jl;
        ++ju;
      }
      if (ndim >= 3) {
        --kl;
        ++ku;
      }
    }
  }
  if constexpr (Direction == 1) {
    ++ju;
    if (replace_first_order) {
      --jl;
      ++ju;
    }
    if (replace_first_order) {
      --il;
      ++iu;
      if (ndim >= 3) {
        --kl;
        ++ku;
      }
    }
  }
  if constexpr (Direction == 2) {
    ++ku;
    if (replace_first_order) {
      --kl;
      ++ku;
    }
    if (replace_first_order) {
      --jl;
      ++ju;
      --il;
      ++iu;
    }
  }
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed relativistic Hydro flux", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        HydroPrimitiveState left{}, right{};
        ReconstructFaceStatesMesh<Method, Direction>(primitive, b, k, j, i, eos, left, right);
        const auto& coords = primitive.GetCoords(b);
        Real flux[5]{};
        if (mode == HydroMode::sr) {
          riemann::Implementation<riemann::Physics::relativistic_hydro, Solver>::Solve(
              BuildSRHDFluxState(left, eos, Direction),
              BuildSRHDFluxState(right, eos, Direction), flux);
        } else {
          const auto metric = spacetime.MetricAt(geometry::FaceLocation(Direction),
                                                 geometry_block_offset + b, k, j, i, coords);
          riemann::Implementation<riemann::Physics::relativistic_hydro, Solver>::Solve(
              BuildGRHDFluxState(left, eos, Direction, metric),
              BuildGRHDFluxState(right, eos, Direction, metric), flux);
        }
        auto block_conserved = conserved(b);
        for (int n = 0; n < 5; ++n)
          block_conserved.flux(Direction + 1, n, k, j, i) = flux[n];
      });
}

template <hydro::Reconstruction Method, riemann::Solver Solver>
void CalculateAllDirectionsBlock(const std::shared_ptr<MeshBlockData<Real>>& data,
                                 const HydroMode mode, const eos::RelativisticEOS& eos) {
  auto block = data->GetBlockPointer();
  CalculateDirectionBlock<Method, Solver, 0>(data, mode, eos);
  if (block->pmy_mesh->ndim >= 2)
    CalculateDirectionBlock<Method, Solver, 1>(data, mode, eos);
  if (block->pmy_mesh->ndim >= 3)
    CalculateDirectionBlock<Method, Solver, 2>(data, mode, eos);
}

template <hydro::Reconstruction Method, riemann::Solver Solver>
void CalculateAllDirectionsMesh(MeshData<Real>* data, const HydroMode mode,
                                const eos::RelativisticEOS& eos) {
  const auto mesh = data->GetParentPointer();
  CalculateDirectionMesh<Method, Solver, 0>(data, mode, eos);
  if (mesh->ndim >= 2)
    CalculateDirectionMesh<Method, Solver, 1>(data, mode, eos);
  if (mesh->ndim >= 3)
    CalculateDirectionMesh<Method, Solver, 2>(data, mode, eos);
}

} // namespace

void InitializeExcisionMasksMesh(Mesh* mesh, ParameterInput*, MeshData<Real>* data) {
  if constexpr (geometry::ConfiguredMetricSupportsExcision()) {
    const auto hydro = mesh->packages.Get("hydro");
    const auto geometry_package = mesh->packages.Get("geometry");
    const bool enabled =
        static_cast<HydroMode>(hydro->Param<int>("physics_mode")) == HydroMode::gr &&
        geometry_package->Param<bool>("excision");
    const Real radius = geometry_package->Param<Real>("excision_radius");
    const auto spacetime = geometry::GetGeometry(mesh->packages);
    auto masks = data->PackVariables(std::vector<std::string>{"hydro.excision"});
    const auto ib = data->GetBoundsI(IndexDomain::entire);
    const auto jb = data->GetBoundsJ(IndexDomain::entire);
    const auto kb = data->GetBoundsK(IndexDomain::entire);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU fixed-GR excision masks", parthenon::DevExecSpace(), 0,
        data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          bool floor = false;
          bool flux = false;
          if (enabled) {
            const auto& coords = masks.GetCoords(b);
            floor = spacetime.IsExcised(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k), radius);
            flux = spacetime.NeedsFluxExcision(coords, k, j, i, radius);
          }
          masks(b, 0, k, j, i) = floor ? 1.0 : 0.0;
          masks(b, 1, k, j, i) = flux ? 1.0 : 0.0;
        });
  }
}

TaskStatus CalculateFluxesBlockTask(std::shared_ptr<MeshBlockData<Real>>& data) {
  auto block = data->GetBlockPointer();
  const auto hydro = block->packages.Get("hydro");
  const auto mode = static_cast<HydroMode>(hydro->Param<int>("physics_mode"));
  const auto eos = pangu::eos::ReadRelativistic(*hydro);
  const auto reconstruction =
      static_cast<hydro::Reconstruction>(hydro->Param<int>("reconstruction"));
  const auto solver = static_cast<riemann::Solver>(hydro->Param<int>("riemann"));
  reconstruct::Visit(reconstruction, [&]<hydro::Reconstruction Method>() {
    riemann::Visit<riemann::Physics::relativistic_hydro>(solver, [&]<riemann::Solver Solver>() {
      CalculateAllDirectionsBlock<Method, Solver>(data, mode, eos);
    });
  });
  return TaskStatus::complete;
}

TaskStatus CalculateFluxesMeshTask(MeshData<Real>* data) {
  const auto mesh = data->GetParentPointer();
  const auto hydro = mesh->packages.Get("hydro");
  const auto mode = static_cast<HydroMode>(hydro->Param<int>("physics_mode"));
  PARTHENON_REQUIRE(mode != HydroMode::newtonian,
                    "Relativistic Hydro packed flux dispatch requires SR or GR mode");
  const auto eos = pangu::eos::ReadRelativistic(*hydro);
  const auto reconstruction =
      static_cast<hydro::Reconstruction>(hydro->Param<int>("reconstruction"));
  const auto solver = static_cast<riemann::Solver>(hydro->Param<int>("riemann"));
  reconstruct::Visit(reconstruction, [&]<hydro::Reconstruction Method>() {
    riemann::Visit<riemann::Physics::relativistic_hydro>(solver, [&]<riemann::Solver Solver>() {
      CalculateAllDirectionsMesh<Method, Solver>(data, mode, eos);
    });
  });
  return TaskStatus::complete;
}

TaskStatus ApplyHydroFluxCorrectionBlockTask(std::shared_ptr<MeshBlockData<Real>>& data,
                                             std::shared_ptr<MeshBlockData<Real>>& base,
                                             const Real gam0, const Real gam1, const Real beta_dt) {
  const Real dt = beta_dt;
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("hydro");
  const auto mode = static_cast<HydroMode>(package->Param<int>("physics_mode"));
  if (mode == HydroMode::newtonian)
    return TaskStatus::complete;
  const bool use_fofc = package->Param<bool>("fofc");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  const Real gamma_max = package->Param<Real>("gamma_max");
  const auto geometry_package = block->packages.Get("geometry");
  const bool excision = geometry_package->Param<bool>("excision");
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"hydro.cons"});
  const auto base_conserved = base->PackVariables(std::vector<std::string>{"hydro.cons"});
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto flags = data->PackVariables(std::vector<std::string>{"hydro.fofc"});
  parthenon::VariablePack<Real> excision_masks;
  if constexpr (geometry::ConfiguredMetricSupportsExcision())
    excision_masks = data->PackVariables(std::vector<std::string>{"hydro.excision"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const int ndim = block->pmy_mesh->ndim;
  const auto coords = block->coords;
  const IndexRange flag_i{ib.s - 1, ib.e + 1};
  const IndexRange flag_j{jb.s - (ndim >= 2), jb.e + (ndim >= 2)};
  const IndexRange flag_k{kb.s - (ndim >= 3), kb.e + (ndim >= 3)};
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  block->par_for(
      "PANGU relativistic Hydro FOFC candidate", flag_k.s, flag_k.e, flag_j.s, flag_j.e, flag_i.s,
      flag_i.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real values[5]{};
        for (int n = 0; n < 5; ++n) {
          values[n] =
              gam0 * conserved(n, k, j, i) + gam1 * base_conserved(n, k, j, i) -
              dt * (conserved.flux(X1DIR, n, k, j, i + 1) - conserved.flux(X1DIR, n, k, j, i)) /
                  coords.Dxc<X1DIR>(k, j, i);
          if (ndim >= 2)
            values[n] -=
                dt * (conserved.flux(X2DIR, n, k, j + 1, i) - conserved.flux(X2DIR, n, k, j, i)) /
                coords.Dxc<X2DIR>(k, j, i);
          if (ndim >= 3)
            values[n] -=
                dt * (conserved.flux(X3DIR, n, k + 1, j, i) - conserved.flux(X3DIR, n, k, j, i)) /
                coords.Dxc<X3DIR>(k, j, i);
        }
        const HydroConservedState candidate{
            values[hydro::IDN],
            {values[hydro::IM1], values[hydro::IM2], values[hydro::IM3]},
            values[hydro::IEN]};
        bool bad = false;
        if (use_fofc) {
          if (mode == HydroMode::sr) {
            const auto trial = SolveSRHDC2P(candidate, eos, gamma_max);
            bad = !trial.success || trial.density_floor || trial.pressure_floor ||
                  trial.lorentz_ceiling;
          } else {
            const auto metric = spacetime.MetricAt(geometry::Location::cell_center, geometry_block,
                                                   k, j, i, coords);
            const auto trial = SolveGRHDC2P(candidate, eos, gamma_max, metric);
            bad = !trial.success || trial.density_floor || trial.pressure_floor ||
                  trial.lorentz_ceiling;
          }
        }
        if (geometry::ConfiguredMetricSupportsExcision()) {
          if (mode == HydroMode::gr && excision)
            bad = bad || excision_masks(1, k, j, i) > 0.5;
        }
        flags(0, k, j, i) = bad ? 1.0 : 0.0;
      });
  ReplaceFlaggedFacesBlock<0>(block, primitive, conserved, flags, mode, eos, flag_i, flag_j,
                              flag_k);
  if (ndim >= 2)
    ReplaceFlaggedFacesBlock<1>(block, primitive, conserved, flags, mode, eos, flag_i, flag_j,
                                flag_k);
  if (ndim >= 3)
    ReplaceFlaggedFacesBlock<2>(block, primitive, conserved, flags, mode, eos, flag_i, flag_j,
                                flag_k);
  return TaskStatus::complete;
}

TaskStatus ApplyHydroFluxCorrectionMeshTask(MeshData<Real>* data, MeshData<Real>* base,
                                            const Real gam0, const Real gam1, const Real beta_dt) {
  const Real dt = beta_dt;
  const auto mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("hydro");
  const auto mode = static_cast<HydroMode>(package->Param<int>("physics_mode"));
  if (mode == HydroMode::newtonian)
    return TaskStatus::complete;
  const bool use_fofc = package->Param<bool>("fofc");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  const Real gamma_max = package->Param<Real>("gamma_max");
  const auto geometry_package = mesh->packages.Get("geometry");
  const bool excision = geometry_package->Param<bool>("excision");
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"hydro.cons"});
  const auto base_conserved = base->PackVariables(std::vector<std::string>{"hydro.cons"});
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto flags = data->PackVariables(std::vector<std::string>{"hydro.fofc"});
  parthenon::MeshBlockPack<parthenon::VariablePack<Real>> excision_masks;
  if constexpr (geometry::ConfiguredMetricSupportsExcision())
    excision_masks = data->PackVariables(std::vector<std::string>{"hydro.excision"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int ndim = mesh->ndim;
  const IndexRange flag_i{ib.s - 1, ib.e + 1};
  const IndexRange flag_j{jb.s - (ndim >= 2), jb.e + (ndim >= 2)};
  const IndexRange flag_k{kb.s - (ndim >= 3), kb.e + (ndim >= 3)};
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed relativistic Hydro FOFC candidate",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, flag_k.s, flag_k.e, flag_j.s, flag_j.e,
      flag_i.s, flag_i.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto& coords = conserved.GetCoords(b);
        auto block_conserved = conserved(b);
        bool bad = false;
        if (use_fofc) {
          Real values[5]{};
          for (int n = 0; n < 5; ++n) {
            values[n] = gam0 * conserved(b, n, k, j, i) + gam1 * base_conserved(b, n, k, j, i) -
                        dt *
                            (block_conserved.flux(X1DIR, n, k, j, i + 1) -
                             block_conserved.flux(X1DIR, n, k, j, i)) /
                            coords.Dxc<X1DIR>(k, j, i);
            if (ndim >= 2)
              values[n] -= dt *
                           (block_conserved.flux(X2DIR, n, k, j + 1, i) -
                            block_conserved.flux(X2DIR, n, k, j, i)) /
                           coords.Dxc<X2DIR>(k, j, i);
            if (ndim >= 3)
              values[n] -= dt *
                           (block_conserved.flux(X3DIR, n, k + 1, j, i) -
                            block_conserved.flux(X3DIR, n, k, j, i)) /
                           coords.Dxc<X3DIR>(k, j, i);
          }
          const HydroConservedState candidate{
              values[hydro::IDN],
              {values[hydro::IM1], values[hydro::IM2], values[hydro::IM3]},
              values[hydro::IEN]};
          if (mode == HydroMode::sr) {
            const auto trial = SolveSRHDC2P(candidate, eos, gamma_max);
            bad = !trial.success || trial.density_floor || trial.pressure_floor ||
                  trial.lorentz_ceiling;
          } else {
            const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                                   geometry_block_offset + b, k, j, i, coords);
            const auto trial = SolveGRHDC2P(candidate, eos, gamma_max, metric);
            bad = !trial.success || trial.density_floor || trial.pressure_floor ||
                  trial.lorentz_ceiling;
          }
        }
        if (geometry::ConfiguredMetricSupportsExcision()) {
          if (mode == HydroMode::gr && excision)
            bad = bad || excision_masks(b, 1, k, j, i) > 0.5;
        }
        flags(b, 0, k, j, i) = bad ? 1.0 : 0.0;
      });
  ReplaceFlaggedFacesMesh<0>(data, primitive, conserved, flags, mode, eos, flag_i, flag_j,
                             flag_k);
  if (ndim >= 2)
    ReplaceFlaggedFacesMesh<1>(data, primitive, conserved, flags, mode, eos, flag_i, flag_j,
                               flag_k);
  if (ndim >= 3)
    ReplaceFlaggedFacesMesh<2>(data, primitive, conserved, flags, mode, eos, flag_i, flag_j,
                               flag_k);
  return TaskStatus::complete;
}

void ConservedToPrimitiveBlock(MeshBlockData<Real>* data) {
  auto block = data->GetBlockPointer();
  const auto hydro = block->packages.Get("hydro");
  const auto geometry_package = block->packages.Get("geometry");
  const auto mode = static_cast<HydroMode>(hydro->Param<int>("physics_mode"));
  const auto eos = pangu::eos::ReadRelativistic(*hydro);
  const Real gamma_max = hydro->Param<Real>("gamma_max");
  const bool excision = geometry_package->Param<bool>("excision");
  const Real configured_excision_density = geometry_package->Param<Real>("excision_density");
  const Real configured_excision_pressure = geometry_package->Param<Real>("excision_pressure");
  const Real excision_density =
      configured_excision_density > 0.0 ? configured_excision_density : eos.density_floor;
  const Real excision_pressure =
      configured_excision_pressure > 0.0 ? configured_excision_pressure : eos.pressure_floor;
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto flags = data->PackVariables(std::vector<std::string>{"hydro.fofc"});
  parthenon::VariablePack<Real> excision_masks;
  if constexpr (geometry::ConfiguredMetricSupportsExcision())
    excision_masks = data->PackVariables(std::vector<std::string>{"hydro.excision"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  block->par_for(
      "PANGU relativistic Hydro C2P", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        HydroConservedState state{conserved(hydro::IDN, k, j, i),
                                  {conserved(hydro::IM1, k, j, i), conserved(hydro::IM2, k, j, i),
                                   conserved(hydro::IM3, k, j, i)},
                                  conserved(hydro::IEN, k, j, i)};
        HydroC2PResult result{};
        const auto metric =
            spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
        bool excised = false;
        if (geometry::ConfiguredMetricSupportsExcision())
          excised = excision && excision_masks(0, k, j, i) > 0.5;
        if (excised) {
          result.primitive = {excision_density, {0.0, 0.0, 0.0}, excision_pressure};
          result.success = false;
        } else if (mode == HydroMode::sr) {
          result = SolveSRHDC2P(state, eos, gamma_max);
        } else {
          result = SolveGRHDC2P(state, eos, gamma_max, metric);
        }
        primitive(hydro::IDN, k, j, i) = result.primitive.density;
        primitive(hydro::IV1, k, j, i) = result.primitive.u[0];
        primitive(hydro::IV2, k, j, i) = result.primitive.u[1];
        primitive(hydro::IV3, k, j, i) = result.primitive.u[2];
        primitive(hydro::IPR, k, j, i) = result.primitive.pressure;
        const bool repaired = !result.success || result.density_floor || result.pressure_floor ||
                              result.lorentz_ceiling;
        flags(0, k, j, i) = repaired ? 1.0 : 0.0;
        if (repaired) {
          const auto fixed = mode == HydroMode::sr
                                 ? ConvertSRHDP2C(result.primitive, eos)
                                 : ConvertGRHDP2C(result.primitive, eos, metric);
          conserved(hydro::IDN, k, j, i) = fixed.density;
          conserved(hydro::IM1, k, j, i) = fixed.momentum[0];
          conserved(hydro::IM2, k, j, i) = fixed.momentum[1];
          conserved(hydro::IM3, k, j, i) = fixed.momentum[2];
          conserved(hydro::IEN, k, j, i) = fixed.energy;
        }
      });
}

void ConservedToPrimitiveMesh(MeshData<Real>* data) {
  const auto mesh = data->GetParentPointer();
  const auto hydro = mesh->packages.Get("hydro");
  const auto geometry_package = mesh->packages.Get("geometry");
  const auto mode = static_cast<HydroMode>(hydro->Param<int>("physics_mode"));
  PARTHENON_REQUIRE(mode != HydroMode::newtonian,
                    "Relativistic Hydro packed C2P requires SR or GR mode");
  const auto eos = pangu::eos::ReadRelativistic(*hydro);
  const Real gamma_max = hydro->Param<Real>("gamma_max");
  const bool excision = geometry_package->Param<bool>("excision");
  const Real configured_excision_density = geometry_package->Param<Real>("excision_density");
  const Real configured_excision_pressure = geometry_package->Param<Real>("excision_pressure");
  const Real excision_density =
      configured_excision_density > 0.0 ? configured_excision_density : eos.density_floor;
  const Real excision_pressure =
      configured_excision_pressure > 0.0 ? configured_excision_pressure : eos.pressure_floor;
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto flags = data->PackVariables(std::vector<std::string>{"hydro.fofc"});
  parthenon::MeshBlockPack<parthenon::VariablePack<Real>> excision_masks;
  if constexpr (geometry::ConfiguredMetricSupportsExcision())
    excision_masks = data->PackVariables(std::vector<std::string>{"hydro.excision"});
  const auto ib = data->GetBoundsI(IndexDomain::entire);
  const auto jb = data->GetBoundsJ(IndexDomain::entire);
  const auto kb = data->GetBoundsK(IndexDomain::entire);
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed relativistic Hydro C2P", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        HydroConservedState state{conserved(b, hydro::IDN, k, j, i),
                                  {conserved(b, hydro::IM1, k, j, i),
                                   conserved(b, hydro::IM2, k, j, i),
                                   conserved(b, hydro::IM3, k, j, i)},
                                  conserved(b, hydro::IEN, k, j, i)};
        HydroC2PResult result{};
        const auto& coords = conserved.GetCoords(b);
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                               geometry_block_offset + b, k, j, i, coords);
        bool excised = false;
        if (geometry::ConfiguredMetricSupportsExcision())
          excised = excision && excision_masks(b, 0, k, j, i) > 0.5;
        if (excised) {
          result.primitive = {excision_density, {0.0, 0.0, 0.0}, excision_pressure};
          result.success = false;
        } else if (mode == HydroMode::sr) {
          result = SolveSRHDC2P(state, eos, gamma_max);
        } else {
          result = SolveGRHDC2P(state, eos, gamma_max, metric);
        }
        primitive(b, hydro::IDN, k, j, i) = result.primitive.density;
        primitive(b, hydro::IV1, k, j, i) = result.primitive.u[0];
        primitive(b, hydro::IV2, k, j, i) = result.primitive.u[1];
        primitive(b, hydro::IV3, k, j, i) = result.primitive.u[2];
        primitive(b, hydro::IPR, k, j, i) = result.primitive.pressure;
        const bool repaired = !result.success || result.density_floor || result.pressure_floor ||
                              result.lorentz_ceiling;
        flags(b, 0, k, j, i) = repaired ? 1.0 : 0.0;
        if (repaired) {
          const auto fixed = mode == HydroMode::sr
                                 ? ConvertSRHDP2C(result.primitive, eos)
                                 : ConvertGRHDP2C(result.primitive, eos, metric);
          conserved(b, hydro::IDN, k, j, i) = fixed.density;
          conserved(b, hydro::IM1, k, j, i) = fixed.momentum[0];
          conserved(b, hydro::IM2, k, j, i) = fixed.momentum[1];
          conserved(b, hydro::IM3, k, j, i) = fixed.momentum[2];
          conserved(b, hydro::IEN, k, j, i) = fixed.energy;
        }
      });
}

Real EstimateTimestepBlock(MeshBlockData<Real>* data) {
  auto block = data->GetBlockPointer();
  const auto hydro = block->packages.Get("hydro");
  const auto mode = static_cast<HydroMode>(hydro->Param<int>("physics_mode"));
  const Real cfl = hydro->Param<Real>("cfl");
  const auto eos = pangu::eos::ReadRelativistic(*hydro);
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  const int ndim = block->pmy_mesh->ndim;
  if (!Estimator::characteristic_gr_speeds && mode == HydroMode::gr)
    return cfl * estimator::UnitSpeedTimestep(coords, kb.s, jb.s, ib.s, ndim);
  Real minimum = std::numeric_limits<Real>::max();
  ParReduce(
      "PANGU relativistic Hydro timestep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
        HydroPrimitiveState state{primitive(hydro::IDN, k, j, i),
                                  {primitive(hydro::IV1, k, j, i), primitive(hydro::IV2, k, j, i),
                                   primitive(hydro::IV3, k, j, i)},
                                  primitive(hydro::IPR, k, j, i)};
        Real estimate = Estimator::Start();
        for (int direction = 0; direction < ndim; ++direction) {
          Real plus, minus;
          if (mode == HydroMode::sr) {
            SRSoundSpeeds(state, eos, direction, plus, minus);
          } else if (Estimator::characteristic_gr_speeds) {
            const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                                   geometry_block, k, j, i, coords);
            const auto wave = BuildGRHDFluxState(state, eos, direction, metric);
            plus = wave.lambda_plus;
            minus = wave.lambda_minus;
            // Bound the estimator by the local coordinate light cone; this is
            // a timestep-only guard and leaves the Riemann flux unchanged.
            Real light_plus = 0.0, light_minus = 0.0;
            ComputeCoordinateLightSpeeds(metric, direction, light_plus, light_minus);
            const Real light_speed = fmax(fabs(light_plus), fabs(light_minus));
            plus = fmin(plus, light_speed);
            minus = fmax(minus, -light_speed);
          } else {
            plus = 1.0;
            minus = -1.0;
          }
          const Real speed = fmax(fabs(plus), fabs(minus));
          estimate = Estimator::Add(estimate, speed, coords.Dxc(direction + 1, k, j, i));
        }
        local = fmin(local, Estimator::Finish(estimate));
      },
      Kokkos::Min<Real>(minimum));
  return cfl * minimum;
}

Real EstimateTimestepMesh(MeshData<Real>* data) {
  const auto mesh = data->GetParentPointer();
  const auto hydro = mesh->packages.Get("hydro");
  const auto mode = static_cast<HydroMode>(hydro->Param<int>("physics_mode"));
  if (!Estimator::characteristic_gr_speeds && mode == HydroMode::gr) {
    Real minimum = std::numeric_limits<Real>::max();
    for (int b = 0; b < data->NumBlocks(); ++b)
      minimum = fmin(minimum, EstimateTimestepBlock(data->GetBlockData(b).get()));
    return minimum;
  }
  PARTHENON_REQUIRE(mode != HydroMode::newtonian,
                    "Relativistic Hydro packed timestep requires SR or GR mode");
  const Real cfl = hydro->Param<Real>("cfl");
  const auto eos = pangu::eos::ReadRelativistic(*hydro);
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int ndim = mesh->ndim;
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  Real minimum = std::numeric_limits<Real>::max();
  ParReduce(
      "PANGU packed relativistic Hydro timestep", 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        HydroPrimitiveState state{primitive(b, hydro::IDN, k, j, i),
                                  {primitive(b, hydro::IV1, k, j, i),
                                   primitive(b, hydro::IV2, k, j, i),
                                   primitive(b, hydro::IV3, k, j, i)},
                                  primitive(b, hydro::IPR, k, j, i)};
        const auto& coords = primitive.GetCoords(b);
        Real estimate = Estimator::Start();
        for (int direction = 0; direction < ndim; ++direction) {
          Real plus, minus;
          if (mode == HydroMode::sr) {
            SRSoundSpeeds(state, eos, direction, plus, minus);
          } else if (Estimator::characteristic_gr_speeds) {
            const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                                   geometry_block_offset + b, k, j, i, coords);
            const auto wave = BuildGRHDFluxState(state, eos, direction, metric);
            plus = wave.lambda_plus;
            minus = wave.lambda_minus;
            // Bound the estimator by the local coordinate light cone; this is
            // a timestep-only guard and leaves the Riemann flux unchanged.
            Real light_plus = 0.0, light_minus = 0.0;
            ComputeCoordinateLightSpeeds(metric, direction, light_plus, light_minus);
            const Real light_speed = fmax(fabs(light_plus), fabs(light_minus));
            plus = fmin(plus, light_speed);
            minus = fmax(minus, -light_speed);
          } else {
            plus = 1.0;
            minus = -1.0;
          }
          const Real speed = fmax(fabs(plus), fabs(minus));
          estimate = Estimator::Add(estimate, speed, coords.Dxc(direction + 1, k, j, i));
        }
        local = fmin(local, Estimator::Finish(estimate));
      },
      Kokkos::Min<Real>(minimum));
  return cfl * minimum;
}

TaskStatus ApplySourcesBlockTask(std::shared_ptr<MeshBlockData<Real>>& data, const Real dt) {
  auto block = data->GetBlockPointer();
  const auto hydro = block->packages.Get("hydro");
  if (static_cast<HydroMode>(hydro->Param<int>("physics_mode")) != HydroMode::gr)
    return TaskStatus::complete;
  const auto eos = pangu::eos::ReadRelativistic(*hydro);
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  block->par_for(
      "PANGU fixed-GR geometric source", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto metric =
            spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
        HydroPrimitiveState state{primitive(hydro::IDN, k, j, i),
                                  {primitive(hydro::IV1, k, j, i), primitive(hydro::IV2, k, j, i),
                                   primitive(hydro::IV3, k, j, i)},
                                  primitive(hydro::IPR, k, j, i)};
        Real spatial_u2 = 0.0;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b)
            spatial_u2 += metric.lower[a + 1][b + 1] * state.u[a] * state.u[b];
        const Real lorentz = sqrt(1.0 + spatial_u2);
        Real u[4]{lorentz / metric.lapse, 0.0, 0.0, 0.0};
        for (int axis = 0; axis < 3; ++axis)
          u[axis + 1] = state.u[axis] - metric.lapse * lorentz * metric.upper[0][axis + 1];
        const Real enthalpy = eos.EnthalpyDensity(state.density, state.pressure);
        Real stress[4][4];
        for (int mu = 0; mu < 4; ++mu)
          for (int nu = 0; nu < 4; ++nu)
            stress[mu][nu] = enthalpy * u[mu] * u[nu] + state.pressure * metric.upper[mu][nu];
        const auto derivatives = spacetime.DerivativesAt(geometry_block, k, j, i, coords);
        for (int axis = 0; axis < 3; ++axis) {
          Real source = 0.0;
          for (int mu = 0; mu < 4; ++mu)
            for (int nu = 0; nu < 4; ++nu)
              source += 0.5 * stress[mu][nu] * derivatives.lower[axis][mu][nu];
          conserved(hydro::IM1 + axis, k, j, i) += dt * metric.gdet * source;
        }
      });
  return TaskStatus::complete;
}

TaskStatus ApplySourcesMeshTask(MeshData<Real>* data, const Real dt) {
  const auto mesh = data->GetParentPointer();
  const auto hydro = mesh->packages.Get("hydro");
  if (static_cast<HydroMode>(hydro->Param<int>("physics_mode")) != HydroMode::gr)
    return TaskStatus::complete;
  const auto eos = pangu::eos::ReadRelativistic(*hydro);
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed fixed-GR geometric source", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto& coords = primitive.GetCoords(b);
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                               geometry_block_offset + b, k, j, i, coords);
        HydroPrimitiveState state{primitive(b, hydro::IDN, k, j, i),
                                  {primitive(b, hydro::IV1, k, j, i),
                                   primitive(b, hydro::IV2, k, j, i),
                                   primitive(b, hydro::IV3, k, j, i)},
                                  primitive(b, hydro::IPR, k, j, i)};
        Real spatial_u2 = 0.0;
        for (int a = 0; a < 3; ++a)
          for (int c = 0; c < 3; ++c)
            spatial_u2 += metric.lower[a + 1][c + 1] * state.u[a] * state.u[c];
        const Real lorentz = sqrt(1.0 + spatial_u2);
        Real u[4]{lorentz / metric.lapse, 0.0, 0.0, 0.0};
        for (int axis = 0; axis < 3; ++axis)
          u[axis + 1] = state.u[axis] - metric.lapse * lorentz * metric.upper[0][axis + 1];
        const Real enthalpy = eos.EnthalpyDensity(state.density, state.pressure);
        Real stress[4][4];
        for (int mu = 0; mu < 4; ++mu)
          for (int nu = 0; nu < 4; ++nu)
            stress[mu][nu] = enthalpy * u[mu] * u[nu] + state.pressure * metric.upper[mu][nu];
        const auto derivatives =
            spacetime.DerivativesAt(geometry_block_offset + b, k, j, i, coords);
        for (int axis = 0; axis < 3; ++axis) {
          Real source = 0.0;
          for (int mu = 0; mu < 4; ++mu)
            for (int nu = 0; nu < 4; ++nu)
              source += 0.5 * stress[mu][nu] * derivatives.lower[axis][mu][nu];
          conserved(b, hydro::IM1 + axis, k, j, i) += dt * metric.gdet * source;
        }
      });
  return TaskStatus::complete;
}

} // namespace pangu::relativity
