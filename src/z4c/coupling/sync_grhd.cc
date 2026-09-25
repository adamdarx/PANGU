#include "z4c/coupling/sync_grhd.h"

#include <limits>
#include <vector>

#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "z4c/core/finite_difference.h"
#include "reconstruct/hydro_reconstruction.h"
#include "riemann/registry.h"

namespace pangu::nr {
using namespace parthenon::package::prelude;

namespace {

template <int Direction, class Pack>
KOKKOS_INLINE_FUNCTION Real ReadPrimitive(const Pack& primitive, const int block,
                                          const int component, const int k, const int j,
                                          const int i, const int offset) {
  if constexpr (Direction == 0)
    return primitive(block, component, k, j, i + offset);
  if constexpr (Direction == 1)
    return primitive(block, component, k, j + offset, i);
  return primitive(block, component, k + offset, j, i);
}

template <hydro::Reconstruction Method, int Direction, class Pack>
KOKKOS_INLINE_FUNCTION void ReconstructSyncGRHD(const Pack& primitive, const int block,
                                                const int k, const int j, const int i,
                                                const eos::RelativisticEOS& eos,
                                                relativity::HydroPrimitiveState& left,
                                                relativity::HydroPrimitiveState& right) {
  Real left_values[5]{};
  Real right_values[5]{};
  for (int component = 0; component < 5; ++component) {
    reconstruct::ReconstructFaceStencil<Method>(
        [&](const int offset) {
          return ReadPrimitive<Direction>(primitive, block, component, k, j, i, offset);
        },
        left_values[component], right_values[component]);
  }
  left = {fmax(left_values[hydro::IDN], eos.density_floor),
          {left_values[hydro::IV1], left_values[hydro::IV2], left_values[hydro::IV3]},
          fmax(left_values[hydro::IPR], eos.pressure_floor)};
  right = {fmax(right_values[hydro::IDN], eos.density_floor),
           {right_values[hydro::IV1], right_values[hydro::IV2], right_values[hydro::IV3]},
           fmax(right_values[hydro::IPR], eos.pressure_floor)};
}

template <hydro::Reconstruction Method, riemann::Solver Solver, int Direction>
void CalculateSyncGRHDDirection(MeshData<Real>* data, const eos::RelativisticEOS& eos) {
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"hydro.cons"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
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
  const auto spacetime = geometry::GetGeometry(data->GetParentPointer()->packages);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed sync GRHD flux", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        relativity::HydroPrimitiveState left{}, right{};
        ReconstructSyncGRHD<Method, Direction>(primitive, block, k, j, i, eos, left, right);
        const auto& coordinates = primitive.GetCoords(block);
        const auto metric = spacetime.MetricAt(geometry::FaceLocation(Direction), block, k, j, i,
                                               coordinates, z4c, adm);
        const auto left_state = BuildSyncGRHDFluxState(left, eos, Direction, metric);
        const auto right_state = BuildSyncGRHDFluxState(right, eos, Direction, metric);
        Real flux[5]{};
        riemann::Implementation<riemann::Physics::relativistic_hydro, Solver>::Solve(
            left_state, right_state, flux);
        auto block_conserved = conserved(block);
        for (int component = 0; component < 5; ++component)
          block_conserved.flux(Direction + 1, component, k, j, i) = flux[component];
      });
}

template <hydro::Reconstruction Method, riemann::Solver Solver>
void CalculateAllSyncGRHDDirections(MeshData<Real>* data, const eos::RelativisticEOS& eos) {
  CalculateSyncGRHDDirection<Method, Solver, 0>(data, eos);
  if (data->GetParentPointer()->ndim >= 2)
    CalculateSyncGRHDDirection<Method, Solver, 1>(data, eos);
  if (data->GetParentPointer()->ndim >= 3)
    CalculateSyncGRHDDirection<Method, Solver, 2>(data, eos);
}

template <class Pack> struct FieldDerivativeAccessor {
  Pack field;
  int block;
  int component;
  int k;
  int j;
  int i;

  KOKKOS_INLINE_FUNCTION Real operator()(const int dk, const int dj, const int di) const {
    return field(block, component, k + dk, j + dj, i + di);
  }
};

template <int Order>
TaskStatus ApplySyncMatterSources(MeshData<Real>* current, MeshData<Real>* next, const Real dt,
                                  const std::string& conserved_field) {
  const auto z4c = current->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = current->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto tmunu = current->PackVariables(std::vector<std::string>{"nr.tmunu"});
  auto conserved = next->PackVariables(std::vector<std::string>{conserved_field});
  const auto mesh = current->GetParentPointer();
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const auto ib = current->GetBoundsI(IndexDomain::interior);
  const auto jb = current->GetBoundsJ(IndexDomain::interior);
  const auto kb = current->GetBoundsK(IndexDomain::interior);
  const int ndim = mesh->ndim;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed sync GRHD geometric source",
      parthenon::DevExecSpace(), 0, current->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const auto& coordinates = z4c.GetCoords(block);
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, block, k, j, i,
                                               coordinates, z4c, adm);
        SyncGRHDGeometryDerivatives derivatives{};
        for (int direction = 0; direction < ndim; ++direction) {
          const Real inverse_spacing = direction == 0   ? 1.0 / coordinates.Dxc<X1DIR>(k, j, i)
                                       : direction == 1 ? 1.0 / coordinates.Dxc<X2DIR>(k, j, i)
                                                        : 1.0 / coordinates.Dxc<X3DIR>(k, j, i);
          derivatives.lapse[direction] = fd::First<Order>(
              direction, inverse_spacing,
              FieldDerivativeAccessor<decltype(z4c)>{
                  z4c, block, Index(Z4cComponent::alpha), k, j, i});
          for (int axis = 0; axis < 3; ++axis) {
            derivatives.shift[direction][axis] = fd::First<Order>(
                direction, inverse_spacing,
                FieldDerivativeAccessor<decltype(z4c)>{
                    z4c, block, Index(Z4cComponent::betax) + axis, k, j, i});
          }
          for (int component = 0; component < 6; ++component) {
            derivatives.spatial_metric[direction][component] = fd::First<Order>(
                direction, inverse_spacing,
                FieldDerivativeAccessor<decltype(adm)>{
                    adm, block, Index(ADMComponent::gxx) + component, k, j, i});
          }
        }
        for (int component = 0; component < 6; ++component)
          derivatives.extrinsic[component] =
              adm(block, Index(ADMComponent::kxx) + component, k, j, i);
        Real source[5]{};
        ComputeSyncGRHDSource(LoadStressEnergy(tmunu, block, k, j, i), metric, derivatives,
                              source);
        for (int component = 0; component < 5; ++component)
          conserved(block, component, k, j, i) += dt * source[component];
      });
  return TaskStatus::complete;
}

} // namespace

TaskStatus BuildSyncGRHDStressEnergyBlockTask(MeshBlockData<Real>* data) {
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  auto tmunu = data->PackVariables(std::vector<std::string>{"nr.tmunu"});
  const auto block = data->GetBlockPointer();
  const auto spacetime = geometry::GetGeometry(block->packages);
  const auto coordinates = block->coords;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  block->par_for(
      "PANGU sync GRHD stress-energy initialization", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, 0, k, j, i,
                                               coordinates, z4c, adm);
        const relativity::HydroPrimitiveState primitive_state{
            primitive(hydro::IDN, k, j, i),
            {primitive(hydro::IV1, k, j, i), primitive(hydro::IV2, k, j, i),
             primitive(hydro::IV3, k, j, i)},
            primitive(hydro::IPR, k, j, i)};
        const relativity::HydroConservedState conserved_state{
            conserved(hydro::IDN, k, j, i),
            {conserved(hydro::IM1, k, j, i), conserved(hydro::IM2, k, j, i),
             conserved(hydro::IM3, k, j, i)},
            conserved(hydro::IEN, k, j, i)};
        const auto matter = BuildSyncGRHDStressEnergy(primitive_state, conserved_state, metric);
        for (int component = 0; component < 6; ++component)
          tmunu(component, k, j, i) = matter.stress[component];
        tmunu(Index(StressEnergyComponent::energy), k, j, i) = matter.energy;
        for (int axis = 0; axis < 3; ++axis)
          tmunu(Index(StressEnergyComponent::momentum_x) + axis, k, j, i) =
              matter.momentum[axis];
      });
  return TaskStatus::complete;
}

TaskStatus BuildSyncGRHDStressEnergyMeshTask(MeshData<Real>* data) {
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  auto tmunu = data->PackVariables(std::vector<std::string>{"nr.tmunu"});
  const auto mesh = data->GetParentPointer();
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const auto first = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = first->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = first->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = first->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed sync GRHD stress-energy", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const auto& coordinates = primitive.GetCoords(block);
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, block, k, j, i,
                                               coordinates, z4c, adm);
        const relativity::HydroPrimitiveState primitive_state{
            primitive(block, hydro::IDN, k, j, i),
            {primitive(block, hydro::IV1, k, j, i), primitive(block, hydro::IV2, k, j, i),
             primitive(block, hydro::IV3, k, j, i)},
            primitive(block, hydro::IPR, k, j, i)};
        const relativity::HydroConservedState conserved_state{
            conserved(block, hydro::IDN, k, j, i),
            {conserved(block, hydro::IM1, k, j, i), conserved(block, hydro::IM2, k, j, i),
             conserved(block, hydro::IM3, k, j, i)},
            conserved(block, hydro::IEN, k, j, i)};
        const auto matter = BuildSyncGRHDStressEnergy(primitive_state, conserved_state, metric);
        for (int component = 0; component < 6; ++component)
          tmunu(block, component, k, j, i) = matter.stress[component];
        tmunu(block, Index(StressEnergyComponent::energy), k, j, i) = matter.energy;
        for (int axis = 0; axis < 3; ++axis)
          tmunu(block, Index(StressEnergyComponent::momentum_x) + axis, k, j, i) =
              matter.momentum[axis];
      });
  return TaskStatus::complete;
}

TaskStatus CalculateSyncGRHDFluxesMeshTask(MeshData<Real>* data) {
  const auto hydro_package = data->GetParentPointer()->packages.Get("hydro");
  const auto eos = pangu::eos::ReadRelativistic(*hydro_package);
  const auto reconstruction =
      static_cast<hydro::Reconstruction>(hydro_package->Param<int>("reconstruction"));
  const auto solver = static_cast<riemann::Solver>(hydro_package->Param<int>("riemann"));
  PARTHENON_REQUIRE(riemann::Describe(solver).relativistic_hydro,
                    "sync GRHD requires a Riemann solver registered for relativistic Hydro");
  riemann::Visit<riemann::Physics::relativistic_hydro>(solver, [&]<riemann::Solver Solver>() {
    reconstruct::Visit(reconstruction, [&]<hydro::Reconstruction Method>() {
      CalculateAllSyncGRHDDirections<Method, Solver>(data, eos);
    });
  });
  return TaskStatus::complete;
}

TaskStatus ApplySyncGRHDSourcesMeshTask(MeshData<Real>* current, MeshData<Real>* next,
                                        const Real dt) {
  return ApplySyncMatterSourcesMeshTask(current, next, dt, "hydro.cons");
}

TaskStatus ApplySyncMatterSourcesMeshTask(MeshData<Real>* current, MeshData<Real>* next,
                                          const Real dt,
                                          const std::string& conserved_field) {
  const int order = current->GetParentPointer()
                        ->packages.Get("numerical_relativity")
                        ->Param<int>("finite_difference_order");
  if (order == 2)
    return ApplySyncMatterSources<2>(current, next, dt, conserved_field);
  if (order == 4)
    return ApplySyncMatterSources<4>(current, next, dt, conserved_field);
  return ApplySyncMatterSources<6>(current, next, dt, conserved_field);
}

void SyncGRHDConservedToPrimitiveBlock(MeshBlockData<Real>* data) {
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto flags = data->PackVariables(std::vector<std::string>{"hydro.fofc"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("hydro");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  const Real gamma_max = package->Param<Real>("gamma_max");
  const auto spacetime = geometry::GetGeometry(block->packages);
  const auto coordinates = block->coords;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  block->par_for(
      "PANGU sync GRHD conserved to primitive", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, 0, k, j, i,
                                               coordinates, z4c, adm);
        const relativity::HydroConservedState state{
            conserved(hydro::IDN, k, j, i),
            {conserved(hydro::IM1, k, j, i), conserved(hydro::IM2, k, j, i),
             conserved(hydro::IM3, k, j, i)},
            conserved(hydro::IEN, k, j, i)};
        const auto result = SolveSyncGRHDC2P(state, eos, gamma_max, metric);
        primitive(hydro::IDN, k, j, i) = result.primitive.density;
        primitive(hydro::IV1, k, j, i) = result.primitive.u[0];
        primitive(hydro::IV2, k, j, i) = result.primitive.u[1];
        primitive(hydro::IV3, k, j, i) = result.primitive.u[2];
        primitive(hydro::IPR, k, j, i) = result.primitive.pressure;
        const bool repaired = !result.success || result.density_floor || result.pressure_floor ||
                              result.lorentz_ceiling;
        flags(0, k, j, i) = repaired ? 1.0 : 0.0;
        if (repaired) {
          const auto fixed = ConvertSyncGRHDP2C(result.primitive, eos, metric);
          conserved(hydro::IDN, k, j, i) = fixed.density;
          conserved(hydro::IM1, k, j, i) = fixed.momentum[0];
          conserved(hydro::IM2, k, j, i) = fixed.momentum[1];
          conserved(hydro::IM3, k, j, i) = fixed.momentum[2];
          conserved(hydro::IEN, k, j, i) = fixed.energy;
        }
      });
}

void SyncGRHDConservedToPrimitiveMesh(MeshData<Real>* data) {
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto flags = data->PackVariables(std::vector<std::string>{"hydro.fofc"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("hydro");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  const Real gamma_max = package->Param<Real>("gamma_max");
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const auto ib = data->GetBoundsI(IndexDomain::entire);
  const auto jb = data->GetBoundsJ(IndexDomain::entire);
  const auto kb = data->GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed sync GRHD conserved to primitive",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const auto& coordinates = conserved.GetCoords(block);
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, block, k, j, i,
                                               coordinates, z4c, adm);
        const relativity::HydroConservedState state{
            conserved(block, hydro::IDN, k, j, i),
            {conserved(block, hydro::IM1, k, j, i), conserved(block, hydro::IM2, k, j, i),
             conserved(block, hydro::IM3, k, j, i)},
            conserved(block, hydro::IEN, k, j, i)};
        const auto result = SolveSyncGRHDC2P(state, eos, gamma_max, metric);
        primitive(block, hydro::IDN, k, j, i) = result.primitive.density;
        primitive(block, hydro::IV1, k, j, i) = result.primitive.u[0];
        primitive(block, hydro::IV2, k, j, i) = result.primitive.u[1];
        primitive(block, hydro::IV3, k, j, i) = result.primitive.u[2];
        primitive(block, hydro::IPR, k, j, i) = result.primitive.pressure;
        const bool repaired = !result.success || result.density_floor || result.pressure_floor ||
                              result.lorentz_ceiling;
        flags(block, 0, k, j, i) = repaired ? 1.0 : 0.0;
        if (repaired) {
          const auto fixed = ConvertSyncGRHDP2C(result.primitive, eos, metric);
          conserved(block, hydro::IDN, k, j, i) = fixed.density;
          conserved(block, hydro::IM1, k, j, i) = fixed.momentum[0];
          conserved(block, hydro::IM2, k, j, i) = fixed.momentum[1];
          conserved(block, hydro::IM3, k, j, i) = fixed.momentum[2];
          conserved(block, hydro::IEN, k, j, i) = fixed.energy;
        }
      });
}

TaskStatus SyncGRHDConservedToPrimitiveMeshTask(MeshData<Real>* data) {
  SyncGRHDConservedToPrimitiveMesh(data);
  return TaskStatus::complete;
}

Real EstimateSyncGRHDTimestepBlock(MeshBlockData<Real>* data) {
  const auto block = data->GetBlockPointer();
  const Real cfl = block->packages.Get("hydro")->Param<Real>("cfl");
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  Real minimum = block->coords.Dxc<X1DIR>(kb.s, jb.s, ib.s);
  if (block->pmy_mesh->ndim >= 2)
    minimum = fmin(minimum, block->coords.Dxc<X2DIR>(kb.s, jb.s, ib.s));
  if (block->pmy_mesh->ndim >= 3)
    minimum = fmin(minimum, block->coords.Dxc<X3DIR>(kb.s, jb.s, ib.s));
  return cfl * minimum;
}

Real EstimateSyncGRHDTimestepMesh(MeshData<Real>* data) {
  Real minimum = std::numeric_limits<Real>::max();
  for (int block = 0; block < data->NumBlocks(); ++block)
    minimum = fmin(minimum, EstimateSyncGRHDTimestepBlock(data->GetBlockData(block).get()));
  return minimum;
}

} // namespace pangu::nr
