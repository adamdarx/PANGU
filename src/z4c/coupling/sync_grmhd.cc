#include "z4c/coupling/sync_grmhd.h"

#include <limits>
#include <vector>

#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "mhd/mhd_types.h"
#include "z4c/coupling/sync_grhd.h"
#include "reconstruct/hydro_reconstruction.h"
#include "riemann/registry.h"

namespace pangu::nr {
using namespace parthenon::package::prelude;

namespace {

// First-order flux correction replaces flagged faces with the registered fallback solver.
using FirstOrderSolver =
    riemann::Implementation<riemann::Physics::relativistic_mhd, riemann::first_order_fallback>;

constexpr int E2X1 = 0;
constexpr int E3X1 = 1;
constexpr int E1X2 = 2;
constexpr int E3X2 = 3;
constexpr int E1X3 = 4;
constexpr int E2X3 = 5;

template <int Direction, class Pack>
KOKKOS_INLINE_FUNCTION Real ReadOffset(const Pack& values, const int block,
                                       const int component, const int k, const int j,
                                       const int i, const int offset) {
  if constexpr (Direction == 0)
    return values(block, component, k, j, i + offset);
  if constexpr (Direction == 1)
    return values(block, component, k, j + offset, i);
  return values(block, component, k + offset, j, i);
}

template <hydro::Reconstruction Method, int Direction, class Pack>
KOKKOS_INLINE_FUNCTION void ReconstructValue(const Pack& values, const int block,
                                              const int component, const int k,
                                              const int j, const int i, Real& left,
                                              Real& right) {
  reconstruct::ReconstructFaceStencil<Method>(
      [&](const int offset) {
        return ReadOffset<Direction>(values, block, component, k, j, i, offset);
      },
      left, right);
}

template <hydro::Reconstruction Method, riemann::Solver Solver, int Direction>
void CalculateSyncGRMHDDirection(MeshData<Real>* data, const eos::RelativisticEOS& eos) {
  using TE = parthenon::TopologicalElement;
  const auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  const auto magnetic = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
  auto face_emf = data->PackVariables(std::vector<std::string>{"mhd.face_emf"});
  const auto bface = data->PackVariables(std::vector<std::string>{"mhd.b_face"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int ndim = data->GetParentPointer()->ndim;
  int il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  if constexpr (Direction == 0) {
    ++iu;
    if (ndim >= 2) {
      --jl;
      ++ju;
    }
    if (ndim >= 3) {
      --kl;
      ++ku;
    }
  } else if constexpr (Direction == 1) {
    ++ju;
    --il;
    ++iu;
    if (ndim >= 3) {
      --kl;
      ++ku;
    }
  } else {
    ++ku;
    --jl;
    ++ju;
    --il;
    ++iu;
  }
  constexpr TE face = Direction == 0 ? TE::F1 : (Direction == 1 ? TE::F2 : TE::F3);
  constexpr int emf_t1 =
      Direction == 0 ? E2X1 : (Direction == 1 ? E3X2 : E1X3);
  constexpr int emf_t2 =
      Direction == 0 ? E3X1 : (Direction == 1 ? E1X2 : E2X3);
  const int interior_is = ib.s;
  const int interior_ie = ib.e;
  const int interior_js = jb.s;
  const int interior_je = jb.e;
  const int interior_ks = kb.s;
  const int interior_ke = kb.e;
  const auto spacetime = geometry::GetGeometry(data->GetParentPointer()->packages);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed sync GRMHD Riemann solve",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        // The CT corner solve needs one transverse ghost slab for each EMF
        // component, but never the intersection of both transverse slabs.
        // Avoid evaluating Geometry in those unused outer corners: Parthenon
        // intentionally does not guarantee derived cell data there after a
        // runtime AMR remesh when the ghost width equals the block extent.
        const bool outside_i = i < interior_is || i > interior_ie;
        const bool outside_j = j < interior_js || j > interior_je;
        const bool outside_k = k < interior_ks || k > interior_ke;
        if constexpr (Direction == 0) {
          if (outside_j && outside_k)
            return;
        } else if constexpr (Direction == 1) {
          if (outside_i && outside_k)
            return;
        } else {
          if (outside_i && outside_j)
            return;
        }
        Real left_fluid[5]{}, right_fluid[5]{};
        Real left_magnetic[3]{}, right_magnetic[3]{};
        for (int component = 0; component < 5; ++component)
          ReconstructValue<Method, Direction>(primitive, block, component, k, j, i,
                                               left_fluid[component],
                                               right_fluid[component]);
        for (int component = 0; component < 3; ++component)
          ReconstructValue<Method, Direction>(magnetic, block, component, k, j, i,
                                               left_magnetic[component],
                                               right_magnetic[component]);
        const auto& coordinates = primitive.GetCoords(block);
        const auto metric = spacetime.MetricAt(geometry::FaceLocation(Direction), block, k,
                                               j, i, coordinates, z4c, adm);
        const Real inverse_volume = 1.0 / sqrt(metric.spatial_det);
        relativity::MHDPrimitiveState left{
            {fmax(left_fluid[mhd::IDN], eos.density_floor),
             {left_fluid[mhd::IV1], left_fluid[mhd::IV2], left_fluid[mhd::IV3]},
             fmax(eos.PressureFromInternalEnergyDensity(
                      left_fluid[mhd::IPR]),
                  eos.pressure_floor)},
            {left_magnetic[0] * inverse_volume, left_magnetic[1] * inverse_volume,
             left_magnetic[2] * inverse_volume}};
        relativity::MHDPrimitiveState right{
            {fmax(right_fluid[mhd::IDN], eos.density_floor),
             {right_fluid[mhd::IV1], right_fluid[mhd::IV2],
              right_fluid[mhd::IV3]},
             fmax(eos.PressureFromInternalEnergyDensity(
                      right_fluid[mhd::IPR]),
                  eos.pressure_floor)},
            {right_magnetic[0] * inverse_volume, right_magnetic[1] * inverse_volume,
             right_magnetic[2] * inverse_volume}};
        const Real normal = bface(block, face, 0, k, j, i) * inverse_volume;
        left.magnetic[Direction] = normal;
        right.magnetic[Direction] = normal;
        const auto left_state = BuildSyncGRMHDFluxState(left, eos, Direction, metric);
        const auto right_state = BuildSyncGRMHDFluxState(right, eos, Direction, metric);
        const auto flux =
            riemann::Implementation<riemann::Physics::relativistic_mhd, Solver>::Solve(
                left_state, right_state, metric);
        auto block_conserved = conserved(block);
        for (int component = 0; component < 5; ++component)
          block_conserved.flux(Direction + 1, component, k, j, i) =
              flux.flux[component];
        const int tangent1 = (Direction + 1) % 3;
        const int tangent2 = (Direction + 2) % 3;
        const Real electric_t1 = flux.induction[tangent2];
        const Real electric_t2 = -flux.induction[tangent1];
        face_emf(block, emf_t1, k, j, i) = electric_t1;
        face_emf(block, emf_t2, k, j, i) = electric_t2;
      });
}

template <hydro::Reconstruction Method, riemann::Solver Solver>
void CalculateAllSyncGRMHDDirections(MeshData<Real>* data, const eos::RelativisticEOS& eos) {
  CalculateSyncGRMHDDirection<Method, Solver, 0>(data, eos);
  if (data->GetParentPointer()->ndim >= 2)
    CalculateSyncGRMHDDirection<Method, Solver, 1>(data, eos);
  if (data->GetParentPointer()->ndim >= 3)
    CalculateSyncGRMHDDirection<Method, Solver, 2>(data, eos);
}

template <class Primitive, class Conserved, class BCell, class Z4c, class ADM,
          class GeometryType>
KOKKOS_INLINE_FUNCTION void PrimitiveToConservedPoint(
    const Primitive& primitive, const Conserved& conserved, const BCell& bcell, const Z4c& z4c,
    const ADM& adm, const int block, const int k, const int j, const int i,
    const eos::RelativisticEOS& eos, const GeometryType& spacetime) {
  const auto& coordinates = primitive.GetCoords(block);
  const auto metric = spacetime.MetricAt(geometry::Location::cell_center, block, k, j, i,
                                         coordinates, z4c, adm);
  const Real inverse_volume = 1.0 / sqrt(metric.spatial_det);
  const relativity::MHDPrimitiveState state{
      {primitive(block, mhd::IDN, k, j, i),
       {primitive(block, mhd::IV1, k, j, i), primitive(block, mhd::IV2, k, j, i),
        primitive(block, mhd::IV3, k, j, i)},
       eos.PressureFromInternalEnergyDensity(
           primitive(block, mhd::IPR, k, j, i))},
      {bcell(block, 0, k, j, i) * inverse_volume,
       bcell(block, 1, k, j, i) * inverse_volume,
       bcell(block, 2, k, j, i) * inverse_volume}};
  const auto converted = ConvertSyncGRMHDP2C(state, eos, metric);
  conserved(block, mhd::IDN, k, j, i) = converted.density;
  conserved(block, mhd::IM1, k, j, i) = converted.momentum[0];
  conserved(block, mhd::IM2, k, j, i) = converted.momentum[1];
  conserved(block, mhd::IM3, k, j, i) = converted.momentum[2];
  conserved(block, mhd::IEN, k, j, i) = converted.energy;
}

KOKKOS_INLINE_FUNCTION bool RequiredReconstructionCell(
    const int i, const int j, const int k, const int is, const int ie,
    const int js, const int je, const int ks, const int ke,
    const int stencil_radius, const int ndim) {
  const bool near_i = i >= is - 1 && i <= ie + 1;
  const bool near_j = j >= js - 1 && j <= je + 1;
  const bool near_k = k >= ks - 1 && k <= ke + 1;
  const bool wide_i = i >= is - stencil_radius && i <= ie + stencil_radius;
  const bool wide_j = j >= js - stencil_radius && j <= je + stencil_radius;
  const bool wide_k = k >= ks - stencil_radius && k <= ke + stencil_radius;
  const bool inside_i = i >= is && i <= ie;
  const bool inside_j = j >= js && j <= je;
  const bool inside_k = k >= ks && k <= ke;
  if (wide_i && near_j && near_k && (inside_j || inside_k))
    return true;
  if (ndim >= 2 && near_i && wide_j && near_k && (inside_i || inside_k))
    return true;
  return ndim >= 3 && near_i && near_j && wide_k && (inside_i || inside_j);
}

template <int Direction>
void ReplaceSyncGRMHDFlaggedFaces(MeshData<Real>* data, const eos::RelativisticEOS& eos) {
  using TE = parthenon::TopologicalElement;
  const auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  const auto magnetic = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  const auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
  auto face_emf = data->PackVariables(std::vector<std::string>{"mhd.face_emf"});
  const auto bface = data->PackVariables(std::vector<std::string>{"mhd.b_face"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int ndim = data->GetParentPointer()->ndim;
  int il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  if constexpr (Direction == 0) {
    ++iu;
    if (ndim >= 2) {
      --jl;
      ++ju;
    }
    if (ndim >= 3) {
      --kl;
      ++ku;
    }
  } else if constexpr (Direction == 1) {
    ++ju;
    --il;
    ++iu;
    if (ndim >= 3) {
      --kl;
      ++ku;
    }
  } else {
    ++ku;
    --jl;
    ++ju;
    --il;
    ++iu;
  }
  constexpr TE face = Direction == 0 ? TE::F1 : (Direction == 1 ? TE::F2 : TE::F3);
  constexpr int emf_t1 = Direction == 0 ? E2X1 : (Direction == 1 ? E3X2 : E1X3);
  constexpr int emf_t2 = Direction == 0 ? E3X1 : (Direction == 1 ? E1X2 : E2X3);
  const int interior_is = ib.s;
  const int interior_ie = ib.e;
  const int interior_js = jb.s;
  const int interior_je = jb.e;
  const int interior_ks = kb.s;
  const int interior_ke = kb.e;
  const auto spacetime = geometry::GetGeometry(data->GetParentPointer()->packages);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU sync GRMHD stage-local FOFC faces",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const bool outside_i = i < interior_is || i > interior_ie;
        const bool outside_j = j < interior_js || j > interior_je;
        const bool outside_k = k < interior_ks || k > interior_ke;
        if constexpr (Direction == 0) {
          if (outside_j && outside_k)
            return;
        } else if constexpr (Direction == 1) {
          if (outside_i && outside_k)
            return;
        } else if (outside_i && outside_j) {
          return;
        }
        const int left_i = i - (Direction == 0);
        const int left_j = j - (Direction == 1);
        const int left_k = k - (Direction == 2);
        if (!(flags(block, 0, left_k, left_j, left_i) > 0.5 ||
              flags(block, 0, k, j, i) > 0.5))
          return;

        const auto& coordinates = primitive.GetCoords(block);
        const auto metric = spacetime.MetricAt(geometry::FaceLocation(Direction), block, k, j, i,
                                               coordinates, z4c, adm);
        const Real inverse_volume = 1.0 / sqrt(metric.spatial_det);
        relativity::MHDPrimitiveState left{
            {primitive(block, mhd::IDN, left_k, left_j, left_i),
             {primitive(block, mhd::IV1, left_k, left_j, left_i),
              primitive(block, mhd::IV2, left_k, left_j, left_i),
              primitive(block, mhd::IV3, left_k, left_j, left_i)},
             eos.PressureFromInternalEnergyDensity(
                 primitive(block, mhd::IPR, left_k, left_j, left_i))},
            {magnetic(block, 0, left_k, left_j, left_i) * inverse_volume,
             magnetic(block, 1, left_k, left_j, left_i) * inverse_volume,
             magnetic(block, 2, left_k, left_j, left_i) * inverse_volume}};
        relativity::MHDPrimitiveState right{
            {primitive(block, mhd::IDN, k, j, i),
             {primitive(block, mhd::IV1, k, j, i), primitive(block, mhd::IV2, k, j, i),
              primitive(block, mhd::IV3, k, j, i)},
             eos.PressureFromInternalEnergyDensity(
                 primitive(block, mhd::IPR, k, j, i))},
            {magnetic(block, 0, k, j, i) * inverse_volume,
             magnetic(block, 1, k, j, i) * inverse_volume,
             magnetic(block, 2, k, j, i) * inverse_volume}};
        const Real normal = bface(block, face, 0, k, j, i) * inverse_volume;
        left.magnetic[Direction] = normal;
        right.magnetic[Direction] = normal;
        const auto left_state = BuildSyncGRMHDFluxState(left, eos, Direction, metric);
        const auto right_state = BuildSyncGRMHDFluxState(right, eos, Direction, metric);
        const auto flux = FirstOrderSolver::Solve(left_state, right_state, metric);
        auto block_conserved = conserved(block);
        for (int component = 0; component < 5; ++component)
          block_conserved.flux(Direction + 1, component, k, j, i) = flux.flux[component];
        const int tangent1 = (Direction + 1) % 3;
        const int tangent2 = (Direction + 2) % 3;
        face_emf(block, emf_t1, k, j, i) = flux.induction[tangent2];
        face_emf(block, emf_t2, k, j, i) = -flux.induction[tangent1];
      });
}

} // namespace

TaskStatus CalculateSyncGRMHDFluxesMeshTask(MeshData<Real>* data) {
  const auto package = data->GetParentPointer()->packages.Get("mhd");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  const auto reconstruction =
      static_cast<hydro::Reconstruction>(package->Param<int>("reconstruction"));
  const auto solver = static_cast<riemann::Solver>(package->Param<int>("riemann"));
  PARTHENON_REQUIRE(riemann::Describe(solver).relativistic_mhd,
                    "sync GRMHD requires a Riemann solver registered for relativistic MHD");
  riemann::Visit<riemann::Physics::relativistic_mhd>(solver, [&]<riemann::Solver Solver>() {
    reconstruct::VisitRelativisticMHD(reconstruction, [&]<hydro::Reconstruction Method>() {
      CalculateAllSyncGRMHDDirections<Method, Solver>(data, eos);
    });
  });
  return TaskStatus::complete;
}

TaskStatus ResetSyncGRMHDDiagnosticsMeshTask(MeshData<Real>* data) {
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto recovery = data->PackVariables(std::vector<std::string>{"mhd.recovery"});
  const auto ib = data->GetBoundsI(IndexDomain::entire);
  const auto jb = data->GetBoundsJ(IndexDomain::entire);
  const auto kb = data->GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU reset sync GRMHD stage diagnostics",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        flags(block, 0, k, j, i) = 0.0;
        for (int component = 0; component < 3; ++component)
          recovery(block, component, k, j, i) = 0.0;
      });
  return TaskStatus::complete;
}

TaskStatus ApplySyncGRMHDFluxCorrectionMeshTask(MeshData<Real>* data,
                                                 MeshData<Real>* base,
                                                 const Real gam0,
                                                 const Real gam1,
                                                 const Real beta_dt) {
  using TE = parthenon::TopologicalElement;
  auto* mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("mhd");
  if (!package->Param<bool>("fofc"))
    return TaskStatus::complete;
  const auto eos = pangu::eos::ReadMagnetised(*package);
  const Real gamma_max = package->Param<Real>("gamma_max");
  const Real sigma_max = package->Param<Real>("sigma_max");
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
  const auto base_conserved = base->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  const auto base_face = base->PackVariables(std::vector<std::string>{"mhd.b_face"});
  const auto face_emf = data->PackVariables(std::vector<std::string>{"mhd.face_emf"});
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int ndim = mesh->ndim;
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU sync GRMHD stage-local FOFC candidate",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, kb.s - 1,
      kb.e + (ndim >= 3), jb.s - 1, jb.e + (ndim >= 2), ib.s - 1, ib.e + 1,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const int outside = static_cast<int>(i < ib.s || i > ib.e) +
                            static_cast<int>(j < jb.s || j > jb.e) +
                            static_cast<int>(k < kb.s || k > kb.e);
        if (outside > 1)
          return;
        const auto& coordinates = conserved.GetCoords(block);
        const auto block_conserved = conserved(block);
        Real values[5]{};
        for (int component = 0; component < 5; ++component) {
          values[component] =
              gam0 * conserved(block, component, k, j, i) +
              gam1 * base_conserved(block, component, k, j, i) -
              beta_dt *
                  (block_conserved.flux(X1DIR, component, k, j, i + 1) -
                   block_conserved.flux(X1DIR, component, k, j, i)) /
                  coordinates.Dxc<X1DIR>(k, j, i);
          if (ndim >= 2)
            values[component] -=
                beta_dt *
                (block_conserved.flux(X2DIR, component, k, j + 1, i) -
                 block_conserved.flux(X2DIR, component, k, j, i)) /
                coordinates.Dxc<X2DIR>(k, j, i);
          if (ndim >= 3)
            values[component] -=
                beta_dt *
                (block_conserved.flux(X3DIR, component, k + 1, j, i) -
                 block_conserved.flux(X3DIR, component, k, j, i)) /
                coordinates.Dxc<X3DIR>(k, j, i);
        }
        const relativity::HydroConservedState candidate{
            values[mhd::IDN],
            {values[mhd::IM1], values[mhd::IM2], values[mhd::IM3]},
            values[mhd::IEN]};
        Real magnetic[3]{
            gam0 * bcell(block, 0, k, j, i) +
                0.5 * gam1 *
                    (base_face(block, TE::F1, 0, k, j, i) +
                     base_face(block, TE::F1, 0, k, j, i + 1)),
            gam0 * bcell(block, 1, k, j, i) +
                0.5 * gam1 *
                    (base_face(block, TE::F2, 0, k, j, i) +
                     base_face(block, TE::F2, 0, k, j + (ndim >= 2), i)),
            gam0 * bcell(block, 2, k, j, i) +
                0.5 * gam1 *
                    (base_face(block, TE::F3, 0, k, j, i) +
                     base_face(block, TE::F3, 0, k + (ndim >= 3), j, i))};
        magnetic[1] += beta_dt *
                       (face_emf(block, E3X1, k, j, i + 1) -
                        face_emf(block, E3X1, k, j, i)) /
                       coordinates.Dxc<X1DIR>(k, j, i);
        magnetic[2] -= beta_dt *
                       (face_emf(block, E2X1, k, j, i + 1) -
                        face_emf(block, E2X1, k, j, i)) /
                       coordinates.Dxc<X1DIR>(k, j, i);
        if (ndim >= 2) {
          magnetic[0] -= beta_dt *
                         (face_emf(block, E3X2, k, j + 1, i) -
                          face_emf(block, E3X2, k, j, i)) /
                         coordinates.Dxc<X2DIR>(k, j, i);
          magnetic[2] += beta_dt *
                         (face_emf(block, E1X2, k, j + 1, i) -
                          face_emf(block, E1X2, k, j, i)) /
                         coordinates.Dxc<X2DIR>(k, j, i);
        }
        if (ndim >= 3) {
          magnetic[0] += beta_dt *
                         (face_emf(block, E2X3, k + 1, j, i) -
                          face_emf(block, E2X3, k, j, i)) /
                         coordinates.Dxc<X3DIR>(k, j, i);
          magnetic[1] -= beta_dt *
                         (face_emf(block, E1X3, k + 1, j, i) -
                          face_emf(block, E1X3, k, j, i)) /
                         coordinates.Dxc<X3DIR>(k, j, i);
        }
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, block, k,
                                               j, i, coordinates, z4c, adm);
        const auto trial =
            SolveSyncGRMHDC2P(candidate, magnetic, eos, gamma_max, sigma_max, metric);
        const bool bad = !trial.success || trial.density_floor || trial.pressure_floor ||
                         trial.lorentz_ceiling || trial.sigma_ceiling;
        flags(block, 0, k, j, i) = bad ? 1.0 : 0.0;
      });
  ReplaceSyncGRMHDFlaggedFaces<0>(data, eos);
  if (ndim >= 2)
    ReplaceSyncGRMHDFlaggedFaces<1>(data, eos);
  if (ndim >= 3)
    ReplaceSyncGRMHDFlaggedFaces<2>(data, eos);
  return TaskStatus::complete;
}

TaskStatus ApplySyncGRMHDPunctureProtectionMeshTask(MeshData<Real>* data) {
  using TE = parthenon::TopologicalElement;
  auto* mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("mhd");
  if (package->Param<int>("puncture_protection") == 0)
    return TaskStatus::complete;
  const Real chi_threshold = package->Param<Real>("puncture_chi_threshold");
  const Real density = package->Param<Real>("puncture_density");
  const Real pressure = package->Param<Real>("puncture_pressure");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto bface = data->PackVariables(std::vector<std::string>{"mhd.b_face"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  auto recovery = data->PackVariables(std::vector<std::string>{"mhd.recovery"});
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int ndim = mesh->ndim;
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU sync GRMHD puncture atmosphere protection",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        if (z4c(block, Index(Z4cComponent::chi), k, j, i) >= chi_threshold)
          return;
        const auto& coordinates = conserved.GetCoords(block);
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, block, k,
                                               j, i, coordinates, z4c, adm);
        const Real inverse_volume = 1.0 / sqrt(metric.spatial_det);
        const Real densitized_magnetic[3]{
            0.5 * (bface(block, TE::F1, 0, k, j, i) +
                   bface(block, TE::F1, 0, k, j, i + 1)),
            0.5 * (bface(block, TE::F2, 0, k, j, i) +
                   bface(block, TE::F2, 0, k, j + (ndim >= 2), i)),
            0.5 * (bface(block, TE::F3, 0, k, j, i) +
                   bface(block, TE::F3, 0, k + (ndim >= 3), j, i))};
        const relativity::MHDPrimitiveState atmosphere{
            {density, {0.0, 0.0, 0.0}, pressure},
            {densitized_magnetic[0] * inverse_volume,
             densitized_magnetic[1] * inverse_volume,
             densitized_magnetic[2] * inverse_volume}};
        const auto reset = ConvertSyncGRMHDP2C(atmosphere, eos, metric);
        recovery(block, 0, k, j, i) += reset.density - conserved(block, mhd::IDN, k, j, i);
        recovery(block, 1, k, j, i) += reset.energy - conserved(block, mhd::IEN, k, j, i);
        recovery(block, 2, k, j, i) = 1.0;
        conserved(block, mhd::IDN, k, j, i) = reset.density;
        conserved(block, mhd::IM1, k, j, i) = reset.momentum[0];
        conserved(block, mhd::IM2, k, j, i) = reset.momentum[1];
        conserved(block, mhd::IM3, k, j, i) = reset.momentum[2];
        conserved(block, mhd::IEN, k, j, i) = reset.energy;
        flags(block, 0, k, j, i) = 1.0;
      });
  return TaskStatus::complete;
}

TaskStatus ApplySyncGRMHDSourcesMeshTask(MeshData<Real>* current,
                                         MeshData<Real>* next, const Real dt) {
  return ApplySyncMatterSourcesMeshTask(current, next, dt, "mhd.cons");
}

void SyncGRMHDPrimitiveToConservedBlock(MeshBlockData<Real>* data) {
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto block = data->GetBlockPointer();
  const auto eos = pangu::eos::ReadRelativistic(*block->packages.Get("mhd"));
  const auto spacetime = geometry::GetGeometry(block->packages);
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  block->par_for(
      "PANGU sync GRMHD primitive to conserved", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, 0, k, j,
                                               i, block->coords, z4c, adm);
        const Real inverse_volume = 1.0 / sqrt(metric.spatial_det);
        const relativity::MHDPrimitiveState state{
            {primitive(mhd::IDN, k, j, i),
             {primitive(mhd::IV1, k, j, i), primitive(mhd::IV2, k, j, i),
              primitive(mhd::IV3, k, j, i)},
             eos.PressureFromInternalEnergyDensity(
                 primitive(mhd::IPR, k, j, i))},
            {bcell(0, k, j, i) * inverse_volume,
             bcell(1, k, j, i) * inverse_volume,
             bcell(2, k, j, i) * inverse_volume}};
        const auto converted = ConvertSyncGRMHDP2C(state, eos, metric);
        conserved(mhd::IDN, k, j, i) = converted.density;
        conserved(mhd::IM1, k, j, i) = converted.momentum[0];
        conserved(mhd::IM2, k, j, i) = converted.momentum[1];
        conserved(mhd::IM3, k, j, i) = converted.momentum[2];
        conserved(mhd::IEN, k, j, i) = converted.energy;
      });
}

void SyncGRMHDPrimitiveToConservedMesh(MeshData<Real>* data) {
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto mesh = data->GetParentPointer();
  const auto eos = pangu::eos::ReadRelativistic(*mesh->packages.Get("mhd"));
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const auto ib = data->GetBoundsI(IndexDomain::entire);
  const auto jb = data->GetBoundsJ(IndexDomain::entire);
  const auto kb = data->GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed sync GRMHD primitive to conserved",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        PrimitiveToConservedPoint(primitive, conserved, bcell, z4c, adm, block, k, j, i,
                                  eos, spacetime);
      });
}

void SyncGRMHDConservedToPrimitiveBlock(MeshBlockData<Real>* data) {
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto divb = data->PackVariables(std::vector<std::string>{"mhd.divb"});
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto recovery = data->PackVariables(std::vector<std::string>{"mhd.recovery"});
  const auto bface = data->Get("mhd.b_face").data;
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("mhd");
  const auto eos = pangu::eos::ReadMagnetised(*package);
  const Real gamma_max = package->Param<Real>("gamma_max");
  const Real sigma_max = package->Param<Real>("sigma_max");
  const auto reconstruction =
      static_cast<hydro::Reconstruction>(package->Param<int>("reconstruction"));
  const int stencil_radius = reconstruct::Describe(reconstruction).stencil_radius;
  const int ndim = block->pmy_mesh->ndim;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto interior_ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto interior_jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto interior_kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  block->par_for(
      "PANGU sync GRMHD conserved to primitive", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (!RequiredReconstructionCell(i, j, k, interior_ib.s, interior_ib.e,
                                        interior_jb.s, interior_jb.e, interior_kb.s,
                                        interior_kb.e, stencil_radius, ndim))
          return;
        Real densitized_magnetic[3]{
            0.5 * (bface(0, 0, 0, 0, k, j, i) +
                   bface(0, 0, 0, 0, k, j, i + 1)),
            0.5 * (bface(1, 0, 0, 0, k, j, i) +
                   bface(1, 0, 0, 0, k, j + (ndim >= 2), i)),
            0.5 * (bface(2, 0, 0, 0, k, j, i) +
                   bface(2, 0, 0, 0, k + (ndim >= 3), j, i))};
        for (int axis = 0; axis < 3; ++axis)
          bcell(axis, k, j, i) = densitized_magnetic[axis];
        Real divergence =
            (bface(0, 0, 0, 0, k, j, i + 1) - bface(0, 0, 0, 0, k, j, i)) /
            block->coords.Dxc<X1DIR>(k, j, i);
        if (ndim >= 2)
          divergence +=
              (bface(1, 0, 0, 0, k, j + 1, i) - bface(1, 0, 0, 0, k, j, i)) /
              block->coords.Dxc<X2DIR>(k, j, i);
        if (ndim >= 3)
          divergence +=
              (bface(2, 0, 0, 0, k + 1, j, i) - bface(2, 0, 0, 0, k, j, i)) /
              block->coords.Dxc<X3DIR>(k, j, i);
        divb(0, k, j, i) = divergence;
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, 0, k, j,
                                               i, block->coords, z4c, adm);
        const relativity::HydroConservedState state{
            conserved(mhd::IDN, k, j, i),
            {conserved(mhd::IM1, k, j, i), conserved(mhd::IM2, k, j, i),
             conserved(mhd::IM3, k, j, i)},
            conserved(mhd::IEN, k, j, i)};
        const auto result = SolveSyncGRMHDC2P(state, densitized_magnetic, eos, gamma_max,
                                             sigma_max, metric);
        primitive(mhd::IDN, k, j, i) = result.primitive.fluid.density;
        primitive(mhd::IV1, k, j, i) = result.primitive.fluid.u[0];
        primitive(mhd::IV2, k, j, i) = result.primitive.fluid.u[1];
        primitive(mhd::IV3, k, j, i) = result.primitive.fluid.u[2];
        primitive(mhd::IPR, k, j, i) = result.internal_energy;
        const bool repaired = !result.success || result.density_floor ||
                              result.pressure_floor || result.lorentz_ceiling ||
                              result.sigma_ceiling;
        flags(0, k, j, i) =
            (flags(0, k, j, i) > 0.5 || repaired) ? 1.0 : 0.0;
        if (repaired) {
          const auto fixed = ConvertSyncGRMHDP2C(result.primitive, eos, metric);
          recovery(0, k, j, i) += fixed.density - conserved(mhd::IDN, k, j, i);
          recovery(1, k, j, i) += fixed.energy - conserved(mhd::IEN, k, j, i);
          conserved(mhd::IDN, k, j, i) = fixed.density;
          conserved(mhd::IM1, k, j, i) = fixed.momentum[0];
          conserved(mhd::IM2, k, j, i) = fixed.momentum[1];
          conserved(mhd::IM3, k, j, i) = fixed.momentum[2];
          conserved(mhd::IEN, k, j, i) = fixed.energy;
        }
      });
}

void SyncGRMHDConservedToPrimitiveMesh(MeshData<Real>* data) {
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto divb = data->PackVariables(std::vector<std::string>{"mhd.divb"});
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto recovery = data->PackVariables(std::vector<std::string>{"mhd.recovery"});
  const auto bface = data->PackVariables(std::vector<std::string>{"mhd.b_face"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("mhd");
  const auto eos = pangu::eos::ReadMagnetised(*package);
  const Real gamma_max = package->Param<Real>("gamma_max");
  const Real sigma_max = package->Param<Real>("sigma_max");
  const auto reconstruction =
      static_cast<hydro::Reconstruction>(package->Param<int>("reconstruction"));
  const int stencil_radius = reconstruct::Describe(reconstruction).stencil_radius;
  const int ndim = mesh->ndim;
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const auto ib = data->GetBoundsI(IndexDomain::entire);
  const auto jb = data->GetBoundsJ(IndexDomain::entire);
  const auto kb = data->GetBoundsK(IndexDomain::entire);
  const auto interior_ib = data->GetBoundsI(IndexDomain::interior);
  const auto interior_jb = data->GetBoundsJ(IndexDomain::interior);
  const auto interior_kb = data->GetBoundsK(IndexDomain::interior);
  using TE = parthenon::TopologicalElement;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed sync GRMHD conserved to primitive",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        if (!RequiredReconstructionCell(i, j, k, interior_ib.s, interior_ib.e,
                                        interior_jb.s, interior_jb.e, interior_kb.s,
                                        interior_kb.e, stencil_radius, ndim))
          return;
        Real densitized_magnetic[3]{
            0.5 * (bface(block, TE::F1, 0, k, j, i) +
                   bface(block, TE::F1, 0, k, j, i + 1)),
            0.5 * (bface(block, TE::F2, 0, k, j, i) +
                   bface(block, TE::F2, 0, k, j + (ndim >= 2), i)),
            0.5 * (bface(block, TE::F3, 0, k, j, i) +
                   bface(block, TE::F3, 0, k + (ndim >= 3), j, i))};
        for (int axis = 0; axis < 3; ++axis)
          bcell(block, axis, k, j, i) = densitized_magnetic[axis];
        const auto& coordinates = conserved.GetCoords(block);
        Real divergence = (bface(block, TE::F1, 0, k, j, i + 1) -
                           bface(block, TE::F1, 0, k, j, i)) /
                          coordinates.Dxc<X1DIR>(k, j, i);
        if (ndim >= 2)
          divergence += (bface(block, TE::F2, 0, k, j + 1, i) -
                         bface(block, TE::F2, 0, k, j, i)) /
                        coordinates.Dxc<X2DIR>(k, j, i);
        if (ndim >= 3)
          divergence += (bface(block, TE::F3, 0, k + 1, j, i) -
                         bface(block, TE::F3, 0, k, j, i)) /
                        coordinates.Dxc<X3DIR>(k, j, i);
        divb(block, 0, k, j, i) = divergence;
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, block, k,
                                               j, i, coordinates, z4c, adm);
        const relativity::HydroConservedState state{
            conserved(block, mhd::IDN, k, j, i),
            {conserved(block, mhd::IM1, k, j, i),
             conserved(block, mhd::IM2, k, j, i),
             conserved(block, mhd::IM3, k, j, i)},
            conserved(block, mhd::IEN, k, j, i)};
        const auto result = SolveSyncGRMHDC2P(state, densitized_magnetic, eos, gamma_max,
                                             sigma_max, metric);
        primitive(block, mhd::IDN, k, j, i) = result.primitive.fluid.density;
        primitive(block, mhd::IV1, k, j, i) = result.primitive.fluid.u[0];
        primitive(block, mhd::IV2, k, j, i) = result.primitive.fluid.u[1];
        primitive(block, mhd::IV3, k, j, i) = result.primitive.fluid.u[2];
        primitive(block, mhd::IPR, k, j, i) = result.internal_energy;
        const bool repaired = !result.success || result.density_floor ||
                              result.pressure_floor || result.lorentz_ceiling ||
                              result.sigma_ceiling;
        flags(block, 0, k, j, i) =
            (flags(block, 0, k, j, i) > 0.5 || repaired) ? 1.0 : 0.0;
        if (repaired) {
          const auto fixed = ConvertSyncGRMHDP2C(result.primitive, eos, metric);
          recovery(block, 0, k, j, i) +=
              fixed.density - conserved(block, mhd::IDN, k, j, i);
          recovery(block, 1, k, j, i) +=
              fixed.energy - conserved(block, mhd::IEN, k, j, i);
          conserved(block, mhd::IDN, k, j, i) = fixed.density;
          conserved(block, mhd::IM1, k, j, i) = fixed.momentum[0];
          conserved(block, mhd::IM2, k, j, i) = fixed.momentum[1];
          conserved(block, mhd::IM3, k, j, i) = fixed.momentum[2];
          conserved(block, mhd::IEN, k, j, i) = fixed.energy;
        }
      });
}

TaskStatus SyncGRMHDConservedToPrimitiveMeshTask(MeshData<Real>* data) {
  SyncGRMHDConservedToPrimitiveMesh(data);
  return TaskStatus::complete;
}

TaskStatus BuildSyncGRMHDStressEnergyBlockTask(MeshBlockData<Real>* data) {
  const auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  const auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  auto tmunu = data->PackVariables(std::vector<std::string>{"nr.tmunu"});
  const auto block = data->GetBlockPointer();
  const auto eos = pangu::eos::ReadRelativistic(*block->packages.Get("mhd"));
  const auto spacetime = geometry::GetGeometry(block->packages);
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  block->par_for(
      "PANGU sync GRMHD stress-energy initialization", kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, 0, k, j,
                                               i, block->coords, z4c, adm);
        const Real inverse_volume = 1.0 / sqrt(metric.spatial_det);
        const relativity::MHDPrimitiveState primitive_state{
            {primitive(mhd::IDN, k, j, i),
             {primitive(mhd::IV1, k, j, i), primitive(mhd::IV2, k, j, i),
              primitive(mhd::IV3, k, j, i)},
             eos.PressureFromInternalEnergyDensity(
                 primitive(mhd::IPR, k, j, i))},
            {bcell(0, k, j, i) * inverse_volume,
             bcell(1, k, j, i) * inverse_volume,
             bcell(2, k, j, i) * inverse_volume}};
        const relativity::HydroConservedState conserved_state{
            conserved(mhd::IDN, k, j, i),
            {conserved(mhd::IM1, k, j, i), conserved(mhd::IM2, k, j, i),
             conserved(mhd::IM3, k, j, i)},
            conserved(mhd::IEN, k, j, i)};
        const auto matter =
            BuildSyncGRMHDStressEnergy(primitive_state, conserved_state, metric);
        for (int component = 0; component < 6; ++component)
          tmunu(component, k, j, i) = matter.stress[component];
        tmunu(Index(StressEnergyComponent::energy), k, j, i) = matter.energy;
        for (int axis = 0; axis < 3; ++axis)
          tmunu(Index(StressEnergyComponent::momentum_x) + axis, k, j, i) =
              matter.momentum[axis];
      });
  return TaskStatus::complete;
}

TaskStatus BuildSyncGRMHDStressEnergyMeshTask(MeshData<Real>* data) {
  const auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  const auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  auto tmunu = data->PackVariables(std::vector<std::string>{"nr.tmunu"});
  const auto mesh = data->GetParentPointer();
  const auto eos = pangu::eos::ReadRelativistic(*mesh->packages.Get("mhd"));
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed sync GRMHD stress-energy",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const auto& coordinates = primitive.GetCoords(block);
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center, block, k,
                                               j, i, coordinates, z4c, adm);
        const Real inverse_volume = 1.0 / sqrt(metric.spatial_det);
        const relativity::MHDPrimitiveState primitive_state{
            {primitive(block, mhd::IDN, k, j, i),
             {primitive(block, mhd::IV1, k, j, i),
              primitive(block, mhd::IV2, k, j, i),
              primitive(block, mhd::IV3, k, j, i)},
             eos.PressureFromInternalEnergyDensity(
                 primitive(block, mhd::IPR, k, j, i))},
            {bcell(block, 0, k, j, i) * inverse_volume,
             bcell(block, 1, k, j, i) * inverse_volume,
             bcell(block, 2, k, j, i) * inverse_volume}};
        const relativity::HydroConservedState conserved_state{
            conserved(block, mhd::IDN, k, j, i),
            {conserved(block, mhd::IM1, k, j, i),
             conserved(block, mhd::IM2, k, j, i),
             conserved(block, mhd::IM3, k, j, i)},
            conserved(block, mhd::IEN, k, j, i)};
        const auto matter =
            BuildSyncGRMHDStressEnergy(primitive_state, conserved_state, metric);
        for (int component = 0; component < 6; ++component)
          tmunu(block, component, k, j, i) = matter.stress[component];
        tmunu(block, Index(StressEnergyComponent::energy), k, j, i) = matter.energy;
        for (int axis = 0; axis < 3; ++axis)
          tmunu(block, Index(StressEnergyComponent::momentum_x) + axis, k, j, i) =
              matter.momentum[axis];
      });
  return TaskStatus::complete;
}

Real EstimateSyncGRMHDTimestepBlock(MeshBlockData<Real>* data) {
  const auto block = data->GetBlockPointer();
  const Real cfl = block->packages.Get("mhd")->Param<Real>("cfl");
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

Real EstimateSyncGRMHDTimestepMesh(MeshData<Real>* data) {
  Real minimum = std::numeric_limits<Real>::max();
  for (int block = 0; block < data->NumBlocks(); ++block)
    minimum = fmin(minimum,
                   EstimateSyncGRMHDTimestepBlock(data->GetBlockData(block).get()));
  return minimum;
}

} // namespace pangu::nr
