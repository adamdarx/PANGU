#include "z4c/evolution/package.h"
#include "z4c/evolution/tasks.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "geometry_assembly.h"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "z4c/core/component_indices.h"
#include "z4c/evolution/constraints.h"
#include "z4c/core/finite_difference.h"
#include "z4c/evolution/amr.h"
#include "z4c/coupling/stress_energy.h"
#include "z4c/coupling/sync_grhd.h"
#include "z4c/coupling/sync_grmhd.h"
#include "z4c/initial_data/tov.h"
#include "z4c/diagnostics/weyl.h"
#include "z4c/evolution/z4c_rhs.h"
#include "pangu.h"
#include "utils/error_checking.hpp"
#include "utils/summable_array.hpp"

namespace pangu::nr {
using namespace parthenon::package::prelude;

namespace {

template <class Z4cPack, class ADMPack>
KOKKOS_INLINE_FUNCTION void ConvertPoint(const Z4cPack& z4c, const ADMPack& adm, const int block,
                                         const int k, const int j, const int i,
                                         const Real chi_psi_power) {
  const Real chi = z4c(block, Index(Z4cComponent::chi), k, j, i);
  const Real inverse_chi = pow(chi, 4.0 / chi_psi_power);
  const Real trace_k = z4c(block, Index(Z4cComponent::khat), k, j, i) +
                       2.0 * z4c(block, Index(Z4cComponent::theta), k, j, i);
  for (int component = 0; component < 6; ++component) {
    const Real conformal_metric = z4c(block, Index(Z4cComponent::gxx) + component, k, j, i);
    const Real conformal_a = z4c(block, Index(Z4cComponent::axx) + component, k, j, i);
    adm(block, Index(ADMComponent::gxx) + component, k, j, i) = inverse_chi * conformal_metric;
    adm(block, Index(ADMComponent::kxx) + component, k, j, i) =
        inverse_chi * (conformal_a + conformal_metric * trace_k / 3.0);
  }
  adm(block, Index(ADMComponent::psi4), k, j, i) = inverse_chi;
}

template <int Component> Real MaxConstraintComponentMesh(MeshData<Real>* data) {
  const auto block = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto constraints = data->PackVariables(std::vector<std::string>{"nr.constraints"});
  const auto mask = data->PackVariables(std::vector<std::string>{"nr.constraint_mask"});
  Real maximum = 0.0;
  ParReduce(
      "PANGU NR constraint history", 0, constraints.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        if (mask(b, 0, k, j, i) > 0.5)
          local = fmax(local, fabs(constraints(b, Component, k, j, i)));
      },
      Kokkos::Max<Real>(maximum));
  return maximum;
}

parthenon::summable_array_t<Real, 24> ConstraintSumsMesh(MeshData<Real>* data) {
  const auto block = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto constraints = data->PackVariables(std::vector<std::string>{"nr.constraints"});
  const auto mask = data->PackVariables(std::vector<std::string>{"nr.constraint_mask"});
  parthenon::summable_array_t<Real, 24> sums;
  ParReduce(
      "PANGU NR constraint L1/L2 history", 0, constraints.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    parthenon::summable_array_t<Real, 24>& local) {
        if (mask(b, 0, k, j, i) <= 0.5)
          return;
        const Real volume = constraints.GetCoords(b).CellVolume(k, j, i);
        for (int component = 0; component < kConstraintComponents; ++component) {
          const Real value = constraints(b, component, k, j, i);
          local[3 * component] += fabs(value) * volume;
          local[3 * component + 1] += value * value * volume;
          local[3 * component + 2] += volume;
        }
      },
      Kokkos::Sum<parthenon::summable_array_t<Real, 24>>(sums));
  return sums;
}

std::vector<Real> ConstraintMaxHistoryMesh(MeshData<Real>* data) {
  return {MaxConstraintComponentMesh<Index(ConstraintComponent::combined)>(data),
          MaxConstraintComponentMesh<Index(ConstraintComponent::hamiltonian)>(data),
          MaxConstraintComponentMesh<Index(ConstraintComponent::momentum_norm)>(data),
          MaxConstraintComponentMesh<Index(ConstraintComponent::z_norm)>(data),
          MaxConstraintComponentMesh<Index(ConstraintComponent::momentum_x)>(data),
          MaxConstraintComponentMesh<Index(ConstraintComponent::momentum_y)>(data),
          MaxConstraintComponentMesh<Index(ConstraintComponent::momentum_z)>(data),
          MaxConstraintComponentMesh<Index(ConstraintComponent::theta)>(data)};
}

std::vector<Real> ConstraintNormHistoryMesh(MeshData<Real>* data) {
  auto sums = ConstraintSumsMesh(data);
  Real global_sums[24]{};
  Kokkos::fence();
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(sums.data(), global_sums, 24, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#else
  for (int index = 0; index < 24; ++index)
    global_sums[index] = sums[index];
#endif
  std::vector<Real> history;
  history.reserve(16);
  for (int component = 0; component < kConstraintComponents; ++component) {
    const Real volume = global_sums[3 * component + 2];
    const Real inverse_volume = volume > 0.0 ? 1.0 / volume : 0.0;
    history.push_back(global_sums[3 * component] * inverse_volume);
    history.push_back(sqrt(global_sums[3 * component + 1] * inverse_volume));
  }
  return history;
}

std::vector<Real> ConstraintMaskHistoryMesh(MeshData<Real>* data) {
  const auto block = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto mask = data->PackVariables(std::vector<std::string>{"nr.constraint_mask"});
  parthenon::summable_array_t<Real, 2> counts{};
  ParReduce(
      "PANGU NR constraint mask history", 0, mask.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    parthenon::summable_array_t<Real, 2>& local) {
        const int index = mask(b, 0, k, j, i) > 0.5 ? 1 : 0;
        local[index] += 1.0;
      },
      Kokkos::Sum<parthenon::summable_array_t<Real, 2>>(counts));
  return {counts[0], counts[1]};
}

// Record every ODE tracker as (x,y,z,vx,vy,vz). Only rank zero contributes to
// Parthenon's subsequent history sum reduction, so MPI does not multiply the
// replicated package parameters.
std::vector<Real> PunctureTrackerHistoryMesh(MeshData<Real>* data) {
  const auto first = data->GetBlockData(0)->GetBlockPointer();
  const auto package = first->packages.Get("numerical_relativity");
  const int tracker_count = package->Param<int>("tracker_count");
  const auto& position = package->Param<std::vector<Real>>("tracker_position");
  const auto& velocity = package->Param<std::vector<Real>>("tracker_velocity");
  PARTHENON_REQUIRE(position.size() == static_cast<std::size_t>(3 * tracker_count) &&
                        velocity.size() == position.size(),
                    "invalid NR tracker history state");
  std::vector<Real> result(6 * tracker_count, 0.0);
  if (parthenon::Globals::my_rank == 0) {
    for (int tracker = 0; tracker < tracker_count; ++tracker) {
      for (int axis = 0; axis < 3; ++axis) {
        result[6 * tracker + axis] = position[3 * tracker + axis];
        result[6 * tracker + 3 + axis] = velocity[3 * tracker + axis];
      }
    }
  }
  return result;
}

template <class Pack>
KOKKOS_INLINE_FUNCTION void
ApplySommerfeldBoundaryRHS(const Pack& input, const int block, const int k, const int j,
                           const int i, const Real inverse_spacing[3], const Real x, const Real y,
                           const Real z, Real output[kZ4cComponents]) {
  // AthenaK's Z4c outer boundary condition deliberately uses second-order
  // centered derivatives even when the volume stencil is higher order.
  const Real radius = sqrt(x * x + y * y + z * z);
  if (radius <= 1.0e-14)
    return;
  const Real radial[3] = {x / radius, y / radius, z / radius};
  const auto accessor = [&](const int component) {
    return rhs::ComponentAccessor<Pack>{input, block, component, k, j, i};
  };
  Real dkhat[3]{}, dtheta[3]{}, dgamma[3][3]{}, da[3][6]{};
  for (int direction = 0; direction < 3; ++direction) {
    dkhat[direction] =
        fd::First<2>(direction, inverse_spacing[direction], accessor(Index(Z4cComponent::khat)));
    dtheta[direction] =
        fd::First<2>(direction, inverse_spacing[direction], accessor(Index(Z4cComponent::theta)));
    for (int component = 0; component < 3; ++component) {
      dgamma[component][direction] = fd::First<2>(direction, inverse_spacing[direction],
                                                  accessor(Index(Z4cComponent::gamx) + component));
    }
    for (int component = 0; component < 6; ++component) {
      da[direction][component] = fd::First<2>(direction, inverse_spacing[direction],
                                              accessor(Index(Z4cComponent::axx) + component));
    }
  }

  Real radial_dkhat = 0.0;
  Real radial_dtheta = 0.0;
  for (int direction = 0; direction < 3; ++direction) {
    radial_dkhat += radial[direction] * dkhat[direction];
    radial_dtheta += radial[direction] * dtheta[direction];
  }
  const Real sqrt_two = sqrt(2.0);
  output[Index(Z4cComponent::khat)] =
      -sqrt_two * input(block, Index(Z4cComponent::khat), k, j, i) / radius -
      sqrt_two * radial_dkhat;
  output[Index(Z4cComponent::theta)] =
      -input(block, Index(Z4cComponent::theta), k, j, i) / radius - radial_dtheta;
  for (int component = 0; component < 3; ++component) {
    Real radial_dgamma = 0.0;
    for (int direction = 0; direction < 3; ++direction)
      radial_dgamma += radial[direction] * dgamma[component][direction];
    output[Index(Z4cComponent::gamx) + component] =
        -input(block, Index(Z4cComponent::gamx) + component, k, j, i) / radius - radial_dgamma;
  }
  for (int component = 0; component < 6; ++component) {
    Real radial_da = 0.0;
    for (int direction = 0; direction < 3; ++direction)
      radial_da += radial[direction] * da[direction][component];
    output[Index(Z4cComponent::axx) + component] =
        -input(block, Index(Z4cComponent::axx) + component, k, j, i) / radius - radial_da;
  }
}

template <parthenon::CoordinateDirection Direction, bool Inner>
void ApplyPackedOutflowFace(MeshData<Real>* data) {
  constexpr IndexDomain domain =
      Inner ? (Direction == X1DIR ? IndexDomain::inner_x1
                                 : (Direction == X2DIR ? IndexDomain::inner_x2
                                                       : IndexDomain::inner_x3))
            : (Direction == X1DIR ? IndexDomain::outer_x1
                                 : (Direction == X2DIR ? IndexDomain::outer_x2
                                                       : IndexDomain::outer_x3));
  auto* mesh = data->GetMeshPointer();
  const auto package = mesh->packages.Get("numerical_relativity");
  const int extrapolation_order = package->Param<int>("boundary_extrapolation_order");
  const auto cell_ib = data->GetBoundsI(IndexDomain::interior);
  const auto cell_jb = data->GetBoundsJ(IndexDomain::interior);
  const auto cell_kb = data->GetBoundsK(IndexDomain::interior);
  const Real domain_coordinate =
      Inner ? mesh->mesh_size.xmin(Direction) : mesh->mesh_size.xmax(Direction);

  using TE = parthenon::TopologicalElement;
  const std::array<TE, 8> elements{TE::CC, TE::F1, TE::F2, TE::F3,
                                   TE::E1, TE::E2, TE::E3, TE::NN};
  for (const auto element : elements) {
    parthenon::MetadataFlag topology = Metadata::Cell;
    switch (parthenon::GetTopologicalType(element)) {
    case parthenon::TopologicalType::Cell:
      topology = Metadata::Cell;
      break;
    case parthenon::TopologicalType::Face:
      topology = Metadata::Face;
      break;
    case parthenon::TopologicalType::Edge:
      topology = Metadata::Edge;
      break;
    case parthenon::TopologicalType::Node:
      topology = Metadata::Node;
      break;
    default:
      PARTHENON_FAIL("unknown topological element in packed SYNC boundary");
    }
    const Metadata::FlagCollection flags{Metadata::FillGhost, topology};
    const auto names = mesh->GetVariableNames(flags);
    if (names.empty())
      continue;

    parthenon::PackIndexMap map;
    const auto fields = data->PackVariables(names, map);
    const auto z4c_range = map["nr.z4c"];
    const auto ib = data->GetBoundsI(domain, element);
    const auto jb = data->GetBoundsJ(domain, element);
    const auto kb = data->GetBoundsK(domain, element);
    const auto interior_i = data->GetBoundsI(IndexDomain::interior, element);
    const auto interior_j = data->GetBoundsJ(IndexDomain::interior, element);
    const auto interior_k = data->GetBoundsK(IndexDomain::interior, element);
    const int reference = Inner ? (Direction == X1DIR   ? interior_i.s
                                   : Direction == X2DIR ? interior_j.s
                                                        : interior_k.s)
                                : (Direction == X1DIR   ? interior_i.e
                                   : Direction == X2DIR ? interior_j.e
                                                        : interior_k.e);
    const int inward = Inner ? 1 : -1;
    const std::string label =
        std::string("PANGU packed SYNC outflow ") + (Inner ? "inner x" : "outer x") +
        std::to_string(static_cast<int>(Direction));
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, label, parthenon::DevExecSpace(), 0, fields.GetDim(5) - 1, 0,
        fields.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int block, const int component, const int k, const int j,
                      const int i) {
          const auto& coordinates = fields.GetCoords(block);
          Real block_face_coordinate = 0.0;
          Real spacing = 1.0;
          if (Direction == X1DIR) {
            block_face_coordinate = coordinates.Xf<X1DIR>(cell_kb.s, cell_jb.s,
                                                          Inner ? cell_ib.s : cell_ib.e + 1);
            spacing = coordinates.Dxc<X1DIR>(cell_kb.s, cell_jb.s, cell_ib.s);
          } else if (Direction == X2DIR) {
            block_face_coordinate = coordinates.Xf<X2DIR>(cell_kb.s,
                                                          Inner ? cell_jb.s : cell_jb.e + 1,
                                                          cell_ib.s);
            spacing = coordinates.Dxc<X2DIR>(cell_kb.s, cell_jb.s, cell_ib.s);
          } else {
            block_face_coordinate = coordinates.Xf<X3DIR>(
                Inner ? cell_kb.s : cell_kb.e + 1, cell_jb.s, cell_ib.s);
            spacing = coordinates.Dxc<X3DIR>(cell_kb.s, cell_jb.s, cell_ib.s);
          }
          if (fabs(block_face_coordinate - domain_coordinate) >= 0.25 * spacing)
            return;

          const int coordinate = Direction == X1DIR ? i : (Direction == X2DIR ? j : k);
          const int distance = Inner ? reference - coordinate : coordinate - reference;
          const int source_k = Direction == X3DIR ? reference : k;
          const int source_j = Direction == X2DIR ? reference : j;
          const int source_i = Direction == X1DIR ? reference : i;
          const int offset_k = Direction == X3DIR ? inward : 0;
          const int offset_j = Direction == X2DIR ? inward : 0;
          const int offset_i = Direction == X1DIR ? inward : 0;
          const Real f0 = fields(block, element, component, source_k, source_j, source_i);
          const bool extrapolate_z4c =
              element == TE::CC && component >= z4c_range.first && component <= z4c_range.second &&
              extrapolation_order != 0;
          if (!extrapolate_z4c) {
            fields(block, element, component, k, j, i) = f0;
            return;
          }
          const Real delta = distance;
          const Real f1 = fields(block, element, component, source_k + offset_k,
                                 source_j + offset_j, source_i + offset_i);
          if (extrapolation_order == 2) {
            fields(block, element, component, k, j, i) = f0 + delta * (f0 - f1);
          } else {
            const Real f2 = fields(block, element, component, source_k + 2 * offset_k,
                                   source_j + 2 * offset_j, source_i + 2 * offset_i);
            if (extrapolation_order == 3) {
              fields(block, element, component, k, j, i) =
                  0.5 * (f0 * (1.0 + delta) * (2.0 + delta) +
                         delta * (f2 + delta * f2 - 2.0 * f1 * (2.0 + delta)));
            } else {
              const Real f3 = fields(block, element, component, source_k + 3 * offset_k,
                                     source_j + 3 * offset_j, source_i + 3 * offset_i);
              fields(block, element, component, k, j, i) =
                  (-3.0 * f1 * delta * (2.0 + delta) * (3.0 + delta) +
                   f0 * (1.0 + delta) * (2.0 + delta) * (3.0 + delta) +
                   delta * (1.0 + delta) *
                       (-f3 * (2.0 + delta) + 3.0 * f2 * (3.0 + delta))) /
                  6.0;
            }
          }
        });
  }
}

} // namespace

TaskStatus ApplyPackedSyncOutflowBoundariesMeshTask(std::shared_ptr<MeshData<Real>>& data,
                                                    const bool coarse) {
  PARTHENON_REQUIRE(!coarse && !data->GetMeshPointer()->multilevel,
                    "packed SYNC outflow is a uniform fine-grid operation");
  ApplyPackedOutflowFace<X1DIR, true>(data.get());
  ApplyPackedOutflowFace<X1DIR, false>(data.get());
  ApplyPackedOutflowFace<X2DIR, true>(data.get());
  ApplyPackedOutflowFace<X2DIR, false>(data.get());
  ApplyPackedOutflowFace<X3DIR, true>(data.get());
  ApplyPackedOutflowFace<X3DIR, false>(data.get());
  return TaskStatus::complete;
}

void CheckRefinementMesh(MeshData<Real>* data, parthenon::ParArray1D<AmrTag>& amr_tags) {
  const auto package = data->GetMeshPointer()->packages.Get("numerical_relativity");
  const int method = package->Param<int>("amr_method");
  const Real refine_tolerance = package->Param<Real>("refine_tolerance");
  const Real derefine_tolerance = package->Param<Real>("derefine_tolerance");
  const Real chi_threshold = package->Param<Real>("amr_chi_min");
  const Real dchi_threshold = package->Param<Real>("amr_dchi_max");
  const int tracker_count = package->Param<int>("tracker_count");
  const auto& tracker_position = package->Param<std::vector<Real>>("tracker_position");
  const auto& tracker_radius = package->Param<std::vector<Real>>("tracker_refinement_radii");
  const auto& tracker_level = package->Param<std::vector<int>>("tracker_refinement_levels");
  const auto& fixed_radius = package->Param<std::vector<Real>>("amr_fixed_radii");
  const auto& fixed_level = package->Param<std::vector<int>>("amr_fixed_levels");
  constexpr int maximum_trackers = 4;
  PARTHENON_REQUIRE(tracker_count >= 1 && tracker_count <= maximum_trackers &&
                        tracker_position.size() == static_cast<std::size_t>(3 * tracker_count) &&
                        tracker_radius.size() == static_cast<std::size_t>(tracker_count) &&
                        tracker_level.size() == static_cast<std::size_t>(tracker_count) &&
                        fixed_radius.size() == fixed_level.size() && fixed_radius.size() <= 16,
                    "invalid NR compact-object tracker state");
  Kokkos::Array<Real, 3 * maximum_trackers> tracker_centers{};
  Kokkos::Array<Real, maximum_trackers> tracker_radii{};
  Kokkos::Array<int, maximum_trackers> tracker_levels{};
  for (int component = 0; component < 3 * tracker_count; ++component)
    tracker_centers[component] = tracker_position[component];
  for (int tracker = 0; tracker < tracker_count; ++tracker) {
    tracker_radii[tracker] = tracker_radius[tracker];
    tracker_levels[tracker] = tracker_level[tracker];
  }
  Kokkos::Array<Real, 16> fixed_radii{};
  Kokkos::Array<int, 16> fixed_levels{};
  for (std::size_t region = 0; region < fixed_radius.size(); ++region) {
    fixed_radii[region] = fixed_radius[region];
    fixed_levels[region] = fixed_level[region];
  }
  const int fixed_count = fixed_radius.size();
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int nx1 = ib.e - ib.s + 1;
  const int nx2 = jb.e - jb.s + 1;
  const int nx3 = kb.e - kb.s + 1;
  const int cells_per_block = nx1 * nx2 * nx3;
  const int root_level = data->GetMeshPointer()->GetRootLevel();
  parthenon::ParArray1D<int> levels("PANGU NR AMR block levels", z4c.GetDim(5));
  parthenon::ParArray2D<Real> bounds("PANGU NR AMR block bounds", z4c.GetDim(5), 6);
  auto host_levels = Kokkos::create_mirror_view(levels);
  auto host_bounds = Kokkos::create_mirror_view(bounds);
  for (int block = 0; block < z4c.GetDim(5); ++block) {
    const auto pointer = data->GetBlockData(block)->GetBlockPointer();
    host_levels(block) = pointer->loc.level() - root_level;
    host_bounds(block, 0) = pointer->block_size.xmin(X1DIR);
    host_bounds(block, 1) = pointer->block_size.xmax(X1DIR);
    host_bounds(block, 2) = pointer->block_size.xmin(X2DIR);
    host_bounds(block, 3) = pointer->block_size.xmax(X2DIR);
    host_bounds(block, 4) = pointer->block_size.xmin(X3DIR);
    host_bounds(block, 5) = pointer->block_size.xmax(X3DIR);
  }
  Kokkos::deep_copy(levels, host_levels);
  Kokkos::deep_copy(bounds, host_bounds);
  auto scatter_tags = amr_tags.ToScatterView<Kokkos::Experimental::ScatterMax>();

  // One team owns one MeshBlock so the reductions below reproduce AthenaK's
  // block-level min/max decisions.  ScatterMax composes this package with any
  // independent Parthenon refinement criteria.
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "PANGU Z4c AMR", parthenon::DevExecSpace(), 0, 0, 0,
      z4c.GetDim(5) - 1, KOKKOS_LAMBDA(parthenon::team_mbr_t team_member, const int b) {
        int flag_value = amr::kSame;
        if (method == 1) {
          Kokkos::Array<Real, 6> block_bounds{};
          for (int component = 0; component < 6; ++component)
            block_bounds[component] = bounds(b, component);
          flag_value = amr::TrackerTag<Real>(block_bounds, levels(b), tracker_count,
                                             tracker_centers, tracker_radii, tracker_levels);
        } else if (method == 2) {
          Real minimum = std::numeric_limits<Real>::max();
          parthenon::par_reduce_inner(
              parthenon::inner_loop_pattern_ttr_tag, team_member, 0, cells_per_block - 1,
              [&](const int index, Real& local) {
                const int k = kb.s + index / (nx2 * nx1);
                const int remainder = index % (nx2 * nx1);
                const int j = jb.s + remainder / nx1;
                const int i = ib.s + remainder % nx1;
                local = fmin(local, z4c(b, Index(Z4cComponent::chi), k, j, i));
              },
              Kokkos::Min<Real>(minimum));
          flag_value = amr::ChiTag(minimum, chi_threshold);
        } else if (method == 3 || method == 4) {
          Real maximum = 0.0;
          parthenon::par_reduce_inner(
              parthenon::inner_loop_pattern_ttr_tag, team_member, 0, cells_per_block - 1,
              [&](const int index, Real& local) {
                const int k = kb.s + index / (nx2 * nx1);
                const int remainder = index % (nx2 * nx1);
                const int j = jb.s + remainder / nx1;
                const int i = ib.s + remainder % nx1;
                const Real dx = z4c(b, Index(Z4cComponent::chi), k, j, i + 1) -
                                z4c(b, Index(Z4cComponent::chi), k, j, i - 1);
                const Real dy = z4c(b, Index(Z4cComponent::chi), k, j + 1, i) -
                                z4c(b, Index(Z4cComponent::chi), k, j - 1, i);
                const Real dz = z4c(b, Index(Z4cComponent::chi), k + 1, j, i) -
                                z4c(b, Index(Z4cComponent::chi), k - 1, j, i);
                const Real indicator = amr::DchiIndicator(
                    dx, dy, dz, z4c(b, Index(Z4cComponent::chi), k, j, i), method == 4);
                local = fmax(local, indicator);
              },
              Kokkos::Max<Real>(maximum));
          const Real upper = method == 3 ? dchi_threshold : refine_tolerance;
          const Real lower = method == 3 ? 0.5 * dchi_threshold : derefine_tolerance;
          flag_value = amr::DchiTag(maximum, upper, lower);
        }
        Kokkos::Array<Real, 6> block_bounds{};
        for (int component = 0; component < 6; ++component)
          block_bounds[component] = bounds(b, component);
        flag_value = amr::ApplyFixedRadii<Real>(flag_value, block_bounds, levels(b), fixed_count,
                                                fixed_radii, fixed_levels);
        const auto flag = static_cast<AmrTag>(flag_value);
        auto tags_access = scatter_tags.access();
        tags_access(b).update(flag);
      });
  amr_tags.ContributeScatter(scatter_tags);
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput* pin) {
  PARTHENON_REQUIRE(pin->GetInteger("parthenon/mesh", "nx2") > 1 &&
                        pin->GetInteger("parthenon/mesh", "nx3") > 1,
                    "sync Z4c evolution requires a three-dimensional mesh");
  auto package = std::make_shared<StateDescriptor>("numerical_relativity");
  const std::string problem_id = pin->GetString("parthenon/job", "problem_id");
  if (problem_id == "sync_grhd_tov" || problem_id == "sync_grmhd_tov")
    package->AddParam<TOVProfile>("tov_profile", BuildPolytropicTOV(pin));
  const std::string matter_source_name =
      pin->GetOrAddString("numerical_relativity", "matter_source", "vacuum");
  StressEnergySourceOptions matter_options{};
  if (matter_source_name == "vacuum") {
    matter_options.source = StressEnergySource::vacuum;
  } else if (matter_source_name == "zero") {
    matter_options.source = StressEnergySource::zero;
  } else if (matter_source_name == "constant") {
    matter_options.source = StressEnergySource::constant;
  } else if (matter_source_name == "analytic") {
    matter_options.source = StressEnergySource::analytic;
  } else if (matter_source_name == "grhd") {
    matter_options.source = StressEnergySource::grhd;
  } else if (matter_source_name == "grmhd") {
    matter_options.source = StressEnergySource::grmhd;
  } else {
    PARTHENON_FAIL(
        "numerical_relativity/matter_source must be vacuum, zero, constant, analytic, grhd, or "
        "grmhd");
  }
  constexpr const char* matter_component_names[kStressEnergyComponents] = {
      "matter_sxx", "matter_sxy", "matter_sxz", "matter_syy", "matter_syz",
      "matter_szz", "matter_energy", "matter_momentum_x", "matter_momentum_y",
      "matter_momentum_z"};
  for (int component = 0; component < kStressEnergyComponents; ++component) {
    matter_options.constant[component] =
        pin->GetOrAddReal("numerical_relativity", matter_component_names[component], 0.0);
  }
  matter_options.amplitude =
      pin->GetOrAddReal("numerical_relativity", "matter_amplitude", 0.0);
  matter_options.wave_number[0] =
      pin->GetOrAddReal("numerical_relativity", "matter_wave_number_x1", 1.0);
  matter_options.wave_number[1] =
      pin->GetOrAddReal("numerical_relativity", "matter_wave_number_x2", 0.0);
  matter_options.wave_number[2] =
      pin->GetOrAddReal("numerical_relativity", "matter_wave_number_x3", 0.0);
  matter_options.angular_frequency =
      pin->GetOrAddReal("numerical_relativity", "matter_angular_frequency", 1.0);
  const Real cfl = pin->GetOrAddReal("numerical_relativity", "cfl", 0.4);
  const int finite_difference_order =
      pin->GetOrAddInteger("numerical_relativity", "finite_difference_order", 2);
  const int ghost_zones = pin->GetOrAddInteger("parthenon/mesh", "nghost", 2);
  const int requested_boundary_extrapolation_order =
      pin->GetOrAddInteger("numerical_relativity", "boundary_extrapolation_order", 2);
  const int boundary_extrapolation_order =
      requested_boundary_extrapolation_order == 1 ? 2 : requested_boundary_extrapolation_order;
  const Real refine_tolerance = pin->GetOrAddReal("numerical_relativity", "refine_tolerance",
                                                  std::numeric_limits<Real>::max());
  const Real derefine_tolerance =
      pin->GetOrAddReal("numerical_relativity", "derefine_tolerance", -1.0);
  const std::string amr_method_name =
      pin->GetOrAddString("numerical_relativity", "amr_method", "trivial");
  int amr_method = 0;
  if (amr_method_name == "trivial") {
    amr_method = 0;
  } else if (amr_method_name == "tracker") {
    amr_method = 1;
  } else if (amr_method_name == "chi") {
    amr_method = 2;
  } else if (amr_method_name == "dchi") {
    amr_method = 3;
  } else if (amr_method_name == "normalized_dchi") {
    amr_method = 4;
  } else {
    PARTHENON_FAIL("unknown numerical_relativity/amr_method");
  }
  const Real amr_chi_min = pin->GetOrAddReal("numerical_relativity", "amr_chi_min", 0.2);
  const Real amr_dchi_max = pin->GetOrAddReal("numerical_relativity", "amr_dchi_max", 0.01);
  const bool tracker_enabled =
      pin->GetOrAddBoolean("numerical_relativity", "tracker_enabled", false);
  const int tracker_count = pin->GetOrAddInteger("numerical_relativity", "tracker_count", 1);
  const bool weyl_enabled = pin->GetOrAddBoolean("numerical_relativity", "weyl_enabled", false);
  const bool waveform_enabled =
      pin->GetOrAddBoolean("numerical_relativity", "waveform_enabled", false);
  const bool horizon_enabled =
      pin->GetOrAddBoolean("numerical_relativity", "horizon_enabled", false);
  // Match AthenaK's FastFlow output contract: the comparatively expensive
  // point-by-point surface grid is opt-in, while the compact horizon summary
  // remains available whenever the finder is enabled.
  const bool horizon_output_grid =
      pin->GetOrAddBoolean("numerical_relativity", "horizon_output_grid", false);
  const int horizon_count =
      pin->GetOrAddInteger("numerical_relativity", "horizon_count", horizon_enabled ? 1 : 0);
  const int waveform_num_radii =
      pin->GetOrAddInteger("numerical_relativity", "waveform_num_radii", waveform_enabled ? 1 : 0);
  const int waveform_geodesic_level =
      pin->GetOrAddInteger("numerical_relativity", "waveform_geodesic_level", 10);
  const int requested_waveform_interpolation_points =
      pin->GetOrAddInteger("numerical_relativity", "waveform_interpolation_points", 0);
  const int waveform_interpolation_points = requested_waveform_interpolation_points <= 0
                                                ? 2 * ghost_zones
                                                : requested_waveform_interpolation_points;
  const Real waveform_dt = pin->GetOrAddReal("numerical_relativity", "waveform_dt", 1.0);
  std::vector<Real> waveform_radii;
  waveform_radii.reserve(waveform_num_radii);
  for (int radius = 0; radius < waveform_num_radii; ++radius) {
    waveform_radii.push_back(pin->GetOrAddReal("numerical_relativity",
                                               "waveform_radius_" + std::to_string(radius + 1),
                                               20.0 + 10.0 * radius));
  }
  const int horizon_ntheta = pin->GetOrAddInteger("numerical_relativity", "horizon_ntheta", 16);
  const int horizon_lmax = pin->GetOrAddInteger("numerical_relativity", "horizon_lmax", 4);
  const int horizon_iterations =
      pin->GetOrAddInteger("numerical_relativity", "horizon_iterations", 50);
  const Real horizon_initial_radius =
      pin->GetOrAddReal("numerical_relativity", "horizon_initial_radius", 0.5);
  const Real horizon_flow_gain =
      pin->GetOrAddReal("numerical_relativity", "horizon_flow_gain", 1.0);
  const std::string horizon_flow =
      pin->GetOrAddString("numerical_relativity", "horizon_flow", "standard");
  int horizon_flow_flag = 0;
  if (horizon_flow == "expansion") {
    horizon_flow_flag = 1;
  } else if (horizon_flow == "standard") {
    horizon_flow_flag = 2;
  } else if (horizon_flow == "shear") {
    horizon_flow_flag = 3;
  } else {
    PARTHENON_FAIL("horizon_flow must be expansion, standard, or shear");
  }
  const Real horizon_mass_tolerance =
      pin->GetOrAddReal("numerical_relativity", "horizon_mass_tolerance", 1.0e-2);
  const Real horizon_hmean_limit =
      pin->GetOrAddReal("numerical_relativity", "horizon_hmean_limit", 100.0);
  const Real horizon_expand_guess =
      pin->GetOrAddReal("numerical_relativity", "horizon_expand_guess", 1.0);
  const std::string diagnostic_directory =
      pin->GetOrAddString("numerical_relativity", "diagnostic_directory", "nr_diagnostics");
  const Real tracker_refinement_radius =
      pin->GetOrAddReal("numerical_relativity", "tracker_refinement_radius", -1.0);
  const Real tracker_center_x1 =
      pin->GetOrAddReal("numerical_relativity", "tracker_center_x1", 0.0);
  const Real tracker_center_x2 =
      pin->GetOrAddReal("numerical_relativity", "tracker_center_x2", 0.0);
  const Real tracker_center_x3 =
      pin->GetOrAddReal("numerical_relativity", "tracker_center_x3", 0.0);
  PARTHENON_REQUIRE(tracker_count >= 1 && tracker_count <= 4,
                    "numerical_relativity/tracker_count must be in [1,4]");
  PARTHENON_REQUIRE((horizon_enabled && horizon_count >= 1 && horizon_count <= 4) ||
                        (!horizon_enabled && horizon_count == 0),
                    "horizon_count must be in [1,4] when enabled and zero when disabled");
  std::vector<Real> tracker_positions;
  std::vector<Real> tracker_masses;
  std::vector<Real> tracker_refinement_radii;
  std::vector<int> tracker_refinement_levels;
  tracker_positions.reserve(3 * tracker_count);
  tracker_masses.reserve(tracker_count);
  tracker_refinement_radii.reserve(tracker_count);
  tracker_refinement_levels.reserve(tracker_count);
  for (int tracker = 0; tracker < tracker_count; ++tracker) {
    const std::string prefix = "tracker_" + std::to_string(tracker) + "_center_x";
    tracker_positions.push_back(pin->GetOrAddReal("numerical_relativity", prefix + "1",
                                                  tracker == 0 ? tracker_center_x1 : 0.0));
    tracker_positions.push_back(pin->GetOrAddReal("numerical_relativity", prefix + "2",
                                                  tracker == 0 ? tracker_center_x2 : 0.0));
    tracker_positions.push_back(pin->GetOrAddReal("numerical_relativity", prefix + "3",
                                                  tracker == 0 ? tracker_center_x3 : 0.0));
    tracker_masses.push_back(pin->GetOrAddReal(
        "numerical_relativity", "tracker_" + std::to_string(tracker) + "_mass", 1.0));
    const Real requested_tracker_radius = pin->GetOrAddReal(
        "numerical_relativity", "tracker_" + std::to_string(tracker) + "_refinement_radius",
        tracker == 0 ? tracker_refinement_radius : 0.0);
    // Preserve the established PANGU input convention where -1 disables the
    // tracker refinement ball.  AthenaK's equivalent disabled radius is zero.
    tracker_refinement_radii.push_back(requested_tracker_radius < 0.0 ? 0.0
                                                                      : requested_tracker_radius);
    tracker_refinement_levels.push_back(pin->GetOrAddInteger(
        "numerical_relativity", "tracker_" + std::to_string(tracker) + "_refinement_level", -1));
  }
  std::vector<Real> amr_fixed_radii;
  std::vector<int> amr_fixed_levels;
  for (int region = 0; region < 16; ++region) {
    const std::string radius_name = "amr_radius_" + std::to_string(region);
    if (!pin->DoesParameterExist("numerical_relativity", radius_name))
      break;
    amr_fixed_radii.push_back(pin->GetReal("numerical_relativity", radius_name));
    amr_fixed_levels.push_back(pin->GetOrAddInteger(
        "numerical_relativity", "amr_radius_" + std::to_string(region) + "_refinement_level", -1));
  }
  std::vector<Real> horizon_radii;
  std::vector<int> horizon_found(horizon_count, 0);
  std::vector<int> horizon_tracker_indices;
  std::vector<Real> horizon_centers;
  std::vector<Real> horizon_start_times;
  std::vector<Real> horizon_stop_times;
  std::vector<int> horizon_wait_for_punctures;
  std::vector<int> horizon_mass_weighted_center;
  horizon_radii.reserve(horizon_count);
  horizon_tracker_indices.reserve(horizon_count);
  horizon_centers.reserve(3 * horizon_count);
  horizon_start_times.reserve(horizon_count);
  horizon_stop_times.reserve(horizon_count);
  horizon_wait_for_punctures.reserve(horizon_count);
  horizon_mass_weighted_center.reserve(horizon_count);
  for (int horizon = 0; horizon < horizon_count; ++horizon) {
    const std::string suffix = std::to_string(horizon);
    horizon_radii.push_back(pin->GetOrAddReal(
        "numerical_relativity", "horizon_initial_radius_" + suffix, horizon_initial_radius));
    horizon_tracker_indices.push_back(pin->GetOrAddInteger(
        "numerical_relativity", "horizon_tracker_" + suffix, std::min(horizon, tracker_count - 1)));
    horizon_centers.push_back(
        pin->GetOrAddReal("numerical_relativity", "horizon_center_x1_" + suffix, 0.0));
    horizon_centers.push_back(
        pin->GetOrAddReal("numerical_relativity", "horizon_center_x2_" + suffix, 0.0));
    horizon_centers.push_back(
        pin->GetOrAddReal("numerical_relativity", "horizon_center_x3_" + suffix, 0.0));
    horizon_start_times.push_back(pin->GetOrAddReal(
        "numerical_relativity", "horizon_start_time_" + suffix, std::numeric_limits<Real>::max()));
    horizon_stop_times.push_back(
        pin->GetOrAddReal("numerical_relativity", "horizon_stop_time_" + suffix, -1.0));
    horizon_wait_for_punctures.push_back(pin->GetOrAddBoolean(
        "numerical_relativity", "horizon_wait_for_punctures_" + suffix, false));
    horizon_mass_weighted_center.push_back(pin->GetOrAddBoolean(
        "numerical_relativity", "horizon_mass_weighted_center_" + suffix, false));
  }
  for (const Real mass : tracker_masses)
    PARTHENON_REQUIRE(mass > 0.0, "each compact-object tracker mass must be positive");
  PARTHENON_REQUIRE(cfl > 0.0 && cfl <= 1.0, "numerical_relativity/cfl must be in (0,1]");
  PARTHENON_REQUIRE(finite_difference_order == 2 || finite_difference_order == 4 ||
                        finite_difference_order == 6,
                    "numerical_relativity/finite_difference_order must be 2, 4, or 6");
  PARTHENON_REQUIRE(boundary_extrapolation_order >= 0 && boundary_extrapolation_order <= 4 &&
                        boundary_extrapolation_order != 1,
                    "boundary_extrapolation_order must be 0, 1, 2, 3, or 4");
  PARTHENON_REQUIRE(refine_tolerance >= 0.0,
                    "numerical_relativity/refine_tolerance must be nonnegative");
  PARTHENON_REQUIRE(derefine_tolerance >= -1.0,
                    "numerical_relativity/derefine_tolerance must be at least -1");
  PARTHENON_REQUIRE(tracker_refinement_radius >= -1.0,
                    "numerical_relativity/tracker_refinement_radius must be at least -1");
  PARTHENON_REQUIRE(amr_chi_min >= 0.0 && amr_dchi_max >= 0.0,
                    "Z4c AMR thresholds must be nonnegative");
  for (const Real radius : tracker_refinement_radii)
    PARTHENON_REQUIRE(radius >= 0.0, "tracker refinement radii must be nonnegative");
  for (const Real radius : amr_fixed_radii)
    PARTHENON_REQUIRE(radius >= 0.0, "fixed AMR radii must be nonnegative");
  PARTHENON_REQUIRE(!waveform_enabled || weyl_enabled,
                    "waveform_enabled requires numerical_relativity/weyl_enabled=true");
  PARTHENON_REQUIRE(waveform_num_radii >= 0 && waveform_geodesic_level > 0 && waveform_dt > 0.0,
                    "invalid numerical-relativity waveform grid or cadence");
  PARTHENON_REQUIRE(waveform_interpolation_points >= 2 &&
                        waveform_interpolation_points <= 2 * ghost_zones + 1 &&
                        waveform_interpolation_points <= 9,
                    "waveform_interpolation_points must be in [2,min(2*nghost+1,9)]");
  for (const Real radius : waveform_radii)
    PARTHENON_REQUIRE(radius > 0.0, "waveform extraction radii must be positive");
  PARTHENON_REQUIRE(horizon_ntheta >= 4 && horizon_lmax >= 1 && horizon_lmax <= 16 &&
                        horizon_ntheta >= horizon_lmax + 1 && horizon_iterations > 0 &&
                        horizon_initial_radius > 0.0 && horizon_flow_gain > 0.0 &&
                        horizon_mass_tolerance > 0.0 && horizon_hmean_limit > 0.0 &&
                        horizon_expand_guess > 0.0,
                    "invalid numerical-relativity apparent-horizon controls");
  for (int horizon = 0; horizon < horizon_count; ++horizon) {
    PARTHENON_REQUIRE(horizon_radii[horizon] > 0.0,
                      "each apparent-horizon initial radius must be positive");
    PARTHENON_REQUIRE(horizon_tracker_indices[horizon] >= -1 &&
                          horizon_tracker_indices[horizon] < tracker_count,
                      "each apparent horizon tracker must be -1 or reference a configured tracker");
  }
  PARTHENON_REQUIRE(!diagnostic_directory.empty(),
                    "numerical_relativity/diagnostic_directory must not be empty");
  const int required_ghost_zones = fd::RequiredGhostZones(finite_difference_order);
  PARTHENON_REQUIRE(ghost_zones >= required_ghost_zones,
                    "Z4c finite-difference order requires at least order/2+1 mesh ghost zones");
  PARTHENON_REQUIRE(!horizon_enabled || ghost_zones == required_ghost_zones,
                    "fast-flow requires nghost=2,3,4 for finite-difference "
                    "order=2,4,6 respectively");
  PARTHENON_REQUIRE(!waveform_enabled || ghost_zones >= waveform_interpolation_points / 2,
                    "insufficient ghost zones for waveform interpolation");
  PARTHENON_REQUIRE(boundary_extrapolation_order == 0 ||
                        boundary_extrapolation_order <= std::min(ghost_zones, 4),
                    "extrapolation order exceeds available ghost zones");

  package->AddParam<driver::StageContribution>(std::string(driver::stage_contribution_key),
                                               SyncStageContribution());
  package->AddParam<Real>("cfl", cfl);
  package->AddParam<int>("finite_difference_order", finite_difference_order);
  package->AddParam<int>("boundary_extrapolation_order", boundary_extrapolation_order);
  package->AddParam<Real>("refine_tolerance", refine_tolerance);
  package->AddParam<Real>("derefine_tolerance", derefine_tolerance);
  package->AddParam<int>("amr_method", amr_method);
  package->AddParam<std::string>("amr_method_name", amr_method_name);
  package->AddParam<Real>("amr_chi_min", amr_chi_min);
  package->AddParam<Real>("amr_dchi_max", amr_dchi_max);
  package->AddParam<bool>("tracker_enabled", tracker_enabled);
  package->AddParam<int>("tracker_count", tracker_count);
  package->AddParam<bool>("weyl_enabled", weyl_enabled);
  package->AddParam<bool>("waveform_enabled", waveform_enabled);
  package->AddParam<bool>("horizon_enabled", horizon_enabled);
  package->AddParam<bool>("horizon_output_grid", horizon_output_grid);
  package->AddParam<int>("horizon_count", horizon_count);
  package->AddParam<int>("waveform_geodesic_level", waveform_geodesic_level);
  package->AddParam<int>("waveform_interpolation_points", waveform_interpolation_points);
  package->AddParam<Real>("waveform_dt", waveform_dt);
  package->AddParam<std::vector<Real>>("waveform_radii", waveform_radii);
  package->AddParam<Real>("waveform_last_output_time", 0.0, parthenon::Params::Mutability::Restart);
  package->AddParam<int>("horizon_ntheta", horizon_ntheta);
  package->AddParam<int>("horizon_lmax", horizon_lmax);
  package->AddParam<int>("horizon_iterations", horizon_iterations);
  package->AddParam<Real>("horizon_flow_gain", horizon_flow_gain);
  package->AddParam<std::string>("horizon_flow", horizon_flow);
  package->AddParam<int>("horizon_flow_flag", horizon_flow_flag);
  package->AddParam<Real>("horizon_mass_tolerance", horizon_mass_tolerance);
  package->AddParam<Real>("horizon_hmean_limit", horizon_hmean_limit);
  package->AddParam<Real>("horizon_expand_guess", horizon_expand_guess);
  package->AddParam<std::vector<Real>>("horizon_radii", horizon_radii,
                                       parthenon::Params::Mutability::Restart);
  package->AddParam<std::vector<int>>("horizon_found", horizon_found,
                                      parthenon::Params::Mutability::Restart);
  package->AddParam<std::vector<int>>("horizon_tracker_indices", horizon_tracker_indices,
                                      parthenon::Params::Mutability::Restart);
  package->AddParam<std::vector<Real>>("horizon_centers", horizon_centers,
                                       parthenon::Params::Mutability::Restart);
  package->AddParam<std::vector<Real>>("horizon_start_times", horizon_start_times);
  package->AddParam<std::vector<Real>>("horizon_stop_times", horizon_stop_times);
  package->AddParam<std::vector<int>>("horizon_wait_for_punctures", horizon_wait_for_punctures);
  package->AddParam<std::vector<int>>("horizon_mass_weighted_center", horizon_mass_weighted_center);
  package->AddParam<Real>(
      "horizon_merger_distance",
      pin->GetOrAddReal("numerical_relativity", "horizon_merger_distance", 0.1));
  package->AddParam<std::string>("diagnostic_directory", diagnostic_directory);
  package->AddParam<Real>("tracker_refinement_radius", tracker_refinement_radius);
  package->AddParam<Real>("tracker_center_x1", tracker_center_x1);
  package->AddParam<Real>("tracker_center_x2", tracker_center_x2);
  package->AddParam<Real>("tracker_center_x3", tracker_center_x3);
  package->AddParam<std::vector<Real>>("tracker_position", tracker_positions,
                                       parthenon::Params::Mutability::Restart);
  package->AddParam<std::vector<Real>>("tracker_mass", tracker_masses);
  package->AddParam<std::vector<Real>>("tracker_refinement_radii", tracker_refinement_radii);
  package->AddParam<std::vector<int>>("tracker_refinement_levels", tracker_refinement_levels);
  package->AddParam<std::vector<Real>>("amr_fixed_radii", amr_fixed_radii);
  package->AddParam<std::vector<int>>("amr_fixed_levels", amr_fixed_levels);
  package->AddParam<std::vector<Real>>("tracker_velocity",
                                       std::vector<Real>(3 * tracker_count, 0.0),
                                       parthenon::Params::Mutability::Restart);
  package->AddParam<std::string>("formulation", "z4c");
  package->AddParam<bool>("vacuum", matter_options.source == StressEnergySource::vacuum);
  package->AddParam<std::string>("matter_source_name", matter_source_name);
  package->AddParam<StressEnergySourceOptions>("matter_source_options", matter_options);

  Z4cOptions options{};
  options.chi_psi_power =
      pin->GetOrAddReal("numerical_relativity", "chi_psi_power", options.chi_psi_power);
  options.chi_div_floor =
      pin->GetOrAddReal("numerical_relativity", "chi_div_floor", options.chi_div_floor);
  options.chi_min_floor =
      pin->GetOrAddReal("numerical_relativity", "chi_min_floor", options.chi_min_floor);
  options.floor_chi = pin->GetOrAddBoolean("numerical_relativity", "floor_chi", options.floor_chi);
  const Real requested_dissipation = pin->GetOrAddReal("numerical_relativity", "dissipation", 0.0);
  const int stencil_ghost_zones = fd::RequiredGhostZones(finite_difference_order);
  options.dissipation = requested_dissipation * pow(2.0, -2.0 * stencil_ghost_zones) *
                        (stencil_ghost_zones % 2 == 0 ? -1.0 : 1.0);
  options.damp_kappa1 =
      pin->GetOrAddReal("numerical_relativity", "damp_kappa1", options.damp_kappa1);
  options.damp_kappa2 =
      pin->GetOrAddReal("numerical_relativity", "damp_kappa2", options.damp_kappa2);
  options.lapse_oplog =
      pin->GetOrAddReal("numerical_relativity", "lapse_oplog", options.lapse_oplog);
  options.lapse_harmonic_factor = pin->GetOrAddReal("numerical_relativity", "lapse_harmonic_factor",
                                                    options.lapse_harmonic_factor);
  options.lapse_harmonic =
      pin->GetOrAddReal("numerical_relativity", "lapse_harmonic", options.lapse_harmonic);
  options.lapse_advect =
      pin->GetOrAddReal("numerical_relativity", "lapse_advect", options.lapse_advect);
  options.slow_start_lapse =
      pin->GetOrAddBoolean("numerical_relativity", "slow_start_lapse", options.slow_start_lapse);
  options.slow_start_amplitude = pin->GetOrAddReal("numerical_relativity", "slow_start_amplitude",
                                                   options.slow_start_amplitude);
  options.slow_start_time =
      pin->GetOrAddReal("numerical_relativity", "slow_start_time", options.slow_start_time);
  options.slow_start_index =
      pin->GetOrAddReal("numerical_relativity", "slow_start_index", options.slow_start_index);
  options.shift_gamma =
      pin->GetOrAddReal("numerical_relativity", "shift_gamma", options.shift_gamma);
  options.shift_alpha2_gamma =
      pin->GetOrAddReal("numerical_relativity", "shift_alpha2_gamma", options.shift_alpha2_gamma);
  options.shift_harmonic =
      pin->GetOrAddReal("numerical_relativity", "shift_harmonic", options.shift_harmonic);
  options.shift_advect =
      pin->GetOrAddReal("numerical_relativity", "shift_advect", options.shift_advect);
  options.shift_eta = pin->GetOrAddReal("numerical_relativity", "shift_eta", options.shift_eta);
  options.use_z4c = pin->GetOrAddBoolean("numerical_relativity", "use_z4c", options.use_z4c);
  PARTHENON_REQUIRE(options.chi_psi_power != 0.0,
                    "numerical_relativity/chi_psi_power must be nonzero");
  PARTHENON_REQUIRE(options.chi_min_floor > 0.0,
                    "numerical_relativity/chi_min_floor must be positive");
  PARTHENON_REQUIRE(!options.slow_start_lapse || options.slow_start_time > 0.0,
                    "slow_start_time must be positive when slow_start_lapse is enabled");
  package->AddParam<Z4cOptions>("z4c_options", options);

  Metadata evolved({Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::Restart},
                   std::vector<int>{kZ4cComponents});
  evolved.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                                parthenon::refinement_ops::RestrictAverage>();
  // ADM, matter-source, constraint, and Weyl fields are reconstructed only
  // after all consumers of the preceding RK stage have completed.  AthenaK
  // therefore stores one copy of each; retaining a copy in every Parthenon
  // stage container needlessly multiplies the dominant SYNC memory cost.
  const Metadata adm({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                     std::vector<int>{kADMComponents});
  const Metadata adm_derivatives({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                                 std::vector<int>{kADMMetricDerivativeComponents});
  const Metadata constraints({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                             std::vector<int>{kConstraintComponents});
  const Metadata stress_energy(
      {Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
      std::vector<int>{kStressEnergyComponents},
      std::vector<std::string>{"Sxx", "Sxy", "Sxz", "Syy", "Syz", "Szz", "E", "Sx",
                               "Sy", "Sz"});
  Metadata weyl({Metadata::Cell, Metadata::Derived, Metadata::FillGhost, Metadata::OneCopy},
                std::vector<int>{kWeylComponents});
  weyl.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                             parthenon::refinement_ops::RestrictAverage>();
  Metadata constraint_mask(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::Restart},
      std::vector<int>{1}, std::vector<std::string>{"active"});
  constraint_mask.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                                        parthenon::refinement_ops::RestrictAverage>();
  package->AddField("nr.z4c", evolved);
  package->AddField("nr.adm", adm);
  if (horizon_enabled)
    package->AddField("nr.adm_derivatives", adm_derivatives);
  package->AddField("nr.constraints", constraints);
  if (HasStressEnergyField(matter_options.source))
    package->AddField("nr.tmunu", stress_energy);
  package->AddField("nr.weyl", weyl);
  package->AddField("nr.constraint_mask", constraint_mask);

  parthenon::HstVec_list history;
  history.emplace_back(parthenon::HistoryOutputVec(parthenon::UserHistoryOperation::max,
                                                   ConstraintMaxHistoryMesh, "nr_constraint_max"));
  history.emplace_back(parthenon::HistoryOutputVec(
      parthenon::UserHistoryOperation::max, ConstraintNormHistoryMesh, "nr_constraint_norm"));
  history.emplace_back(parthenon::HistoryOutputVec(
      parthenon::UserHistoryOperation::sum, ConstraintMaskHistoryMesh, "nr_constraint_mask_cells"));
  if (tracker_enabled) {
    history.emplace_back(parthenon::HistoryOutputVec(
        parthenon::UserHistoryOperation::sum, PunctureTrackerHistoryMesh, "nr_puncture_tracker"));
  }
  package->AddParam<>(parthenon::hist_vec_param_key, history);

  if constexpr (geometry::ConfiguredGeometryIsSynchronized()) {
    // Parthenon invokes every package's PreFillDerived callbacks before any
    // package FillDerived callback.  Reconstruct the ADM view in that ordered
    // pre-pass so hydro/MHD primitive recovery never observes an uninitialized
    // synchronized Geometry after loading only the independent restart fields.
    // The ordinary NR FillDerived pass below then adds constraints/Weyl and is
    // deliberately self-contained, so callers outside the generic Parthenon
    // initialization path retain the existing contract.
    package->PreFillDerivedBlock = [](MeshBlockData<Real>* data) {
      Z4cToADMFieldsBlockTask(data);
      // StateDescriptor callbacks do not carry a TaskID dependency between the
      // PreFillDerived and FillDerived phases.  Each packed kernel may obtain a
      // different CUDA execution-space instance, so make the reconstructed ADM
      // view visible before the MHD package starts primitive recovery.  This is
      // essential after dynamic AMR creates many fine blocks at once.
      Kokkos::fence("PANGU synchronized ADM pre-fill");
    };
    package->PreFillDerivedMesh = [](MeshData<Real>* data) {
      Z4cToADMFieldsMeshTask(data);
      Kokkos::fence("PANGU synchronized ADM pre-fill");
    };
    if (matter_options.source == StressEnergySource::grhd ||
        matter_options.source == StressEnergySource::grmhd) {
      // Independent restart data contain Z4c and the fluid conserved state,
      // but not Tmunu.  All package FillDerived callbacks run between the
      // pre/post passes, so rebuild stress energy only after hydro/MHD has
      // recovered primitives, then recompute the matter constraints.  This
      // ordering is the Parthenon equivalent of AthenaK's restart sequence.
      package->PostFillDerivedBlock = [](MeshBlockData<Real>* data) {
        BuildStressEnergyBlockTask(data, 0.0);
        Z4cToADMBlockTask(data);
      };
      package->PostFillDerivedMesh = [](MeshData<Real>* data) {
        BuildStressEnergyMeshTask(data, 0.0);
        Z4cToADMMeshTask(data);
      };
    }
  }
  package->FillDerivedBlock = Z4cToADMBlockTask;
  package->FillDerivedMesh = Z4cToADMMeshTask;
  package->EstimateTimestepBlock = EstimateTimestepBlock;
  package->EstimateTimestepMesh = EstimateTimestepMesh;
  package->CheckRefinementMesh = CheckRefinementMesh;
  return package;
}

void AdvancePunctureTracker(Mesh* mesh, const Real dt) {
  const auto package = mesh->packages.Get("numerical_relativity");
  if (!package->Param<bool>("tracker_enabled"))
    return;
  const int tracker_count = package->Param<int>("tracker_count");
  auto* position = package->MutableParam<std::vector<Real>>("tracker_position");
  auto* velocity = package->MutableParam<std::vector<Real>>("tracker_velocity");
  PARTHENON_REQUIRE(position->size() == static_cast<std::size_t>(3 * tracker_count) &&
                        velocity->size() == position->size(),
                    "NR puncture tracker state has an invalid size");

  // Match AthenaK's ODE tracker: interpolate beta^i with a tensor-product
  // Lagrange stencil containing twice the configured number of ghost cells,
  // then advance dx^i/dt=-beta^i once after the complete RK step. Each compact
  // object is located and reduced independently.
  for (int tracker = 0; tracker < tracker_count; ++tracker) {
    const int offset = 3 * tracker;
    std::shared_ptr<MeshBlock> owner;
    int owner_level = std::numeric_limits<int>::min();
    for (const auto& block : mesh->block_list) {
      const auto& size = block->block_size;
      const bool contains =
          (*position)[offset] >= size.xmin(X1DIR) && (*position)[offset] < size.xmax(X1DIR) &&
          (*position)[offset + 1] >= size.xmin(X2DIR) &&
          (*position)[offset + 1] < size.xmax(X2DIR) &&
          (*position)[offset + 2] >= size.xmin(X3DIR) && (*position)[offset + 2] < size.xmax(X3DIR);
      if (contains && block->loc.level() > owner_level) {
        owner = block;
        owner_level = block->loc.level();
      }
    }

    Real local_state[4]{};
    if (owner) {
      const int nghost = parthenon::Globals::nghost;
      const int points = 2 * nghost;
      constexpr int kMaximumPoints = 16;
      PARTHENON_REQUIRE(points <= kMaximumPoints,
                        "NR puncture tracker supports at most eight ghost zones");
      const auto data = owner->meshblock_data.Get();
      const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
      const auto ib = owner->cellbounds.GetBoundsI(IndexDomain::interior);
      const auto jb = owner->cellbounds.GetBoundsJ(IndexDomain::interior);
      const auto kb = owner->cellbounds.GetBoundsK(IndexDomain::interior);
      const auto& size = owner->block_size;
      const Real dx[3] = {(size.xmax(X1DIR) - size.xmin(X1DIR)) / size.nx(X1DIR),
                          (size.xmax(X2DIR) - size.xmin(X2DIR)) / size.nx(X2DIR),
                          (size.xmax(X3DIR) - size.xmin(X3DIR)) / size.nx(X3DIR)};
      const Real xmin[3] = {size.xmin(X1DIR), size.xmin(X2DIR), size.xmin(X3DIR)};
      const int interior_start[3] = {ib.s, jb.s, kb.s};
      int nearest_left[3]{};
      Kokkos::Array<Real, 3 * kMaximumPoints> weights{};
      for (int direction = 0; direction < 3; ++direction) {
        nearest_left[direction] = static_cast<int>(
            floor(((*position)[offset + direction] - (xmin[direction] + 0.5 * dx[direction])) /
                  dx[direction]));
        for (int point = 0; point < points; ++point) {
          Real weight = 1.0;
          const int point_index = nearest_left[direction] - nghost + point + 1;
          const Real point_coordinate = xmin[direction] + (point_index + 0.5) * dx[direction];
          for (int other = 0; other < points; ++other) {
            if (other == point)
              continue;
            const int other_index = nearest_left[direction] - nghost + other + 1;
            const Real other_coordinate = xmin[direction] + (other_index + 0.5) * dx[direction];
            weight *= ((*position)[offset + direction] - other_coordinate) /
                      (point_coordinate - other_coordinate);
          }
          weights[direction * kMaximumPoints + point] = weight;
        }
      }
      const int first_i = interior_start[0] + nearest_left[0] - nghost + 1;
      const int first_j = interior_start[1] + nearest_left[1] - nghost + 1;
      const int first_k = interior_start[2] + nearest_left[2] - nghost + 1;
      Kokkos::View<Real[3]> interpolated("NR puncture tracker velocity");
      Kokkos::parallel_for(
          "PANGU NR puncture tracker interpolation",
          Kokkos::RangePolicy<parthenon::DevExecSpace>(0, 3), KOKKOS_LAMBDA(const int axis) {
            Real value = 0.0;
            for (int pk = 0; pk < points; ++pk)
              for (int pj = 0; pj < points; ++pj)
                for (int pi = 0; pi < points; ++pi)
                  value += weights[pi] * weights[kMaximumPoints + pj] *
                           weights[2 * kMaximumPoints + pk] *
                           z4c(Index(Z4cComponent::betax) + axis, first_k + pk, first_j + pj,
                               first_i + pi);
            interpolated(axis) = -value;
          });
      const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), interpolated);
      for (int axis = 0; axis < 3; ++axis)
        local_state[axis] = host(axis);
      local_state[3] = 1.0;
    }

    Real global_state[4]{};
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(
        MPI_Allreduce(local_state, global_state, 4, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#else
    for (int component = 0; component < 4; ++component)
      global_state[component] = local_state[component];
#endif
    PARTHENON_REQUIRE(global_state[3] > 0.5,
                      "NR puncture tracker could not find a compact object on the mesh");
    for (int axis = 0; axis < 3; ++axis) {
      (*velocity)[offset + axis] = global_state[axis] / global_state[3];
      (*position)[offset + axis] += dt * (*velocity)[offset + axis];
    }
  }
}

TaskStatus BuildStressEnergyBlockTask(MeshBlockData<Real>* data, const Real time) {
  const auto block = data->GetBlockPointer();
  const auto options = block->packages.Get("numerical_relativity")
                           ->Param<StressEnergySourceOptions>("matter_source_options");
  if (!HasStressEnergyField(options.source))
    return TaskStatus::complete;
  if (options.source == StressEnergySource::grhd)
    return BuildSyncGRHDStressEnergyBlockTask(data);
  if (options.source == StressEnergySource::grmhd)
    return BuildSyncGRMHDStressEnergyBlockTask(data);
  const auto tmunu = data->PackVariables(std::vector<std::string>{"nr.tmunu"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  block->par_for(
      "PANGU analytic NR stress-energy", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto matter = EvaluateStressEnergySource(
            options, coordinates.Xc<X1DIR>(k, j, i), coordinates.Xc<X2DIR>(k, j, i),
            coordinates.Xc<X3DIR>(k, j, i), time);
        for (int component = 0; component < 6; ++component)
          tmunu(component, k, j, i) = matter.stress[component];
        tmunu(Index(StressEnergyComponent::energy), k, j, i) = matter.energy;
        for (int axis = 0; axis < 3; ++axis)
          tmunu(Index(StressEnergyComponent::momentum_x) + axis, k, j, i) = matter.momentum[axis];
      });
  return TaskStatus::complete;
}

TaskStatus BuildStressEnergyMeshTask(MeshData<Real>* data, const Real time) {
  const auto first = data->GetBlockData(0)->GetBlockPointer();
  const auto options = first->packages.Get("numerical_relativity")
                           ->Param<StressEnergySourceOptions>("matter_source_options");
  if (!HasStressEnergyField(options.source))
    return TaskStatus::complete;
  if (options.source == StressEnergySource::grhd)
    return BuildSyncGRHDStressEnergyMeshTask(data);
  if (options.source == StressEnergySource::grmhd)
    return BuildSyncGRMHDStressEnergyMeshTask(data);
  const auto tmunu = data->PackVariables(std::vector<std::string>{"nr.tmunu"});
  const auto ib = first->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = first->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = first->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed analytic NR stress-energy", parthenon::DevExecSpace(), 0,
      tmunu.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const auto& coordinates = tmunu.GetCoords(block);
        const auto matter = EvaluateStressEnergySource(
            options, coordinates.Xc<X1DIR>(k, j, i), coordinates.Xc<X2DIR>(k, j, i),
            coordinates.Xc<X3DIR>(k, j, i), time);
        for (int component = 0; component < 6; ++component)
          tmunu(block, component, k, j, i) = matter.stress[component];
        tmunu(block, Index(StressEnergyComponent::energy), k, j, i) = matter.energy;
        for (int axis = 0; axis < 3; ++axis)
          tmunu(block, Index(StressEnergyComponent::momentum_x) + axis, k, j, i) =
              matter.momentum[axis];
      });
  return TaskStatus::complete;
}

TaskStatus Z4cToADMFieldsBlockTask(MeshBlockData<Real>* data) {
  auto block = data->GetBlockPointer();
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto chi_psi_power =
      block->packages.Get("numerical_relativity")->Param<Z4cOptions>("z4c_options").chi_psi_power;
  block->par_for(
      "PANGU NR Z4c to ADM", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        ConvertPoint(z4c, adm, 0, k, j, i, chi_psi_power);
      });
  return TaskStatus::complete;
}

TaskStatus Z4cToADMStageBlockTask(MeshBlockData<Real>* data) {
  Z4cToADMFieldsBlockTask(data);
  ComputeConstraintsBlockTask(data);
  return TaskStatus::complete;
}

TaskStatus Z4cToADMBlockTask(MeshBlockData<Real>* data) {
  Z4cToADMStageBlockTask(data);
  if (data->GetBlockPointer()->packages.Get("numerical_relativity")->Param<bool>("weyl_enabled"))
    ComputeWeylBlockTask(data);
  return TaskStatus::complete;
}

TaskStatus Z4cToADMFieldsMeshTask(MeshData<Real>* data) {
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto first = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = first->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = first->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = first->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto chi_psi_power =
      first->packages.Get("numerical_relativity")->Param<Z4cOptions>("z4c_options").chi_psi_power;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed NR Z4c to ADM", parthenon::DevExecSpace(), 0,
      z4c.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        ConvertPoint(z4c, adm, block, k, j, i, chi_psi_power);
      });
  return TaskStatus::complete;
}

TaskStatus Z4cToADMStageMeshTask(MeshData<Real>* data) {
  Z4cToADMFieldsMeshTask(data);
  ComputeConstraintsMeshTask(data);
  return TaskStatus::complete;
}

TaskStatus Z4cToADMMeshTask(MeshData<Real>* data) {
  Z4cToADMStageMeshTask(data);
  const auto first = data->GetBlockData(0)->GetBlockPointer();
  if (first->packages.Get("numerical_relativity")->Param<bool>("weyl_enabled"))
    ComputeWeylMeshTask(data);
  return TaskStatus::complete;
}

template <int Order> TaskStatus ComputeWeylMeshImpl(MeshData<Real>* data) {
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto weyl = data->PackVariables(std::vector<std::string>{"nr.weyl"});
  const auto first = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = first->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = first->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = first->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU NR Weyl scalars", parthenon::DevExecSpace(), 0,
      adm.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const auto& coordinates = adm.GetCoords(block);
        const Real inverse_spacing[3] = {1.0 / coordinates.Dxc<X1DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X2DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X3DIR>(k, j, i)};
        const auto value = ComputeWeylScalars<Order>(
            adm, block, k, j, i, inverse_spacing, coordinates.Xc<X1DIR>(k, j, i),
            coordinates.Xc<X2DIR>(k, j, i), coordinates.Xc<X3DIR>(k, j, i));
        weyl(block, Index(WeylComponent::psi4_real), k, j, i) = value.real;
        weyl(block, Index(WeylComponent::psi4_imag), k, j, i) = value.imag;
      });
  return TaskStatus::complete;
}

TaskStatus ComputeWeylMeshTask(MeshData<Real>* data) {
  const auto package =
      data->GetBlockData(0)->GetBlockPointer()->packages.Get("numerical_relativity");
  const int order = package->Param<int>("finite_difference_order");
  if (order == 2)
    return ComputeWeylMeshImpl<2>(data);
  if (order == 4)
    return ComputeWeylMeshImpl<4>(data);
  return ComputeWeylMeshImpl<6>(data);
}

template <int Order> TaskStatus ComputeWeylBlockImpl(MeshBlockData<Real>* data) {
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto weyl = data->PackVariables(std::vector<std::string>{"nr.weyl"});
  const auto block = data->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  block->par_for(
      "PANGU NR block Weyl scalars", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real inverse_spacing[3] = {1.0 / coordinates.Dxc<X1DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X2DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X3DIR>(k, j, i)};
        const auto value = ComputeWeylScalars<Order>(
            adm, 0, k, j, i, inverse_spacing, coordinates.Xc<X1DIR>(k, j, i),
            coordinates.Xc<X2DIR>(k, j, i), coordinates.Xc<X3DIR>(k, j, i));
        weyl(Index(WeylComponent::psi4_real), k, j, i) = value.real;
        weyl(Index(WeylComponent::psi4_imag), k, j, i) = value.imag;
      });
  return TaskStatus::complete;
}

TaskStatus ComputeWeylBlockTask(MeshBlockData<Real>* data) {
  const auto package = data->GetBlockPointer()->packages.Get("numerical_relativity");
  const int order = package->Param<int>("finite_difference_order");
  if (order == 2)
    return ComputeWeylBlockImpl<2>(data);
  if (order == 4)
    return ComputeWeylBlockImpl<4>(data);
  return ComputeWeylBlockImpl<6>(data);
}

template <int Order> TaskStatus ComputeConstraintsMeshImpl(MeshData<Real>* data) {
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto constraints = data->PackVariables(std::vector<std::string>{"nr.constraints"});
  const auto first = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = first->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = first->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = first->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto options =
      first->packages.Get("numerical_relativity")->Param<Z4cOptions>("z4c_options");
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU NR constraints", parthenon::DevExecSpace(), 0, z4c.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const auto& coordinates = z4c.GetCoords(block);
        const Real inverse_spacing[3] = {1.0 / coordinates.Dxc<X1DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X2DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X3DIR>(k, j, i)};
        ConstraintValues local{};
        ComputeZ4cConstraints<Order>(z4c, adm, block, k, j, i, inverse_spacing,
                                     options.chi_psi_power, local);
        for (int component = 0; component < kConstraintComponents; ++component)
          constraints(block, component, k, j, i) = local.values[component];
      });
  return TaskStatus::complete;
}

template <int Order> TaskStatus ComputeMatterConstraintsMeshImpl(MeshData<Real>* data) {
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto tmunu = data->PackVariables(std::vector<std::string>{"nr.tmunu"});
  const auto constraints = data->PackVariables(std::vector<std::string>{"nr.constraints"});
  const auto first = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = first->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = first->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = first->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto options =
      first->packages.Get("numerical_relativity")->Param<Z4cOptions>("z4c_options");
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU NR matter constraints", parthenon::DevExecSpace(), 0,
      z4c.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const auto& coordinates = z4c.GetCoords(block);
        const Real inverse_spacing[3] = {1.0 / coordinates.Dxc<X1DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X2DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X3DIR>(k, j, i)};
        ConstraintValues local{};
        ComputeZ4cConstraints<Order>(z4c, adm, block, k, j, i, inverse_spacing,
                                     options.chi_psi_power,
                                     LoadStressEnergy(tmunu, block, k, j, i), local);
        for (int component = 0; component < kConstraintComponents; ++component)
          constraints(block, component, k, j, i) = local.values[component];
      });
  return TaskStatus::complete;
}

TaskStatus ComputeConstraintsMeshTask(MeshData<Real>* data) {
  const auto first = data->GetBlockData(0)->GetBlockPointer();
  const int order =
      first->packages.Get("numerical_relativity")->Param<int>("finite_difference_order");
  const bool vacuum = first->packages.Get("numerical_relativity")->Param<bool>("vacuum");
  if (!vacuum) {
    if (order == 2)
      return ComputeMatterConstraintsMeshImpl<2>(data);
    if (order == 4)
      return ComputeMatterConstraintsMeshImpl<4>(data);
    return ComputeMatterConstraintsMeshImpl<6>(data);
  }
  if (order == 2)
    return ComputeConstraintsMeshImpl<2>(data);
  if (order == 4)
    return ComputeConstraintsMeshImpl<4>(data);
  return ComputeConstraintsMeshImpl<6>(data);
}

template <int Order> TaskStatus ComputeConstraintsBlockImpl(MeshBlockData<Real>* data) {
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto constraints = data->PackVariables(std::vector<std::string>{"nr.constraints"});
  auto block = data->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto options =
      block->packages.Get("numerical_relativity")->Param<Z4cOptions>("z4c_options");
  const auto coordinates = block->coords;
  block->par_for(
      "PANGU NR block constraints", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real inverse_spacing[3] = {1.0 / coordinates.Dxc<X1DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X2DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X3DIR>(k, j, i)};
        ConstraintValues local{};
        ComputeZ4cConstraints<Order>(z4c, adm, 0, k, j, i, inverse_spacing, options.chi_psi_power,
                                     local);
        for (int component = 0; component < kConstraintComponents; ++component)
          constraints(component, k, j, i) = local.values[component];
      });
  return TaskStatus::complete;
}

template <int Order> TaskStatus ComputeMatterConstraintsBlockImpl(MeshBlockData<Real>* data) {
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto tmunu = data->PackVariables(std::vector<std::string>{"nr.tmunu"});
  const auto constraints = data->PackVariables(std::vector<std::string>{"nr.constraints"});
  auto block = data->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto options =
      block->packages.Get("numerical_relativity")->Param<Z4cOptions>("z4c_options");
  const auto coordinates = block->coords;
  block->par_for(
      "PANGU NR block matter constraints", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real inverse_spacing[3] = {1.0 / coordinates.Dxc<X1DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X2DIR>(k, j, i),
                                         1.0 / coordinates.Dxc<X3DIR>(k, j, i)};
        ConstraintValues local{};
        const auto matter =
            LoadStressEnergy([&](const int component) { return tmunu(component, k, j, i); });
        ComputeZ4cConstraints<Order>(z4c, adm, 0, k, j, i, inverse_spacing, options.chi_psi_power,
                                     matter, local);
        for (int component = 0; component < kConstraintComponents; ++component)
          constraints(component, k, j, i) = local.values[component];
      });
  return TaskStatus::complete;
}

TaskStatus ComputeConstraintsBlockTask(MeshBlockData<Real>* data) {
  const auto block = data->GetBlockPointer();
  const int order =
      block->packages.Get("numerical_relativity")->Param<int>("finite_difference_order");
  const bool vacuum = block->packages.Get("numerical_relativity")->Param<bool>("vacuum");
  if (!vacuum) {
    if (order == 2)
      return ComputeMatterConstraintsBlockImpl<2>(data);
    if (order == 4)
      return ComputeMatterConstraintsBlockImpl<4>(data);
    return ComputeMatterConstraintsBlockImpl<6>(data);
  }
  if (order == 2)
    return ComputeConstraintsBlockImpl<2>(data);
  if (order == 4)
    return ComputeConstraintsBlockImpl<4>(data);
  return ComputeConstraintsBlockImpl<6>(data);
}

template <int Order, bool WithMatter>
TaskStatus StageUpdateMeshImpl(MeshData<Real>* current, MeshData<Real>* base,
                               const Real gamma_current, const Real gamma_base, const Real beta_dt,
                               const Real time, MeshData<Real>* next) {
  const auto input = current->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto initial = base->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto output = next->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto first = current->GetBlockData(0)->GetBlockPointer();
  const auto ib = first->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = first->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = first->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto options =
      first->packages.Get("numerical_relativity")->Param<Z4cOptions>("z4c_options");
  if constexpr (WithMatter) {
    const auto tmunu = current->PackVariables(std::vector<std::string>{"nr.tmunu"});
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU fused matter Z4c RHS and stage update",
        parthenon::DevExecSpace(), 0, input.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
          const auto& coordinates = input.GetCoords(block);
          const Real inverse_spacing[3] = {1.0 / coordinates.Dxc<X1DIR>(k, j, i),
                                           1.0 / coordinates.Dxc<X2DIR>(k, j, i),
                                           1.0 / coordinates.Dxc<X3DIR>(k, j, i)};
          Real local_rhs[kZ4cComponents]{};
          rhs::EvaluatePointWithMatter<Order>(input, tmunu, block, k, j, i, inverse_spacing,
                                              options, time, local_rhs);
          for (int component = 0; component < kZ4cComponents; ++component) {
            output(block, component, k, j, i) = gamma_current * input(block, component, k, j, i) +
                                                gamma_base * initial(block, component, k, j, i) +
                                                beta_dt * local_rhs[component];
          }
        });
  } else {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU fused vacuum Z4c RHS and stage update",
        parthenon::DevExecSpace(), 0, input.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
          const auto& coordinates = input.GetCoords(block);
          const Real inverse_spacing[3] = {1.0 / coordinates.Dxc<X1DIR>(k, j, i),
                                           1.0 / coordinates.Dxc<X2DIR>(k, j, i),
                                           1.0 / coordinates.Dxc<X3DIR>(k, j, i)};
          Real local_rhs[kZ4cComponents]{};
          rhs::EvaluatePoint<Order>(input, block, k, j, i, inverse_spacing, options, time,
                                    local_rhs);
          for (int component = 0; component < kZ4cComponents; ++component) {
            output(block, component, k, j, i) = gamma_current * input(block, component, k, j, i) +
                                                gamma_base * initial(block, component, k, j, i) +
                                                beta_dt * local_rhs[component];
          }
        });
  }

  // Replace the volume RHS on active cells touching a global outflow face with
  // AthenaK's second-order Sommerfeld characteristic approximation. User
  // boundaries (including nr_reflecting and nr_extrapolate) deliberately do
  // not enter this path; their registered ghost callbacks remain authoritative.
  Kokkos::fence();
  const auto* mesh = first->pmy_mesh;
  for (int block = 0; block < input.GetDim(5); ++block) {
    const auto block_ptr = current->GetBlockData(block)->GetBlockPointer();
    const auto is_nr_outflow = [](const std::string& name) {
      return name == "outflow" || name == "nr_outflow";
    };
    const bool inner_x1 = block_ptr->boundary_flag[parthenon::BoundaryFace::inner_x1] ==
                              parthenon::BoundaryFlag::user &&
                          is_nr_outflow(mesh->mesh_bc_names[0]);
    const bool outer_x1 = block_ptr->boundary_flag[parthenon::BoundaryFace::outer_x1] ==
                              parthenon::BoundaryFlag::user &&
                          is_nr_outflow(mesh->mesh_bc_names[1]);
    const bool inner_x2 = block_ptr->boundary_flag[parthenon::BoundaryFace::inner_x2] ==
                              parthenon::BoundaryFlag::user &&
                          is_nr_outflow(mesh->mesh_bc_names[2]);
    const bool outer_x2 = block_ptr->boundary_flag[parthenon::BoundaryFace::outer_x2] ==
                              parthenon::BoundaryFlag::user &&
                          is_nr_outflow(mesh->mesh_bc_names[3]);
    const bool inner_x3 = block_ptr->boundary_flag[parthenon::BoundaryFace::inner_x3] ==
                              parthenon::BoundaryFlag::user &&
                          is_nr_outflow(mesh->mesh_bc_names[4]);
    const bool outer_x3 = block_ptr->boundary_flag[parthenon::BoundaryFace::outer_x3] ==
                              parthenon::BoundaryFlag::user &&
                          is_nr_outflow(mesh->mesh_bc_names[5]);
    if (!(inner_x1 || outer_x1 || inner_x2 || outer_x2 || inner_x3 || outer_x3))
      continue;
    const auto block_ib = block_ptr->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto block_jb = block_ptr->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto block_kb = block_ptr->cellbounds.GetBoundsK(IndexDomain::interior);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU Z4c Sommerfeld boundary RHS", parthenon::DevExecSpace(),
        block_kb.s, block_kb.e, block_jb.s, block_jb.e, block_ib.s, block_ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const bool on_boundary = (inner_x1 && i == block_ib.s) || (outer_x1 && i == block_ib.e) ||
                                   (inner_x2 && j == block_jb.s) || (outer_x2 && j == block_jb.e) ||
                                   (inner_x3 && k == block_kb.s) || (outer_x3 && k == block_kb.e);
          if (!on_boundary)
            return;
          const auto& coordinates = input.GetCoords(block);
          const Real inverse_spacing[3] = {1.0 / coordinates.Dxc<X1DIR>(k, j, i),
                                           1.0 / coordinates.Dxc<X2DIR>(k, j, i),
                                           1.0 / coordinates.Dxc<X3DIR>(k, j, i)};
          Real local_rhs[kZ4cComponents]{};
          rhs::EvaluatePoint<2>(input, block, k, j, i, inverse_spacing, options, time, local_rhs);
          ApplySommerfeldBoundaryRHS(input, block, k, j, i, inverse_spacing,
                                     coordinates.Xc<X1DIR>(k, j, i), coordinates.Xc<X2DIR>(k, j, i),
                                     coordinates.Xc<X3DIR>(k, j, i), local_rhs);
          for (int component = 0; component < kZ4cComponents; ++component) {
            output(block, component, k, j, i) = gamma_current * input(block, component, k, j, i) +
                                                gamma_base * initial(block, component, k, j, i) +
                                                beta_dt * local_rhs[component];
          }
        });
  }
  return TaskStatus::complete;
}

TaskStatus StageUpdateMeshTask(MeshData<Real>* current, MeshData<Real>* base,
                               const Real gamma_current, const Real gamma_base, const Real beta_dt,
                               const Real time, MeshData<Real>* next) {
  const auto first = current->GetBlockData(0)->GetBlockPointer();
  const int order =
      first->packages.Get("numerical_relativity")->Param<int>("finite_difference_order");
  const bool vacuum = first->packages.Get("numerical_relativity")->Param<bool>("vacuum");
  if (vacuum) {
    if (order == 2)
      return StageUpdateMeshImpl<2, false>(current, base, gamma_current, gamma_base, beta_dt, time,
                                           next);
    if (order == 4)
      return StageUpdateMeshImpl<4, false>(current, base, gamma_current, gamma_base, beta_dt, time,
                                           next);
    return StageUpdateMeshImpl<6, false>(current, base, gamma_current, gamma_base, beta_dt, time,
                                         next);
  }
  if (order == 2)
    return StageUpdateMeshImpl<2, true>(current, base, gamma_current, gamma_base, beta_dt, time,
                                        next);
  if (order == 4)
    return StageUpdateMeshImpl<4, true>(current, base, gamma_current, gamma_base, beta_dt, time,
                                        next);
  return StageUpdateMeshImpl<6, true>(current, base, gamma_current, gamma_base, beta_dt, time, next);
}

TaskStatus AccumulateRKStateMeshTask(MeshData<Real>* current, MeshData<Real>* accumulator,
                                     const Real delta, const bool initialize) {
  const auto input = current->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto output = accumulator->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto ib = current->GetBoundsI(IndexDomain::interior);
  const auto jb = current->GetBoundsJ(IndexDomain::interior);
  const auto kb = current->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU RK accumulator", parthenon::DevExecSpace(), 0,
      input.GetDim(5) - 1, 0, input.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int component, const int k, const int j, const int i) {
        if (initialize)
          output(block, component, k, j, i) = input(block, component, k, j, i);
        else
          output(block, component, k, j, i) += delta * input(block, component, k, j, i);
      });
  return TaskStatus::complete;
}

TaskStatus FloorChiMeshTask(MeshData<Real>* data) {
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto first = data->GetBlockData(0)->GetBlockPointer();
  const auto options =
      first->packages.Get("numerical_relativity")->Param<Z4cOptions>("z4c_options");
  if (!options.floor_chi)
    return TaskStatus::complete;
  const auto ib = first->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = first->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = first->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU NR chi floor", parthenon::DevExecSpace(), 0, z4c.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        const Real value = z4c(block, Index(Z4cComponent::chi), k, j, i);
        if (!Kokkos::isfinite(value) || value < options.chi_min_floor)
          z4c(block, Index(Z4cComponent::chi), k, j, i) = options.chi_min_floor;
      });
  return TaskStatus::complete;
}

TaskStatus EnforceAlgebraicConstraintsBlockTask(MeshBlockData<Real>* data) {
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  auto block = data->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  block->par_for(
      "PANGU NR block algebraic projection", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real metric[6]{};
        for (int component = 0; component < 6; ++component)
          metric[component] = z4c(Index(Z4cComponent::gxx) + component, k, j, i);
        Real determinant = rhs::SpatialDeterminant(metric);
        determinant = determinant > 0.0 ? determinant : 1.0;
        const Real normalization = cbrt(1.0 / determinant);
        for (int component = 0; component < 6; ++component) {
          metric[component] *= normalization;
          z4c(Index(Z4cComponent::gxx) + component, k, j, i) = metric[component];
        }
        Real inverse[6]{};
        rhs::SpatialInverse(1.0, metric, inverse);
        Real trace_a = 0.0;
        for (int first = 0; first < 3; ++first) {
          for (int second = 0; second < 3; ++second) {
            const int component = SpatialSymmetricComponent(first, second);
            trace_a += inverse[component] * z4c(Index(Z4cComponent::axx) + component, k, j, i);
          }
        }
        for (int component = 0; component < 6; ++component) {
          z4c(Index(Z4cComponent::axx) + component, k, j, i) -=
              (1.0 / 3.0) * trace_a * metric[component];
        }
      });
  return TaskStatus::complete;
}

TaskStatus EnforceAlgebraicConstraintsMeshTask(MeshData<Real>* data) {
  const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto first = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = first->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = first->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = first->cellbounds.GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU NR algebraic projection", parthenon::DevExecSpace(), 0,
      z4c.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i) {
        Real metric[6]{};
        for (int component = 0; component < 6; ++component)
          metric[component] = z4c(block, Index(Z4cComponent::gxx) + component, k, j, i);
        Real determinant = rhs::SpatialDeterminant(metric);
        determinant = determinant > 0.0 ? determinant : 1.0;
        const Real normalization = cbrt(1.0 / determinant);
        for (int component = 0; component < 6; ++component) {
          metric[component] *= normalization;
          z4c(block, Index(Z4cComponent::gxx) + component, k, j, i) = metric[component];
        }
        Real inverse[6]{};
        rhs::SpatialInverse(1.0, metric, inverse);
        Real trace_a = 0.0;
        for (int first_axis = 0; first_axis < 3; ++first_axis) {
          for (int second_axis = 0; second_axis < 3; ++second_axis) {
            const int component = SpatialSymmetricComponent(first_axis, second_axis);
            trace_a +=
                inverse[component] * z4c(block, Index(Z4cComponent::axx) + component, k, j, i);
          }
        }
        for (int component = 0; component < 6; ++component) {
          z4c(block, Index(Z4cComponent::axx) + component, k, j, i) -=
              (1.0 / 3.0) * trace_a * metric[component];
        }
      });
  return TaskStatus::complete;
}

Real EstimateTimestepBlock(MeshBlockData<Real>* data) {
  auto block = data->GetBlockPointer();
  const auto cfl = block->packages.Get("numerical_relativity")->Param<Real>("cfl");
  const auto coordinates = block->coords;
  const int ndim = block->pmy_mesh->ndim;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  Real minimum;
  ParReduce(
      "PANGU NR timestep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
        local = fmin(local, coordinates.Dxc<X1DIR>(k, j, i));
        if (ndim >= 2)
          local = fmin(local, coordinates.Dxc<X2DIR>(k, j, i));
        if (ndim >= 3)
          local = fmin(local, coordinates.Dxc<X3DIR>(k, j, i));
      },
      Kokkos::Min<Real>(minimum));
  return cfl * minimum;
}

Real EstimateTimestepMesh(MeshData<Real>* data) {
  const auto first = data->GetBlockData(0)->GetBlockPointer();
  const auto cfl = first->packages.Get("numerical_relativity")->Param<Real>("cfl");
  const auto variables = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const int ndim = data->GetMeshPointer()->ndim;
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  Real minimum;
  ParReduce(
      "PANGU packed NR timestep", 0, variables.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int block, const int k, const int j, const int i, Real& local) {
        const auto& coordinates = variables.GetCoords(block);
        local = fmin(local, coordinates.Dxc<X1DIR>(k, j, i));
        if (ndim >= 2)
          local = fmin(local, coordinates.Dxc<X2DIR>(k, j, i));
        if (ndim >= 3)
          local = fmin(local, coordinates.Dxc<X3DIR>(k, j, i));
      },
      Kokkos::Min<Real>(minimum));
  return cfl * minimum;
}

} // namespace pangu::nr
