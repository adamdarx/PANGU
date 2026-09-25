#include "mhd/mhd_package.h"

// Copyright (c) 2020 James M. Stone and the AthenaK team.
// The GS07 corner-EMF formulas and MHD wave-speed algebra are derived from AthenaK
// under BSD-3-Clause; see LICENSE and NOTICE.

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "driver/contributor.h"
#include "driver/stage_extension.h"
#include "eos/newtonian_mhd_eos.h"
#include "estimator/estimator.h"
#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "mhd/mhd_types.h"
#include "mhd/task_contribution.h"
#include "z4c/coupling/sync_grmhd.h"
#include "pangu.h"
#include "prolong_restrict/prolong_restrict.hpp"
#include "reconstruct/hydro_reconstruction.h"
#include "relativity/physics_assembly.h"
#include "relativity/relativistic_mhd.h"
#include "riemann/registry.h"
#include "srcterms/source_terms.h"
#include "utils/error_checking.hpp"

namespace pangu::mhd {
using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

// First-order flux correction replaces flagged faces with the registered fallback solver.
using FirstOrderSolver =
    riemann::Implementation<riemann::Physics::newtonian_mhd, riemann::first_order_fallback>;

namespace {

constexpr int E2X1 = 0;
constexpr int E3X1 = 1;
constexpr int E1X2 = 2;
constexpr int E3X2 = 3;
constexpr int E1X3 = 4;
constexpr int E2X3 = 5;

EosMode ParseEos(const std::string& name) {
  if (name == "ideal" || name == "adiabatic")
    return EosMode::ideal;
  if (name == "isothermal")
    return EosMode::isothermal;
  PARTHENON_FAIL("mhd/eos must be ideal, adiabatic, or isothermal");
}

relativity::HydroMode ParsePhysics(const std::string& name) {
  if (name == "newtonian")
    return relativity::HydroMode::newtonian;
  if (name == "sr" || name == "special_relativity")
    return relativity::HydroMode::sr;
  if (name == "gr" || name == "fixed_gr")
    return relativity::HydroMode::gr;
  PARTHENON_FAIL("mhd/physics must be newtonian, sr, special_relativity, gr, or fixed_gr");
}

hydro::Reconstruction ParseReconstruction(const std::string& name) {
  const auto method = reconstruct::Parse(name);
  if (!method)
    PARTHENON_FAIL("mhd/reconstruct must be dc, plm, ppm, ppm4, ppmc, or wenoz");
  return *method;
}

riemann::Solver ParseRiemann(const std::string& name) {
  const auto solver = riemann::Parse(name);
  if (!solver)
    PARTHENON_FAIL("mhd/rsolver must name a registered Riemann solver, none, or advect");
  return *solver;
}

eos::NewtonianMHDEOS GetEos(const std::shared_ptr<StateDescriptor>& package) {
  return {static_cast<EosMode>(package->Param<int>("eos_mode")), package->Param<Real>("gamma"),
          package->Param<Real>("iso_sound_speed"), package->Param<Real>("density_floor"),
          package->Param<Real>("pressure_floor")};
}

template <int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION Real ReadOffset(const Pack& pack, const int component, const int k,
                                       const int j, const int i, const int offset) {
  if constexpr (Direction == 0)
    return pack(component, k, j, i + offset);
  if constexpr (Direction == 1)
    return pack(component, k, j + offset, i);
  return pack(component, k + offset, j, i);
}

template <hydro::Reconstruction Method, int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION void ReconstructValue(const Pack& values, const int component, const int k,
                                             const int j, const int i, Real& left, Real& right) {
  reconstruct::ReconstructFaceStencil<Method>(
      [&](const int offset) { return ReadOffset<Direction>(values, component, k, j, i, offset); },
      left, right);
}

template <hydro::Reconstruction Method, int Direction, typename PrimitivePack,
          typename MagneticPack>
KOKKOS_INLINE_FUNCTION void
ReconstructState(const PrimitivePack& primitive, const MagneticPack& magnetic, const int k,
                 const int j, const int i, const Real normal_magnetic,
                 const eos::NewtonianMHDEOS& eos, Primitive& left, Primitive& right) {
  Real lv[kIdealComponents]{};
  Real rv[kIdealComponents]{};
  const int components = eos.HasEnergy() ? kIdealComponents : kIsothermalComponents;
  for (int n = 0; n < components; ++n)
    ReconstructValue<Method, Direction>(primitive, n, k, j, i, lv[n], rv[n]);
  left.density = fmax(lv[IDN], eos.density_floor);
  right.density = fmax(rv[IDN], eos.density_floor);
  for (int axis = 0; axis < 3; ++axis) {
    left.velocity[axis] = lv[IV1 + axis];
    right.velocity[axis] = rv[IV1 + axis];
    if (axis == Direction) {
      left.magnetic[axis] = normal_magnetic;
      right.magnetic[axis] = normal_magnetic;
    } else {
      ReconstructValue<Method, Direction>(magnetic, axis, k, j, i, left.magnetic[axis],
                                          right.magnetic[axis]);
    }
  }
  if (eos.HasEnergy()) {
    left.pressure = fmax(lv[IPR], eos.pressure_floor);
    right.pressure = fmax(rv[IPR], eos.pressure_floor);
  } else {
    left.pressure = eos.iso_sound_speed * eos.iso_sound_speed * left.density;
    right.pressure = eos.iso_sound_speed * eos.iso_sound_speed * right.density;
  }
}

template <riemann::Solver Solver>
KOKKOS_INLINE_FUNCTION void SolveInterface(const Primitive& left, const Primitive& right,
                                           const eos::NewtonianMHDEOS& eos, const int direction,
                                           InterfaceFlux& flux) {
  riemann::Implementation<riemann::Physics::newtonian_mhd, Solver>::Solve(left, right, eos,
                                                                         direction, flux);
}

template <hydro::Reconstruction Method, riemann::Solver Solver, int Direction>
void CalculateDirectionalFluxesBlock(const std::shared_ptr<MeshBlockData<Real>>& data,
                                     const eos::NewtonianMHDEOS& eos) {
  auto block = data->GetBlockPointer();
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto magnetic = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
  auto face_emf = data->PackVariables(std::vector<std::string>{"mhd.face_emf"});
  auto bface = data->Get("mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const int ndim = block->pmy_mesh->ndim;
  const int components = eos.HasEnergy() ? kIdealComponents : kIsothermalComponents;
  const int scalars = block->packages.Get("mhd")->Param<int>("nscalars");

  const int k_extension = ndim >= 3 ? 1 : 0;
  const int j_extension = ndim >= 2 ? 1 : 0;
  if constexpr (Direction == 0) {
    block->par_for(
        "PANGU MHD x1 flux", kb.s - k_extension, kb.e + k_extension, jb.s - j_extension,
        jb.e + j_extension, ib.s, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Primitive left{}, right{};
          const Real bn = bface(0, 0, 0, 0, k, j, i);
          ReconstructState<Method, Direction>(primitive, magnetic, k, j, i, bn, eos, left, right);
          InterfaceFlux flux{};
          SolveInterface<Solver>(left, right, eos, Direction, flux);
          for (int n = 0; n < components; ++n)
            conserved.flux(X1DIR, n, k, j, i) = flux.fluid[n];
          face_emf(E2X1, k, j, i) = flux.electric_t1;
          face_emf(E3X1, k, j, i) = flux.electric_t2;
          for (int scalar = 0; scalar < scalars; ++scalar) {
            Real scalar_left, scalar_right;
            ReconstructValue<Method, Direction>(primitive, components + scalar, k, j, i,
                                                scalar_left, scalar_right);
            const Real fraction = flux.fluid[IDN] >= 0.0 ? scalar_left : scalar_right;
            conserved.flux(X1DIR, components + scalar, k, j, i) =
                flux.fluid[IDN] * fmin(fmax(fraction, 0.0), 1.0);
          }
        });
  } else if constexpr (Direction == 1) {
    block->par_for(
        "PANGU MHD x2 flux", kb.s - k_extension, kb.e + k_extension, jb.s, jb.e + 1, ib.s - 1,
        ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Primitive left{}, right{};
          const Real bn = bface(1, 0, 0, 0, k, j, i);
          ReconstructState<Method, Direction>(primitive, magnetic, k, j, i, bn, eos, left, right);
          InterfaceFlux flux{};
          SolveInterface<Solver>(left, right, eos, Direction, flux);
          for (int n = 0; n < components; ++n)
            conserved.flux(X2DIR, n, k, j, i) = flux.fluid[n];
          face_emf(E3X2, k, j, i) = flux.electric_t1;
          face_emf(E1X2, k, j, i) = flux.electric_t2;
          for (int scalar = 0; scalar < scalars; ++scalar) {
            Real scalar_left, scalar_right;
            ReconstructValue<Method, Direction>(primitive, components + scalar, k, j, i,
                                                scalar_left, scalar_right);
            const Real fraction = flux.fluid[IDN] >= 0.0 ? scalar_left : scalar_right;
            conserved.flux(X2DIR, components + scalar, k, j, i) =
                flux.fluid[IDN] * fmin(fmax(fraction, 0.0), 1.0);
          }
        });
  } else {
    block->par_for(
        "PANGU MHD x3 flux", kb.s, kb.e + 1, jb.s - 1, jb.e + 1, ib.s - 1, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Primitive left{}, right{};
          const Real bn = bface(2, 0, 0, 0, k, j, i);
          ReconstructState<Method, Direction>(primitive, magnetic, k, j, i, bn, eos, left, right);
          InterfaceFlux flux{};
          SolveInterface<Solver>(left, right, eos, Direction, flux);
          for (int n = 0; n < components; ++n)
            conserved.flux(X3DIR, n, k, j, i) = flux.fluid[n];
          face_emf(E1X3, k, j, i) = flux.electric_t1;
          face_emf(E2X3, k, j, i) = flux.electric_t2;
          for (int scalar = 0; scalar < scalars; ++scalar) {
            Real scalar_left, scalar_right;
            ReconstructValue<Method, Direction>(primitive, components + scalar, k, j, i,
                                                scalar_left, scalar_right);
            const Real fraction = flux.fluid[IDN] >= 0.0 ? scalar_left : scalar_right;
            conserved.flux(X3DIR, components + scalar, k, j, i) =
                flux.fluid[IDN] * fmin(fmax(fraction, 0.0), 1.0);
          }
        });
  }
}

TaskStatus CalculateCornerEMFBlock(const std::shared_ptr<MeshBlockData<Real>>& data) {
  auto block = data->GetBlockPointer();
  const auto physics =
      static_cast<relativity::HydroMode>(block->packages.Get("mhd")->Param<int>("physics_mode"));
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto cell_emf = data->PackVariables(std::vector<std::string>{"mhd.cell_emf"});
  auto face_emf = data->PackVariables(std::vector<std::string>{"mhd.face_emf"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
  auto edge = data->Get("bnd_flux::mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coords = block->coords;
  const int ndim = block->pmy_mesh->ndim;

  // A spherical MKS pole is a zero-area physical face, not an ordinary
  // Cartesian reflecting wall. CorrectMKSPolarBoundary supplies KHARMA's cell
  // and face parity; zero every fluid flux and both tangential CT electric
  // fields on the physical pole here.
  const bool mks_geometry = std::string(geometry::ConfiguredMetricName()) == "mks";
  const bool inner_pole =
      mks_geometry &&
      block->boundary_flag[parthenon::BoundaryFace::inner_x2] == parthenon::BoundaryFlag::user &&
      block->pmy_mesh->mesh_bc_names[2] == "reflecting";
  const bool outer_pole =
      mks_geometry &&
      block->boundary_flag[parthenon::BoundaryFace::outer_x2] == parthenon::BoundaryFlag::user &&
      block->pmy_mesh->mesh_bc_names[3] == "reflecting";
  if (ndim >= 2 && (inner_pole || outer_pole)) {
    const int components = conserved.GetDim(4);
    if (inner_pole)
      block->par_for(
          "PANGU MKS inner-pole zero flux", 0, components - 1, kb.s, kb.e, jb.s, jb.s,
          ib.s - 1, ib.e + 1,
          KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
            conserved.flux(X2DIR, n, k, j, i) = 0.0;
          });
    if (outer_pole)
      block->par_for(
          "PANGU MKS outer-pole zero flux", 0, components - 1, kb.s, kb.e, jb.e + 1, jb.e + 1,
          ib.s - 1, ib.e + 1,
          KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
            conserved.flux(X2DIR, n, k, j, i) = 0.0;
          });
  }

  const auto zero_polar_emf = [&]() {
    if (ndim < 2 || (!inner_pole && !outer_pole))
      return;
    const int inner_j = jb.s;
    const int outer_j = jb.e + 1;
    if (inner_pole) {
      block->par_for(
          "PANGU MKS inner-pole zero E1", kb.s, kb.e + (ndim >= 3), inner_j, inner_j, ib.s,
          ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            edge(0, 0, 0, 0, k, j, i) = 0.0;
          });
      block->par_for(
          "PANGU MKS inner-pole zero E3", kb.s, kb.e, inner_j, inner_j, ib.s, ib.e + 1,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            edge(2, 0, 0, 0, k, j, i) = 0.0;
          });
    }
    if (outer_pole) {
      block->par_for(
          "PANGU MKS outer-pole zero E1", kb.s, kb.e + (ndim >= 3), outer_j, outer_j, ib.s,
          ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            edge(0, 0, 0, 0, k, j, i) = 0.0;
          });
      block->par_for(
          "PANGU MKS outer-pole zero E3", kb.s, kb.e, outer_j, outer_j, ib.s, ib.e + 1,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            edge(2, 0, 0, 0, k, j, i) = 0.0;
          });
    }
  };

  const int ex = 1;
  const int ey = ndim >= 2 ? 1 : 0;
  const int ez = ndim >= 3 ? 1 : 0;
  block->par_for(
      "PANGU MHD cell electric field", kb.s - ez, kb.e + ez, jb.s - ey, jb.e + ey, ib.s - ex,
      ib.e + ex, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real v1 = primitive(IV1, k, j, i);
        const Real v2 = primitive(IV2, k, j, i);
        const Real v3 = primitive(IV3, k, j, i);
        const Real b1 = bcell(0, k, j, i);
        const Real b2 = bcell(1, k, j, i);
        const Real b3 = bcell(2, k, j, i);
        if (physics == relativity::HydroMode::newtonian) {
          cell_emf(0, k, j, i) = v3 * b2 - v2 * b3;
          cell_emf(1, k, j, i) = v1 * b3 - v3 * b1;
          cell_emf(2, k, j, i) = v2 * b1 - v1 * b2;
        } else if (physics == relativity::HydroMode::sr) {
          const Real u0 = sqrt(1.0 + v1 * v1 + v2 * v2 + v3 * v3);
          cell_emf(0, k, j, i) = (v3 * b2 - v2 * b3) / u0;
          cell_emf(1, k, j, i) = (v1 * b3 - v3 * b1) / u0;
          cell_emf(2, k, j, i) = (v2 * b1 - v1 * b2) / u0;
        } else {
          const auto metric =
              spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
          const Real spatial_u2 =
              metric.lower[1][1] * v1 * v1 + 2.0 * metric.lower[1][2] * v1 * v2 +
              2.0 * metric.lower[1][3] * v1 * v3 + metric.lower[2][2] * v2 * v2 +
              2.0 * metric.lower[2][3] * v2 * v3 + metric.lower[3][3] * v3 * v3;
          const Real lorentz = sqrt(1.0 + spatial_u2);
          const Real alpha = sqrt(-1.0 / metric.upper[0][0]);
          const Real u0 = lorentz / alpha;
          const Real u1 = v1 - alpha * lorentz * metric.upper[0][1];
          const Real u2 = v2 - alpha * lorentz * metric.upper[0][2];
          const Real u3 = v3 - alpha * lorentz * metric.upper[0][3];
          const Real u_1 = metric.lower[1][0] * u0 + metric.lower[1][1] * u1 +
                           metric.lower[1][2] * u2 + metric.lower[1][3] * u3;
          const Real u_2 = metric.lower[2][0] * u0 + metric.lower[2][1] * u1 +
                           metric.lower[2][2] * u2 + metric.lower[2][3] * u3;
          const Real u_3 = metric.lower[3][0] * u0 + metric.lower[3][1] * u1 +
                           metric.lower[3][2] * u2 + metric.lower[3][3] * u3;
          const Real b0 = u_1 * b1 + u_2 * b2 + u_3 * b3;
          const Real bb1 = (b1 + b0 * u1) / u0;
          const Real bb2 = (b2 + b0 * u2) / u0;
          const Real bb3 = (b3 + b0 * u3) / u0;
          Real emf1 = bb2 * u3 - bb3 * u2;
          Real emf2 = bb3 * u1 - bb1 * u3;
          Real emf3 = bb1 * u2 - bb2 * u1;
          if (!decltype(spacetime)::unit_determinant) {
            emf1 *= metric.gdet;
            emf2 *= metric.gdet;
            emf3 *= metric.gdet;
          }
          cell_emf(0, k, j, i) = emf1;
          cell_emf(1, k, j, i) = emf2;
          cell_emf(2, k, j, i) = emf3;
        }
      });

  if (ndim == 1) {
    block->par_for(
        "PANGU MHD 1D corner EMF", ib.s, ib.e + 1, KOKKOS_LAMBDA(const int i) {
          edge(1, 0, 0, 0, kb.s, jb.s, i) = face_emf(E2X1, kb.s, jb.s, i);
          edge(2, 0, 0, 0, kb.s, jb.s, i) = face_emf(E3X1, kb.s, jb.s, i);
        });
    zero_polar_emf();
    return TaskStatus::complete;
  }

  if (ndim == 2) {
    block->par_for(
        "PANGU MHD 2D corner EMF", jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int j, const int i) {
          edge(1, 0, 0, 0, kb.s, j, i) = face_emf(E2X1, kb.s, j, i);
          edge(0, 0, 0, 0, kb.s, j, i) = face_emf(E1X2, kb.s, j, i);
          Real e3_l2, e3_r2, e3_l1, e3_r1;
          if (conserved.flux(X1DIR, IDN, kb.s, j - 1, i) >= 0.0)
            e3_l2 = face_emf(E3X2, kb.s, j, i - 1) - cell_emf(2, kb.s, j - 1, i - 1);
          else
            e3_l2 = face_emf(E3X2, kb.s, j, i) - cell_emf(2, kb.s, j - 1, i);
          if (conserved.flux(X1DIR, IDN, kb.s, j, i) >= 0.0)
            e3_r2 = face_emf(E3X2, kb.s, j, i - 1) - cell_emf(2, kb.s, j, i - 1);
          else
            e3_r2 = face_emf(E3X2, kb.s, j, i) - cell_emf(2, kb.s, j, i);
          if (conserved.flux(X2DIR, IDN, kb.s, j, i - 1) >= 0.0)
            e3_l1 = face_emf(E3X1, kb.s, j - 1, i) - cell_emf(2, kb.s, j - 1, i - 1);
          else
            e3_l1 = face_emf(E3X1, kb.s, j, i) - cell_emf(2, kb.s, j, i - 1);
          if (conserved.flux(X2DIR, IDN, kb.s, j, i) >= 0.0)
            e3_r1 = face_emf(E3X1, kb.s, j - 1, i) - cell_emf(2, kb.s, j - 1, i);
          else
            e3_r1 = face_emf(E3X1, kb.s, j, i) - cell_emf(2, kb.s, j, i);
          edge(2, 0, 0, 0, kb.s, j, i) =
              0.25 * (e3_l1 + e3_r1 + e3_l2 + e3_r2 + face_emf(E3X2, kb.s, j, i - 1) +
                      face_emf(E3X2, kb.s, j, i) + face_emf(E3X1, kb.s, j - 1, i) +
                      face_emf(E3X1, kb.s, j, i));
        });
    zero_polar_emf();
    return TaskStatus::complete;
  }

  block->par_for(
      "PANGU MHD 3D corner EMF", kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real e1_l3, e1_r3, e1_l2, e1_r2;
        if (conserved.flux(X2DIR, IDN, k - 1, j, i) >= 0.0)
          e1_l3 = face_emf(E1X3, k, j - 1, i) - cell_emf(0, k - 1, j - 1, i);
        else
          e1_l3 = face_emf(E1X3, k, j, i) - cell_emf(0, k - 1, j, i);
        if (conserved.flux(X2DIR, IDN, k, j, i) >= 0.0)
          e1_r3 = face_emf(E1X3, k, j - 1, i) - cell_emf(0, k, j - 1, i);
        else
          e1_r3 = face_emf(E1X3, k, j, i) - cell_emf(0, k, j, i);
        if (conserved.flux(X3DIR, IDN, k, j - 1, i) >= 0.0)
          e1_l2 = face_emf(E1X2, k - 1, j, i) - cell_emf(0, k - 1, j - 1, i);
        else
          e1_l2 = face_emf(E1X2, k, j, i) - cell_emf(0, k, j - 1, i);
        if (conserved.flux(X3DIR, IDN, k, j, i) >= 0.0)
          e1_r2 = face_emf(E1X2, k - 1, j, i) - cell_emf(0, k - 1, j, i);
        else
          e1_r2 = face_emf(E1X2, k, j, i) - cell_emf(0, k, j, i);
        edge(0, 0, 0, 0, k, j, i) = 0.25 * (e1_l3 + e1_r3 + e1_l2 + e1_r2 +
                                            face_emf(E1X2, k - 1, j, i) + face_emf(E1X2, k, j, i) +
                                            face_emf(E1X3, k, j - 1, i) + face_emf(E1X3, k, j, i));
        Real e2_l3, e2_r3, e2_l1, e2_r1;
        if (conserved.flux(X1DIR, IDN, k - 1, j, i) >= 0.0)
          e2_l3 = face_emf(E2X3, k, j, i - 1) - cell_emf(1, k - 1, j, i - 1);
        else
          e2_l3 = face_emf(E2X3, k, j, i) - cell_emf(1, k - 1, j, i);
        if (conserved.flux(X1DIR, IDN, k, j, i) >= 0.0)
          e2_r3 = face_emf(E2X3, k, j, i - 1) - cell_emf(1, k, j, i - 1);
        else
          e2_r3 = face_emf(E2X3, k, j, i) - cell_emf(1, k, j, i);
        if (conserved.flux(X3DIR, IDN, k, j, i - 1) >= 0.0)
          e2_l1 = face_emf(E2X1, k - 1, j, i) - cell_emf(1, k - 1, j, i - 1);
        else
          e2_l1 = face_emf(E2X1, k, j, i) - cell_emf(1, k, j, i - 1);
        if (conserved.flux(X3DIR, IDN, k, j, i) >= 0.0)
          e2_r1 = face_emf(E2X1, k - 1, j, i) - cell_emf(1, k - 1, j, i);
        else
          e2_r1 = face_emf(E2X1, k, j, i) - cell_emf(1, k, j, i);
        edge(1, 0, 0, 0, k, j, i) = 0.25 * (e2_l3 + e2_r3 + e2_l1 + e2_r1 +
                                            face_emf(E2X3, k, j, i - 1) + face_emf(E2X3, k, j, i) +
                                            face_emf(E2X1, k - 1, j, i) + face_emf(E2X1, k, j, i));

        Real e3_l2, e3_r2, e3_l1, e3_r1;
        if (conserved.flux(X1DIR, IDN, k, j - 1, i) >= 0.0)
          e3_l2 = face_emf(E3X2, k, j, i - 1) - cell_emf(2, k, j - 1, i - 1);
        else
          e3_l2 = face_emf(E3X2, k, j, i) - cell_emf(2, k, j - 1, i);
        if (conserved.flux(X1DIR, IDN, k, j, i) >= 0.0)
          e3_r2 = face_emf(E3X2, k, j, i - 1) - cell_emf(2, k, j, i - 1);
        else
          e3_r2 = face_emf(E3X2, k, j, i) - cell_emf(2, k, j, i);
        if (conserved.flux(X2DIR, IDN, k, j, i - 1) >= 0.0)
          e3_l1 = face_emf(E3X1, k, j - 1, i) - cell_emf(2, k, j - 1, i - 1);
        else
          e3_l1 = face_emf(E3X1, k, j, i) - cell_emf(2, k, j, i - 1);
        if (conserved.flux(X2DIR, IDN, k, j, i) >= 0.0)
          e3_r1 = face_emf(E3X1, k, j - 1, i) - cell_emf(2, k, j - 1, i);
        else
          e3_r1 = face_emf(E3X1, k, j, i) - cell_emf(2, k, j, i);
        edge(2, 0, 0, 0, k, j, i) = 0.25 * (e3_l1 + e3_r1 + e3_l2 + e3_r2 +
                                            face_emf(E3X2, k, j, i - 1) + face_emf(E3X2, k, j, i) +
                                            face_emf(E3X1, k, j - 1, i) + face_emf(E3X1, k, j, i));
      });
  zero_polar_emf();
  return TaskStatus::complete;
}

TaskStatus CalculateCornerEMFMesh(MeshData<Real>* data) {
  using TE = parthenon::TopologicalElement;
  auto* mesh = data->GetParentPointer();
  const int ndim = mesh->ndim;
  // KHARMA's face-centred CT intentionally supports only 2D and 3D.  Preserve
  // PANGU's legacy 1D compatibility path, while all KHARMA-relevant CT work is
  // executed pack-wide below.
  if (ndim == 1) {
    for (int b = 0; b < data->NumBlocks(); ++b)
      CalculateCornerEMFBlock(data->GetBlockData(b));
    return TaskStatus::complete;
  }
  const auto physics =
      static_cast<relativity::HydroMode>(mesh->packages.Get("mhd")->Param<int>("physics_mode"));
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto cell_emf = data->PackVariables(std::vector<std::string>{"mhd.cell_emf"});
  auto face_emf = data->PackVariables(std::vector<std::string>{"mhd.face_emf"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
  auto edge_emf = data->PackVariables(std::vector<std::string>{"mhd.edge_emf"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int nblocks = data->NumBlocks();

  // KHARMA evaluates CT over a MeshData pack and keeps the MeshBlock index as
  // an ordinary kernel dimension.  Do the same for every dimensionality.  In
  // particular, a 512-zone radial block then becomes part of one flattened
  // (block,k,j,i) range instead of a rank-2 MDRange tile containing 513 edge
  // points.
  const bool mks_geometry = std::string(geometry::ConfiguredMetricName()) == "mks";
  Kokkos::View<int**> polar_boundary;
  if (mks_geometry && ndim >= 2) {
    polar_boundary = Kokkos::View<int**>("PANGU packed MKS polar flags", nblocks, 2);
    auto polar_boundary_host = Kokkos::create_mirror_view(polar_boundary);
    for (int b = 0; b < nblocks; ++b) {
      auto block = data->GetBlockData(b)->GetBlockPointer();
      polar_boundary_host(b, 0) =
          block->boundary_flag[parthenon::BoundaryFace::inner_x2] ==
              parthenon::BoundaryFlag::user &&
          mesh->mesh_bc_names[2] == "reflecting";
      polar_boundary_host(b, 1) =
          block->boundary_flag[parthenon::BoundaryFace::outer_x2] ==
              parthenon::BoundaryFlag::user &&
          mesh->mesh_bc_names[3] == "reflecting";
    }
    Kokkos::deep_copy(polar_boundary, polar_boundary_host);

    const int components = conserved.GetDim(4);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU packed MKS polar zero flux", parthenon::DevExecSpace(), 0,
        nblocks - 1, 0, 1, 0, components - 1, kb.s - (ndim >= 3),
        kb.e + (ndim >= 3), ib.s - 1, ib.e + 1,
        KOKKOS_LAMBDA(const int b, const int side, const int n, const int k, const int i) {
          if (polar_boundary(b, side) == 0)
            return;
          const int j = side == 0 ? jb.s : jb.e + 1;
          auto block_conserved = conserved(b);
          block_conserved.flux(X2DIR, n, k, j, i) = 0.0;
        });
  }

  const int ex = 1;
  const int ey = ndim >= 2 ? 1 : 0;
  const int ez = ndim >= 3 ? 1 : 0;
  if constexpr (geometry::ConfiguredGeometryIsSynchronized()) {
    const auto z4c = data->PackVariables(std::vector<std::string>{"nr.z4c"});
    const auto adm = data->PackVariables(std::vector<std::string>{"nr.adm"});
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU packed sync GRMHD cell electric field",
        parthenon::DevExecSpace(), 0, nblocks - 1, kb.s - ez, kb.e + ez, jb.s - ey,
        jb.e + ey, ib.s - ex, ib.e + ex,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const Real u1 = primitive(b, IV1, k, j, i);
          const Real u2 = primitive(b, IV2, k, j, i);
          const Real u3 = primitive(b, IV3, k, j, i);
          const auto& coords = primitive.GetCoords(b);
          const auto metric = spacetime.MetricAt(geometry::Location::cell_center, b, k, j,
                                                 i, coords, z4c, adm);
          const Real inverse_lorentz =
              1.0 / sqrt(1.0 + metric.lower[1][1] * u1 * u1 +
                         2.0 * metric.lower[1][2] * u1 * u2 +
                         2.0 * metric.lower[1][3] * u1 * u3 +
                         metric.lower[2][2] * u2 * u2 +
                         2.0 * metric.lower[2][3] * u2 * u3 +
                         metric.lower[3][3] * u3 * u3);
          const Real v1 = metric.lapse * u1 * inverse_lorentz - metric.shift[0];
          const Real v2 = metric.lapse * u2 * inverse_lorentz - metric.shift[1];
          const Real v3 = metric.lapse * u3 * inverse_lorentz - metric.shift[2];
          const Real b1 = bcell(b, 0, k, j, i);
          const Real b2 = bcell(b, 1, k, j, i);
          const Real b3 = bcell(b, 2, k, j, i);
          cell_emf(b, 0, k, j, i) = b2 * v3 - b3 * v2;
          cell_emf(b, 1, k, j, i) = b3 * v1 - b1 * v3;
          cell_emf(b, 2, k, j, i) = b1 * v2 - b2 * v1;
        });
  } else {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU packed MHD cell electric field",
        parthenon::DevExecSpace(), 0, nblocks - 1, kb.s - ez, kb.e + ez, jb.s - ey,
        jb.e + ey, ib.s - ex, ib.e + ex,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real v1 = primitive(b, IV1, k, j, i);
        const Real v2 = primitive(b, IV2, k, j, i);
        const Real v3 = primitive(b, IV3, k, j, i);
        const Real b1 = bcell(b, 0, k, j, i);
        const Real b2 = bcell(b, 1, k, j, i);
        const Real b3 = bcell(b, 2, k, j, i);
        if (physics == relativity::HydroMode::newtonian) {
          cell_emf(b, 0, k, j, i) = v3 * b2 - v2 * b3;
          cell_emf(b, 1, k, j, i) = v1 * b3 - v3 * b1;
          cell_emf(b, 2, k, j, i) = v2 * b1 - v1 * b2;
        } else if (physics == relativity::HydroMode::sr) {
          const Real u0 = sqrt(1.0 + v1 * v1 + v2 * v2 + v3 * v3);
          cell_emf(b, 0, k, j, i) = (v3 * b2 - v2 * b3) / u0;
          cell_emf(b, 1, k, j, i) = (v1 * b3 - v3 * b1) / u0;
          cell_emf(b, 2, k, j, i) = (v2 * b1 - v1 * b2) / u0;
        } else {
          const auto& coords = primitive.GetCoords(b);
          const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                                 geometry_block_offset + b, k, j, i, coords);
          const Real spatial_u2 =
              metric.lower[1][1] * v1 * v1 + 2.0 * metric.lower[1][2] * v1 * v2 +
              2.0 * metric.lower[1][3] * v1 * v3 + metric.lower[2][2] * v2 * v2 +
              2.0 * metric.lower[2][3] * v2 * v3 + metric.lower[3][3] * v3 * v3;
          const Real lorentz = sqrt(1.0 + spatial_u2);
          const Real alpha = sqrt(-1.0 / metric.upper[0][0]);
          const Real u0 = lorentz / alpha;
          const Real u1 = v1 - alpha * lorentz * metric.upper[0][1];
          const Real u2 = v2 - alpha * lorentz * metric.upper[0][2];
          const Real u3 = v3 - alpha * lorentz * metric.upper[0][3];
          const Real u_1 = metric.lower[1][0] * u0 + metric.lower[1][1] * u1 +
                           metric.lower[1][2] * u2 + metric.lower[1][3] * u3;
          const Real u_2 = metric.lower[2][0] * u0 + metric.lower[2][1] * u1 +
                           metric.lower[2][2] * u2 + metric.lower[2][3] * u3;
          const Real u_3 = metric.lower[3][0] * u0 + metric.lower[3][1] * u1 +
                           metric.lower[3][2] * u2 + metric.lower[3][3] * u3;
          const Real b0 = u_1 * b1 + u_2 * b2 + u_3 * b3;
          const Real bb1 = (b1 + b0 * u1) / u0;
          const Real bb2 = (b2 + b0 * u2) / u0;
          const Real bb3 = (b3 + b0 * u3) / u0;
          Real emf1 = bb2 * u3 - bb3 * u2;
          Real emf2 = bb3 * u1 - bb1 * u3;
          Real emf3 = bb1 * u2 - bb2 * u1;
          if (!decltype(spacetime)::unit_determinant) {
            emf1 *= metric.gdet;
            emf2 *= metric.gdet;
            emf3 *= metric.gdet;
          }
          cell_emf(b, 0, k, j, i) = emf1;
          cell_emf(b, 1, k, j, i) = emf2;
          cell_emf(b, 2, k, j, i) = emf3;
        }
        });
  }
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed MHD corner EMF", parthenon::DevExecSpace(), 0,
      nblocks - 1, kb.s, kb.e + ez, jb.s, jb.e + ey, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto block_conserved = conserved(b);
        if (ndim == 2) {
          // In 2-D the transverse edge fields reduce directly to their
          // corresponding face electric fields.  Keep them in the same
          // communicable edge variable as E3, exactly as KHARMA does.
          edge_emf(b, TE::E1, 0, k, j, i) = face_emf(b, E1X2, k, j, i);
          edge_emf(b, TE::E2, 0, k, j, i) = face_emf(b, E2X1, k, j, i);
          Real e3_l2, e3_r2, e3_l1, e3_r1;
          if (block_conserved.flux(X1DIR, IDN, k, j - 1, i) >= 0.0)
            e3_l2 = face_emf(b, E3X2, k, j, i - 1) - cell_emf(b, 2, k, j - 1, i - 1);
          else
            e3_l2 = face_emf(b, E3X2, k, j, i) - cell_emf(b, 2, k, j - 1, i);
          if (block_conserved.flux(X1DIR, IDN, k, j, i) >= 0.0)
            e3_r2 = face_emf(b, E3X2, k, j, i - 1) - cell_emf(b, 2, k, j, i - 1);
          else
            e3_r2 = face_emf(b, E3X2, k, j, i) - cell_emf(b, 2, k, j, i);
          if (block_conserved.flux(X2DIR, IDN, k, j, i - 1) >= 0.0)
            e3_l1 = face_emf(b, E3X1, k, j - 1, i) - cell_emf(b, 2, k, j - 1, i - 1);
          else
            e3_l1 = face_emf(b, E3X1, k, j, i) - cell_emf(b, 2, k, j, i - 1);
          if (block_conserved.flux(X2DIR, IDN, k, j, i) >= 0.0)
            e3_r1 = face_emf(b, E3X1, k, j - 1, i) - cell_emf(b, 2, k, j - 1, i);
          else
            e3_r1 = face_emf(b, E3X1, k, j, i) - cell_emf(b, 2, k, j, i);
          const Real e3 =
              0.25 * (e3_l1 + e3_r1 + e3_l2 + e3_r2 + face_emf(b, E3X2, k, j, i - 1) +
                      face_emf(b, E3X2, k, j, i) + face_emf(b, E3X1, k, j - 1, i) +
                      face_emf(b, E3X1, k, j, i));
          edge_emf(b, TE::E3, 0, k, j, i) = e3;
          return;
        }
        Real e1_l3, e1_r3, e1_l2, e1_r2;
        if (block_conserved.flux(X2DIR, IDN, k - 1, j, i) >= 0.0)
          e1_l3 = face_emf(b, E1X3, k, j - 1, i) - cell_emf(b, 0, k - 1, j - 1, i);
        else
          e1_l3 = face_emf(b, E1X3, k, j, i) - cell_emf(b, 0, k - 1, j, i);
        if (block_conserved.flux(X2DIR, IDN, k, j, i) >= 0.0)
          e1_r3 = face_emf(b, E1X3, k, j - 1, i) - cell_emf(b, 0, k, j - 1, i);
        else
          e1_r3 = face_emf(b, E1X3, k, j, i) - cell_emf(b, 0, k, j, i);
        if (block_conserved.flux(X3DIR, IDN, k, j - 1, i) >= 0.0)
          e1_l2 = face_emf(b, E1X2, k - 1, j, i) - cell_emf(b, 0, k - 1, j - 1, i);
        else
          e1_l2 = face_emf(b, E1X2, k, j, i) - cell_emf(b, 0, k, j - 1, i);
        if (block_conserved.flux(X3DIR, IDN, k, j, i) >= 0.0)
          e1_r2 = face_emf(b, E1X2, k - 1, j, i) - cell_emf(b, 0, k - 1, j, i);
        else
          e1_r2 = face_emf(b, E1X2, k, j, i) - cell_emf(b, 0, k, j, i);
        const Real e1 =
            0.25 * (e1_l3 + e1_r3 + e1_l2 + e1_r2 + face_emf(b, E1X2, k - 1, j, i) +
                    face_emf(b, E1X2, k, j, i) + face_emf(b, E1X3, k, j - 1, i) +
                    face_emf(b, E1X3, k, j, i));
        edge_emf(b, TE::E1, 0, k, j, i) = e1;
        Real e2_l3, e2_r3, e2_l1, e2_r1;
        if (block_conserved.flux(X1DIR, IDN, k - 1, j, i) >= 0.0)
          e2_l3 = face_emf(b, E2X3, k, j, i - 1) - cell_emf(b, 1, k - 1, j, i - 1);
        else
          e2_l3 = face_emf(b, E2X3, k, j, i) - cell_emf(b, 1, k - 1, j, i);
        if (block_conserved.flux(X1DIR, IDN, k, j, i) >= 0.0)
          e2_r3 = face_emf(b, E2X3, k, j, i - 1) - cell_emf(b, 1, k, j, i - 1);
        else
          e2_r3 = face_emf(b, E2X3, k, j, i) - cell_emf(b, 1, k, j, i);
        if (block_conserved.flux(X3DIR, IDN, k, j, i - 1) >= 0.0)
          e2_l1 = face_emf(b, E2X1, k - 1, j, i) - cell_emf(b, 1, k - 1, j, i - 1);
        else
          e2_l1 = face_emf(b, E2X1, k, j, i) - cell_emf(b, 1, k, j, i - 1);
        if (block_conserved.flux(X3DIR, IDN, k, j, i) >= 0.0)
          e2_r1 = face_emf(b, E2X1, k - 1, j, i) - cell_emf(b, 1, k - 1, j, i);
        else
          e2_r1 = face_emf(b, E2X1, k, j, i) - cell_emf(b, 1, k, j, i);
        const Real e2 =
            0.25 * (e2_l3 + e2_r3 + e2_l1 + e2_r1 + face_emf(b, E2X3, k, j, i - 1) +
                    face_emf(b, E2X3, k, j, i) + face_emf(b, E2X1, k - 1, j, i) +
                    face_emf(b, E2X1, k, j, i));
        edge_emf(b, TE::E2, 0, k, j, i) = e2;
        Real e3_l2, e3_r2, e3_l1, e3_r1;
        if (block_conserved.flux(X1DIR, IDN, k, j - 1, i) >= 0.0)
          e3_l2 = face_emf(b, E3X2, k, j, i - 1) - cell_emf(b, 2, k, j - 1, i - 1);
        else
          e3_l2 = face_emf(b, E3X2, k, j, i) - cell_emf(b, 2, k, j - 1, i);
        if (block_conserved.flux(X1DIR, IDN, k, j, i) >= 0.0)
          e3_r2 = face_emf(b, E3X2, k, j, i - 1) - cell_emf(b, 2, k, j, i - 1);
        else
          e3_r2 = face_emf(b, E3X2, k, j, i) - cell_emf(b, 2, k, j, i);
        if (block_conserved.flux(X2DIR, IDN, k, j, i - 1) >= 0.0)
          e3_l1 = face_emf(b, E3X1, k, j - 1, i) - cell_emf(b, 2, k, j - 1, i - 1);
        else
          e3_l1 = face_emf(b, E3X1, k, j, i) - cell_emf(b, 2, k, j, i - 1);
        if (block_conserved.flux(X2DIR, IDN, k, j, i) >= 0.0)
          e3_r1 = face_emf(b, E3X1, k, j - 1, i) - cell_emf(b, 2, k, j - 1, i);
        else
          e3_r1 = face_emf(b, E3X1, k, j, i) - cell_emf(b, 2, k, j, i);
        const Real e3 =
            0.25 * (e3_l1 + e3_r1 + e3_l2 + e3_r2 + face_emf(b, E3X2, k, j, i - 1) +
                    face_emf(b, E3X2, k, j, i) + face_emf(b, E3X1, k, j - 1, i) +
                    face_emf(b, E3X1, k, j, i));
        edge_emf(b, TE::E3, 0, k, j, i) = e3;
      });

  if (mks_geometry && ndim >= 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PANGU packed MKS polar zero EMF", parthenon::DevExecSpace(), 0,
        nblocks - 1, 0, 1, kb.s, kb.e + ez, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int b, const int side, const int k, const int i) {
          if (polar_boundary(b, side) == 0)
            return;
          const int j = side == 0 ? jb.s : jb.e + 1;
          if (i <= ib.e)
            edge_emf(b, TE::E1, 0, k, j, i) = 0.0;
          if (k <= kb.e)
            edge_emf(b, TE::E3, 0, k, j, i) = 0.0;
        });
  }
  return TaskStatus::complete;
}

template <hydro::Reconstruction Method, riemann::Solver Solver>
TaskStatus CalculateFluxesBlockImpl(std::shared_ptr<MeshBlockData<Real>>& data) {
  const auto block = data->GetBlockPointer();
  const auto eos = GetEos(block->packages.Get("mhd"));
  CalculateDirectionalFluxesBlock<Method, Solver, 0>(data, eos);
  if (block->pmy_mesh->ndim >= 2)
    CalculateDirectionalFluxesBlock<Method, Solver, 1>(data, eos);
  if (block->pmy_mesh->ndim >= 3)
    CalculateDirectionalFluxesBlock<Method, Solver, 2>(data, eos);
  return CalculateCornerEMFBlock(data);
}

template <int Component> Real ConservedHistoryMesh(MeshData<Real>* data) {
  const auto block = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  Real total = 0.0;
  ParReduce(
      "PANGU MHD conserved history", 0, conserved.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        local += conserved(b, Component, k, j, i) * conserved.GetCoords(b).CellVolume(k, j, i);
      },
      Kokkos::Sum<Real>(total));
  return total;
}

Real MaxDivBHistoryMesh(MeshData<Real>* data) {
  const auto block = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto divb = data->PackVariables(std::vector<std::string>{"mhd.divb"});
  Real maximum = 0.0;
  ParReduce(
      "PANGU max divB history", 0, divb.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        local = fmax(local, fabs(divb(b, 0, k, j, i)));
      },
      Kokkos::Max<Real>(maximum));
  return maximum;
}

Real FOFCHistoryMesh(MeshData<Real>* data) {
  const auto block = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  Real total = 0.0;
  ParReduce(
      "PANGU MHD FOFC history", 0, flags.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        local += flags(b, 0, k, j, i);
      },
      Kokkos::Sum<Real>(total));
  return total;
}

template <int Component> Real RecoveryHistoryMesh(MeshData<Real>* data) {
  const auto block = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto recovery = data->PackVariables(std::vector<std::string>{"mhd.recovery"});
  Real total = 0.0;
  ParReduce(
      "PANGU sync GRMHD recovery history", 0, recovery.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        const Real weight = Component == 2 ? 1.0 : recovery.GetCoords(b).CellVolume(k, j, i);
        local += recovery(b, Component, k, j, i) * weight;
      },
      Kokkos::Sum<Real>(total));
  return total;
}

template <int Direction, typename PrimitivePack, typename FlagPack, typename ConservedPack,
          typename BCellPack, typename BFaceArray, typename FaceEmfPack>
void ReplaceMHDFlaggedFacesBlock(MeshBlock* block, const PrimitivePack& primitive,
                                 const BCellPack& bcell, const FlagPack& flags,
                                 ConservedPack& conserved, const BFaceArray& bface,
                                 const FaceEmfPack& face_emf, const eos::NewtonianMHDEOS& eos,
                                 const IndexRange& ib, const IndexRange& jb, const IndexRange& kb) {
  int il = ib.s;
  int iu = ib.e;
  int jl = jb.s;
  int ju = jb.e;
  int kl = kb.s;
  int ku = kb.e;
  if constexpr (Direction == 0)
    ++iu;
  if constexpr (Direction == 1)
    ++ju;
  if constexpr (Direction == 2)
    ++ku;
  const int components = eos.HasEnergy() ? kIdealComponents : kIsothermalComponents;
  const int scalars = block->packages.Get("mhd")->Param<int>("nscalars");
  constexpr int emf_t1 = Direction == 0 ? E2X1 : (Direction == 1 ? E3X2 : E1X3);
  constexpr int emf_t2 = Direction == 0 ? E3X1 : (Direction == 1 ? E1X2 : E2X3);
  block->par_for(
      "PANGU MHD FOFC face", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int left_k = k - (Direction == 2);
        const int left_j = j - (Direction == 1);
        const int left_i = i - (Direction == 0);
        const bool has_left = Direction == 0 ? i > ib.s : Direction == 1 ? j > jb.s : k > kb.s;
        const bool has_right = Direction == 0 ? i <= ib.e : Direction == 1 ? j <= jb.e : k <= kb.e;
        const bool left_bad = has_left && flags(0, left_k, left_j, left_i) > 0.5;
        const bool right_bad = has_right && flags(0, k, j, i) > 0.5;
        if (!(left_bad || right_bad))
          return;

        Primitive left{};
        Primitive right{};
        left.density = fmax(primitive(IDN, left_k, left_j, left_i), eos.density_floor);
        right.density = fmax(primitive(IDN, k, j, i), eos.density_floor);
        for (int axis = 0; axis < 3; ++axis) {
          left.velocity[axis] = primitive(IV1 + axis, left_k, left_j, left_i);
          right.velocity[axis] = primitive(IV1 + axis, k, j, i);
          left.magnetic[axis] = bcell(axis, left_k, left_j, left_i);
          right.magnetic[axis] = bcell(axis, k, j, i);
        }
        const Real normal = bface(Direction, 0, 0, 0, k, j, i);
        left.magnetic[Direction] = normal;
        right.magnetic[Direction] = normal;
        left.pressure = eos.HasEnergy()
                            ? fmax(primitive(IPR, left_k, left_j, left_i), eos.pressure_floor)
                            : eos.iso_sound_speed * eos.iso_sound_speed * left.density;
        right.pressure = eos.HasEnergy()
                             ? fmax(primitive(IPR, k, j, i), eos.pressure_floor)
                             : eos.iso_sound_speed * eos.iso_sound_speed * right.density;
        InterfaceFlux flux{};
        FirstOrderSolver::Solve(left, right, eos, Direction, flux);
        for (int n = 0; n < components; ++n)
          conserved.flux(Direction + 1, n, k, j, i) = flux.fluid[n];
        for (int scalar = 0; scalar < scalars; ++scalar) {
          const Real fraction = flux.fluid[IDN] >= 0.0
                                    ? primitive(components + scalar, left_k, left_j, left_i)
                                    : primitive(components + scalar, k, j, i);
          conserved.flux(Direction + 1, components + scalar, k, j, i) =
              flux.fluid[IDN] * fmin(fmax(fraction, 0.0), 1.0);
        }
        face_emf(emf_t1, k, j, i) = flux.electric_t1;
        face_emf(emf_t2, k, j, i) = flux.electric_t2;
      });
}

// KHARMA treats the two spherical poles as reflecting coordinate boundaries.
// PANGU stores all fluid components in one field, so Parthenon's generic
// reflecting callback cannot infer that only the x2 momentum/velocity component
// is odd. The face-centred magnetic field is a split topological vector: its
// normal face vanishes at the pole and its ghost faces are antisymmetric, while
// the two tangential components are symmetric. Apply that contract after the
// generic reflecting fill for every MKS problem.
template <bool Inner>
void CorrectMKSPolarBoundary(std::shared_ptr<MeshBlockData<Real>>& data, const bool coarse) {
  if (coarse || std::string(geometry::ConfiguredMetricName()) != "mks")
    return;

  auto block = data->GetBlockPointer();
  const int boundary_index = Inner ? 2 : 3;
  if (block->pmy_mesh->mesh_bc_names[boundary_index] != "reflecting")
    return;

  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto bface = data->Get("mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto ei = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto ej = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto ek = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const int ndim = block->pmy_mesh->ndim;
  const int cell_j_start = Inner ? ej.s : jb.e + 1;
  const int cell_j_end = Inner ? jb.s - 1 : ej.e;
  const int cell_offset = Inner ? 2 * jb.s - 1 : 2 * jb.e + 1;
  const int fluid_components = conserved.GetDim(4);

  block->par_for(
      Inner ? "PANGU MKS inner-pole fluid parity" : "PANGU MKS outer-pole fluid parity", 0,
      fluid_components - 1, ek.s, ek.e, cell_j_start, cell_j_end, ei.s, ei.e,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        const int reflected_j = cell_offset - j;
        const Real sign = n == IM2 ? -1.0 : 1.0;
        conserved(n, k, j, i) = sign * conserved(n, k, reflected_j, i);
        primitive(n, k, j, i) = sign * primitive(n, k, reflected_j, i);
      });
  block->par_for(
      Inner ? "PANGU MKS inner-pole cell-B parity" : "PANGU MKS outer-pole cell-B parity", 0,
      2, ek.s, ek.e, cell_j_start, cell_j_end, ei.s, ei.e,
      KOKKOS_LAMBDA(const int component, const int k, const int j, const int i) {
        const int reflected_j = cell_offset - j;
        const Real sign = component == 1 ? -1.0 : 1.0;
        bcell(component, k, j, i) = sign * bcell(component, k, reflected_j, i);
      });

  // Tangential face components use the cell-centred reflection index in x2.
  block->par_for(
      Inner ? "PANGU MKS inner-pole tangential face-B parity"
            : "PANGU MKS outer-pole tangential face-B parity",
      0, 1, ek.s, ek.e + (ndim >= 3), cell_j_start, cell_j_end, ei.s, ei.e + 1,
      KOKKOS_LAMBDA(const int tangential, const int k, const int j, const int i) {
        const int reflected_j = cell_offset - j;
        if (tangential == 0 && k <= ek.e)
          bface(0, 0, 0, 0, k, j, i) = bface(0, 0, 0, 0, k, reflected_j, i);
        if (tangential == 1 && i <= ei.e)
          bface(2, 0, 0, 0, k, j, i) = bface(2, 0, 0, 0, k, reflected_j, i);
      });

  // A normal face lies directly on the coordinate pole. KHARMA sets that
  // physical face to zero and reflects the remaining normal faces about it.
  const int pole_face = Inner ? jb.s : jb.e + 1;
  block->par_for(
      Inner ? "PANGU MKS zero inner-pole normal face-B"
            : "PANGU MKS zero outer-pole normal face-B",
      ek.s, ek.e, pole_face, pole_face, ei.s, ei.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        bface(1, 0, 0, 0, k, j, i) = 0.0;
      });
  const int normal_j_start = Inner ? ej.s : jb.e + 2;
  const int normal_j_end = Inner ? jb.s - 1 : ej.e + 1;
  block->par_for(
      Inner ? "PANGU MKS inner-pole normal face-B parity"
            : "PANGU MKS outer-pole normal face-B parity",
      ek.s, ek.e, normal_j_start, normal_j_end, ei.s, ei.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int reflected_j = 2 * pole_face - j;
        bface(1, 0, 0, 0, k, j, i) = -bface(1, 0, 0, 0, k, reflected_j, i);
      });
}

} // namespace

bool HasStageSources(const parthenon::Packages_t& packages) {
  return srcterms::HasActiveSources(packages) ||
         packages.Get("mhd")->Param<int>("physics_mode") == 2;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput* pin) {
  auto package = std::make_shared<StateDescriptor>("mhd");
  package->AddParam<driver::StageContribution>(
      std::string(driver::stage_contribution_key), MHDStageContribution());
  package->AddParam<driver::FluidStageExtension>(
      std::string(driver::fluid_stage_extension_key),
      driver::FluidStageExtension{
          {"mhd.cons"}, {"mhd.prim", "mhd.b_cell"}, HasStageSources, UpdateFaceFieldsMeshTask});
  const auto eos_mode = ParseEos(pin->GetOrAddString("mhd", "eos", "ideal"));
  const auto physics = ParsePhysics(pin->GetOrAddString("mhd", "physics", "newtonian"));
  relativity::ValidateCompiledPhysics(physics, "mhd/physics");
  const auto reconstruction = ParseReconstruction(pin->GetOrAddString("mhd", "reconstruct", "plm"));
  const auto riemann_solver = ParseRiemann(pin->GetOrAddString("mhd", "rsolver", "hlle"));
  const Real gamma = pin->GetOrAddReal("mhd", "gamma", 5.0 / 3.0);
  const Real iso_sound_speed = pin->GetOrAddReal("mhd", "iso_sound_speed", 1.0);
  const Real cfl = pin->GetOrAddReal("mhd", "cfl", 0.3);
  const Real density_floor = pin->GetOrAddReal("mhd", "density_floor", 1.0e-12);
  const Real pressure_floor = pin->GetOrAddReal("mhd", "pressure_floor", 1.0e-12);
  const Real entropy_floor = pin->GetOrAddReal(
      "mhd", "entropy_floor", static_cast<Real>(std::numeric_limits<float>::min()));
  const Real gamma_max = pin->GetOrAddReal("mhd", "gamma_max", 1000.0);
  // AthenaK deliberately defaults the relativistic magnetization ceiling to
  // FLT_MAX (rather than Real's maximum), even in double-precision builds.
  const Real sigma_max = pin->GetOrAddReal("mhd", "sigma_max", std::numeric_limits<float>::max());
  const int nscalars = pin->GetOrAddInteger("mhd", "nscalars", 0);
  const bool fofc = pin->GetOrAddBoolean("mhd", "fofc", true);
  const std::string puncture_protection_name =
      pin->GetOrAddString("mhd", "puncture_protection", "none");
  const int puncture_protection = puncture_protection_name == "chi_atmosphere" ? 1 : 0;
  const Real puncture_chi_threshold =
      pin->GetOrAddReal("mhd", "puncture_chi_threshold", 0.1);
  const Real puncture_density =
      pin->GetOrAddReal("mhd", "puncture_density", density_floor);
  const Real puncture_pressure =
      pin->GetOrAddReal("mhd", "puncture_pressure", pressure_floor);
  const Real refine_tolerance = pin->GetOrAddReal("mhd", "refine_tolerance", 0.25);
  const Real derefine_tolerance = pin->GetOrAddReal("mhd", "derefine_tolerance", 0.05);
  const std::string amr_method =
      pin->GetOrAddString("mhd", "amr_method", "density_gradient");
  const Real amr_value_max = pin->GetOrAddReal(
      "mhd", "amr_value_max", std::numeric_limits<float>::max());
  const int amr_interval = pin->GetOrAddInteger("mhd", "amr_interval", 1);
  const bool amr_initial_refinement =
      pin->GetOrAddBoolean("mhd", "amr_initial_refinement", true);
  Real bondi_k_adi = 1.0;
  Real bondi_r_crit = 8.0;
  const auto problem_id = pin->GetString("parthenon/job", "problem_id");
  if (problem_id == "magnetised_bondi_mks") {
    bondi_k_adi = pin->GetOrAddReal("problem", "k_adi", bondi_k_adi);
    bondi_r_crit = pin->GetOrAddReal("problem", "r_crit", bondi_r_crit);
  }
  const int nghost = pin->GetInteger("parthenon/mesh", "nghost");
  const auto& reconstruction_descriptor = reconstruct::Describe(reconstruction);
  const int required_ghost = physics == relativity::HydroMode::newtonian
                                 ? reconstruction_descriptor.ghost_zones
                                 : reconstruction_descriptor.relativistic_ghost_zones;
  PARTHENON_REQUIRE(gamma > 1.0, "mhd/gamma must be greater than one");
  PARTHENON_REQUIRE(iso_sound_speed > 0.0, "mhd/iso_sound_speed must be positive");
  PARTHENON_REQUIRE(cfl > 0.0 && cfl <= 1.0, "mhd/cfl must be in (0, 1]");
  PARTHENON_REQUIRE(nghost >= required_ghost, "mhd reconstruction requires more ghost cells");
  PARTHENON_REQUIRE(nscalars >= 0, "mhd/nscalars must be non-negative");
  const auto& riemann_descriptor = riemann::Describe(riemann_solver);
  const auto riemann_physics = physics == relativity::HydroMode::newtonian
                                   ? riemann::Physics::newtonian_mhd
                                   : riemann::Physics::relativistic_mhd;
  if (riemann_solver != riemann::Solver::none && !riemann_descriptor.Supports(riemann_physics)) {
    PARTHENON_FAIL("mhd/rsolver=" + std::string(riemann_descriptor.name) + " does not support " +
                   (physics == relativity::HydroMode::newtonian ? "Newtonian" : "relativistic") +
                   " MHD");
  }
  if (physics != relativity::HydroMode::newtonian) {
    PARTHENON_REQUIRE(eos_mode == EosMode::ideal, "Relativistic MHD requires the ideal EOS");
    PARTHENON_REQUIRE(nscalars == 0, "Relativistic MHD does not yet support passive scalars");
    PARTHENON_REQUIRE(reconstruction_descriptor.relativistic_mhd,
                      "Selected reconstruction does not support relativistic MHD");
  }
  PARTHENON_REQUIRE(gamma_max > 1.0, "mhd/gamma_max must be greater than one");
  PARTHENON_REQUIRE(sigma_max > 0.0, "mhd/sigma_max must be positive");
  PARTHENON_REQUIRE(entropy_floor > 0.0, "mhd/entropy_floor must be positive");
  PARTHENON_REQUIRE(puncture_protection_name == "none" ||
                        puncture_protection_name == "chi_atmosphere",
                    "mhd/puncture_protection must be none or chi_atmosphere");
  PARTHENON_REQUIRE(puncture_chi_threshold > 0.0,
                    "mhd/puncture_chi_threshold must be positive");
  PARTHENON_REQUIRE(puncture_density >= density_floor &&
                        puncture_pressure >= pressure_floor,
                    "puncture atmosphere must respect the MHD density/pressure floors");
  if (puncture_protection != 0) {
    PARTHENON_REQUIRE(geometry::ConfiguredGeometryIsSynchronized() &&
                          physics == relativity::HydroMode::gr,
                      "chi-atmosphere puncture protection requires GR MHD with MODE=sync");
  }
  PARTHENON_REQUIRE(amr_method == "density_gradient" ||
                        amr_method == "conserved_density_max",
                    "mhd/amr_method must be density_gradient or conserved_density_max");
  PARTHENON_REQUIRE(amr_interval >= 1, "mhd/amr_interval must be at least one cycle");

  package->AddParam<int>("eos_mode", static_cast<int>(eos_mode));
  package->AddParam<int>("physics_mode", static_cast<int>(physics));
  package->AddParam<int>("reconstruction", static_cast<int>(reconstruction));
  package->AddParam<int>("riemann", static_cast<int>(riemann_solver));
  package->AddParam<Real>("gamma", gamma);
  package->AddParam<Real>("iso_sound_speed", iso_sound_speed);
  package->AddParam<Real>("cfl", cfl);
  package->AddParam<Real>("density_floor", density_floor);
  package->AddParam<Real>("pressure_floor", pressure_floor);
  package->AddParam<Real>("entropy_floor", entropy_floor);
  package->AddParam<Real>("gamma_max", gamma_max);
  package->AddParam<Real>("sigma_max", sigma_max);
  package->AddParam<int>("nscalars", nscalars);
  package->AddParam<bool>("fofc", fofc);
  package->AddParam<int>("puncture_protection", puncture_protection);
  package->AddParam<Real>("puncture_chi_threshold", puncture_chi_threshold);
  package->AddParam<Real>("puncture_density", puncture_density);
  package->AddParam<Real>("puncture_pressure", puncture_pressure);
  package->AddParam<Real>("refine_tolerance", refine_tolerance);
  package->AddParam<Real>("derefine_tolerance", derefine_tolerance);
  package->AddParam<int>("amr_method", amr_method == "conserved_density_max" ? 1 : 0);
  package->AddParam<Real>("amr_value_max", amr_value_max);
  package->AddParam<int>("amr_interval", amr_interval);
  package->AddParam<bool>("amr_initial_refinement", amr_initial_refinement);
  package->AddParam<bool>("amr_tag_enabled", amr_initial_refinement,
                          parthenon::Params::Mutability::Mutable);
  package->AddParam<Real>("bondi_k_adi", bondi_k_adi);
  package->AddParam<Real>("bondi_r_crit", bondi_r_crit);

  const int fluid_components =
      eos_mode == EosMode::ideal ? kIdealComponents : kIsothermalComponents;
  const int components = fluid_components + nscalars;
  std::vector<std::string> conserved_labels{"density", "momentum_density_1", "momentum_density_2",
                                            "momentum_density_3"};
  std::vector<std::string> primitive_labels{"density", "velocity_1", "velocity_2", "velocity_3"};
  if (eos_mode == EosMode::ideal) {
    conserved_labels.emplace_back("total_energy_density");
    primitive_labels.emplace_back(
        physics == relativity::HydroMode::newtonian ? "pressure" : "internal_energy_density");
  }
  for (int scalar = 0; scalar < nscalars; ++scalar) {
    conserved_labels.emplace_back("scalar_density_" + std::to_string(scalar));
    primitive_labels.emplace_back("scalar_" + std::to_string(scalar));
  }
  Metadata conserved({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
                      Metadata::FillGhost, Metadata::Restart},
                     std::vector<int>{components}, conserved_labels);
  conserved.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                                  parthenon::refinement_ops::RestrictAverage>();
  package->AddField("mhd.cons", conserved);
  std::vector<parthenon::MetadataFlag> primitive_flags{Metadata::Cell,
                                                       Metadata::Derived};
  if constexpr (geometry::ConfiguredGeometryIsSynchronized())
    primitive_flags.emplace_back(Metadata::OneCopy);
  package->AddField("mhd.prim",
                    Metadata(primitive_flags, std::vector<int>{components}, primitive_labels));

  Metadata magnetic({Metadata::Face, Metadata::Independent, Metadata::WithFluxes,
                     Metadata::FillGhost, Metadata::Restart});
  magnetic.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                                 parthenon::refinement_ops::RestrictAverage,
                                 parthenon::refinement_ops::ProlongateInternalTothAndRoe>();
  package->AddField<BFace>(magnetic);
  std::vector<parthenon::MetadataFlag> bcell_flags{Metadata::Cell,
                                                   Metadata::Derived};
  if constexpr (geometry::ConfiguredGeometryIsSynchronized())
    bcell_flags.emplace_back(Metadata::OneCopy);
  package->AddField("mhd.b_cell",
                    Metadata(bcell_flags, std::vector<int>{3},
                             std::vector<std::string>{"B1", "B2", "B3"}));
  package->AddField("mhd.divb", Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}));
  package->AddField("mhd.fofc", Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}));
  package->AddField(
      "mhd.recovery",
      Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, std::vector<int>{3},
               std::vector<std::string>{"mass_injection", "energy_injection",
                                        "puncture_mask"}));
  package->AddField("mhd.cell_emf", Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                                             std::vector<int>{3}));
  package->AddField("mhd.face_emf", Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                                             std::vector<int>{6}));
  // A communicable edge field gives CT a single EMF on every MeshBlock
  // interface.  This mirrors KHARMA's B_CT.emf ownership and synchronization
  // contract; the automatically generated bnd_flux remains available to the
  // legacy block kernels but is not used as the packed CT update state.
  package->AddField("mhd.edge_emf",
                    Metadata({Metadata::Edge, Metadata::Derived, Metadata::OneCopy,
                              Metadata::FillGhost}));
  if constexpr (pangu::estimator::Selected::face_signal_speeds) {
    const Metadata signal_speed({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                                std::vector<int>{3},
                                std::vector<std::string>{"x1", "x2", "x3"});
    package->AddField("mhd.cmax", signal_speed);
    package->AddField("mhd.cmin", signal_speed);
  }
  if (physics != relativity::HydroMode::newtonian) {
    const Metadata reconstructed_fluid({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                                       std::vector<int>{5});
    const Metadata reconstructed_magnetic({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                                          std::vector<int>{3});
    package->AddField("mhd.recon_left", reconstructed_fluid);
    package->AddField("mhd.recon_right", reconstructed_fluid);
    package->AddField("mhd.recon_b_left", reconstructed_magnetic);
    package->AddField("mhd.recon_b_right", reconstructed_magnetic);
  }

  parthenon::HstVar_list history;
  history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                   ConservedHistoryMesh<IDN>, "mhd_mass"));
  history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                   ConservedHistoryMesh<IM1>, "mhd_momentum_1"));
  history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                   ConservedHistoryMesh<IM2>, "mhd_momentum_2"));
  history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                   ConservedHistoryMesh<IM3>, "mhd_momentum_3"));
  if (eos_mode == EosMode::ideal)
    history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                     ConservedHistoryMesh<IEN>, "mhd_energy"));
  history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::max,
                                                   MaxDivBHistoryMesh, "mhd_max_abs_divb"));
  history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                   FOFCHistoryMesh, "mhd_fofc_cells"));
  if constexpr (geometry::ConfiguredGeometryIsSynchronized()) {
    history.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, RecoveryHistoryMesh<0>,
        "mhd_floor_mass_injection"));
    history.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, RecoveryHistoryMesh<1>,
        "mhd_floor_energy_injection"));
    history.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, RecoveryHistoryMesh<2>,
        "mhd_puncture_mask_cells"));
  }
  package->AddParam<>(parthenon::hist_param_key, history);
  if constexpr (geometry::ConfiguredGeometryIsSynchronized()) {
    package->FillDerivedMesh = nr::SyncGRMHDConservedToPrimitiveMesh;
    package->EstimateTimestepBlock = nr::EstimateSyncGRMHDTimestepBlock;
    package->EstimateTimestepMesh = nr::EstimateSyncGRMHDTimestepMesh;
  } else {
    package->FillDerivedBlock = ConservedToPrimitiveBlock;
    package->EstimateTimestepBlock = EstimateTimestepBlock;
    package->EstimateTimestepMesh = EstimateTimestepMesh;
  }
  package->CheckRefinementBlock = CheckRefinementBlock;
  package->UserBoundaryFunctions[parthenon::BoundaryFace::inner_x2].push_back(
      CorrectMKSPolarBoundary<true>);
  package->UserBoundaryFunctions[parthenon::BoundaryFace::outer_x2].push_back(
      CorrectMKSPolarBoundary<false>);
  return package;
}

TaskStatus CalculateFluxesBlockTask(std::shared_ptr<MeshBlockData<Real>>& data) {
  const auto package = data->GetBlockPointer()->packages.Get("mhd");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian)
    return relativity::CalculateMHDFluxesBlockTask(data);
  const auto reconstruction =
      static_cast<hydro::Reconstruction>(package->Param<int>("reconstruction"));
  const auto solver = static_cast<riemann::Solver>(package->Param<int>("riemann"));
  if (solver == riemann::Solver::none) {
    auto block = data->GetBlockPointer();
    auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
    auto edge = data->Get("bnd_flux::mhd.b_face").data;
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    const int ndim = block->pmy_mesh->ndim;
    const int components =
        (package->Param<int>("eos_mode") == 0 ? kIdealComponents : kIsothermalComponents) +
        package->Param<int>("nscalars");
    block->par_for(
        "PANGU kinematic MHD zero x1 flux", 0, components - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e + 1, KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
          conserved.flux(X1DIR, n, k, j, i) = 0.0;
        });
    if (ndim >= 2)
      block->par_for(
          "PANGU kinematic MHD zero x2 flux", 0, components - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s,
          ib.e, KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
            conserved.flux(X2DIR, n, k, j, i) = 0.0;
          });
    if (ndim >= 3)
      block->par_for(
          "PANGU kinematic MHD zero x3 flux", 0, components - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s,
          ib.e, KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
            conserved.flux(X3DIR, n, k, j, i) = 0.0;
          });
    block->par_for(
        "PANGU kinematic MHD zero edge EMF", 0, 2, kb.s, kb.e + (ndim >= 3), jb.s,
        jb.e + (ndim >= 2), ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
          edge(n, 0, 0, 0, k, j, i) = 0.0;
        });
    return TaskStatus::complete;
  }
  return reconstruct::Visit(reconstruction, [&]<hydro::Reconstruction Method>() {
    return riemann::Visit<riemann::Physics::newtonian_mhd>(solver, [&]<riemann::Solver Solver>() {
      return CalculateFluxesBlockImpl<Method, Solver>(data);
    });
  });
}

TaskStatus BuildCornerEMFBlockTask(std::shared_ptr<MeshBlockData<Real>>& data) {
  return CalculateCornerEMFBlock(data);
}

TaskStatus BuildCornerEMFMeshTask(MeshData<Real>* data) { return CalculateCornerEMFMesh(data); }

void ConservedToPrimitiveBlock(MeshBlockData<Real>* data) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized()) {
    nr::SyncGRMHDConservedToPrimitiveBlock(data);
    return;
  }
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("mhd");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian) {
    relativity::MHDConservedToPrimitiveBlock(data);
    return;
  }
  const auto eos = GetEos(package);
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto divb = data->PackVariables(std::vector<std::string>{"mhd.divb"});
  auto bface = data->Get("mhd.b_face").data;
  const int fluid_components = eos.HasEnergy() ? kIdealComponents : kIsothermalComponents;
  const int scalars = package->Param<int>("nscalars");
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coordinates = block->coords;
  const int ndim = block->pmy_mesh->ndim;
  block->par_for(
      "PANGU MHD conserved to primitive", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real b1 = 0.5 * (bface(0, 0, 0, 0, k, j, i) + bface(0, 0, 0, 0, k, j, i + 1));
        const Real b2 =
            0.5 * (bface(1, 0, 0, 0, k, j, i) + bface(1, 0, 0, 0, k, j + (ndim >= 2), i));
        const Real b3 =
            0.5 * (bface(2, 0, 0, 0, k, j, i) + bface(2, 0, 0, 0, k + (ndim >= 3), j, i));
        bcell(0, k, j, i) = b1;
        bcell(1, k, j, i) = b2;
        bcell(2, k, j, i) = b3;
        Real density = conserved(IDN, k, j, i);
        if (!(density >= eos.density_floor)) {
          density = eos.density_floor;
          conserved(IDN, k, j, i) = density;
          conserved(IM1, k, j, i) = 0.0;
          conserved(IM2, k, j, i) = 0.0;
          conserved(IM3, k, j, i) = 0.0;
        }
        primitive(IDN, k, j, i) = density;
        Real velocity2 = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
          const Real velocity = conserved(IM1 + axis, k, j, i) / density;
          primitive(IV1 + axis, k, j, i) = velocity;
          velocity2 += velocity * velocity;
        }
        if (eos.HasEnergy()) {
          const Real kinetic = 0.5 * density * velocity2;
          const Real magnetic_energy = 0.5 * (b1 * b1 + b2 * b2 + b3 * b3);
          Real pressure = (eos.gamma - 1.0) * (conserved(IEN, k, j, i) - kinetic - magnetic_energy);
          if (!(pressure >= eos.pressure_floor)) {
            pressure = eos.pressure_floor;
            conserved(IEN, k, j, i) = kinetic + magnetic_energy + pressure / (eos.gamma - 1.0);
          }
          primitive(IPR, k, j, i) = pressure;
        }
        for (int scalar = 0; scalar < scalars; ++scalar) {
          primitive(fluid_components + scalar, k, j, i) =
              fmin(fmax(conserved(fluid_components + scalar, k, j, i) / density, 0.0), 1.0);
        }
        Real divergence = (bface(0, 0, 0, 0, k, j, i + 1) - bface(0, 0, 0, 0, k, j, i)) /
                          coordinates.Dxc<X1DIR>(k, j, i);
        if (ndim >= 2)
          divergence += (bface(1, 0, 0, 0, k, j + 1, i) - bface(1, 0, 0, 0, k, j, i)) /
                        coordinates.Dxc<X2DIR>(k, j, i);
        if (ndim >= 3)
          divergence += (bface(2, 0, 0, 0, k + 1, j, i) - bface(2, 0, 0, 0, k, j, i)) /
                        coordinates.Dxc<X3DIR>(k, j, i);
        divb(0, k, j, i) = divergence;
      });
}

void ConservedToPrimitiveMesh(MeshData<Real>* data) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized()) {
    nr::SyncGRMHDConservedToPrimitiveMesh(data);
    return;
  }
  auto* mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("mhd");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian) {
    relativity::MHDConservedToPrimitiveMesh(data);
    return;
  }
  // The performance-critical relativistic path above is fully pack-wide. Keep
  // the Newtonian block implementation as the framework-compatible fallback
  // until its scalar combinations have native MeshData kernels.
  for (int b = 0; b < data->NumBlocks(); ++b)
    ConservedToPrimitiveBlock(data->GetBlockData(b).get());
}

TaskStatus ConservedToPrimitiveMeshTask(MeshData<Real>* data) {
  ConservedToPrimitiveMesh(data);
  return TaskStatus::complete;
}

Real EstimateTimestepBlock(MeshBlockData<Real>* data) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized())
    return nr::EstimateSyncGRMHDTimestepBlock(data);
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("mhd");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian)
    return relativity::EstimateMHDTimestepBlock(data);
  const auto eos = GetEos(package);
  const Real cfl = package->Param<Real>("cfl");
  const auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  const auto magnetic = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  const int ndim = block->pmy_mesh->ndim;
  Real minimum;
  ParReduce(
      "PANGU MHD timestep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
        Primitive state{primitive(IDN, k, j, i),
                        {primitive(IV1, k, j, i), primitive(IV2, k, j, i), primitive(IV3, k, j, i)},
                        eos.HasEnergy()
                            ? primitive(IPR, k, j, i)
                            : eos.iso_sound_speed * eos.iso_sound_speed * primitive(IDN, k, j, i),
                        {magnetic(0, k, j, i), magnetic(1, k, j, i), magnetic(2, k, j, i)}};
        Real cell_dt =
            coordinates.Dxc<X1DIR>(k, j, i) / (fabs(state.velocity[0]) + eos.FastSpeed(state, 0));
        if (ndim >= 2)
          cell_dt = fmin(cell_dt, coordinates.Dxc<X2DIR>(k, j, i) /
                                      (fabs(state.velocity[1]) + eos.FastSpeed(state, 1)));
        if (ndim >= 3)
          cell_dt = fmin(cell_dt, coordinates.Dxc<X3DIR>(k, j, i) /
                                      (fabs(state.velocity[2]) + eos.FastSpeed(state, 2)));
        local = fmin(local, cell_dt);
      },
      Kokkos::Min<Real>(minimum));
  return cfl * minimum;
}

Real EstimateTimestepMesh(MeshData<Real>* data) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized())
    return nr::EstimateSyncGRMHDTimestepMesh(data);
  auto* mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("mhd");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian)
    return relativity::EstimateMHDTimestepMesh(data);
  Real minimum = std::numeric_limits<Real>::max();
  for (int b = 0; b < data->NumBlocks(); ++b)
    minimum = std::min(minimum, EstimateTimestepBlock(data->GetBlockData(b).get()));
  return minimum;
}

AmrTag CheckRefinementBlock(MeshBlockData<Real>* data) {
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("mhd");
  if (!package->Param<bool>("amr_tag_enabled"))
    return AmrTag::same;
  const int method = package->Param<int>("amr_method");
  const Real refine = package->Param<Real>("refine_tolerance");
  const Real derefine = package->Param<Real>("derefine_tolerance");
  const auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  const auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const int ndim = block->pmy_mesh->ndim;
  if (method == 1) {
    const Real threshold = package->Param<Real>("amr_value_max");
    Real maximum;
    ParReduce(
        "PANGU MHD conserved-density min-max AMR", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
          local = fmax(local, conserved(IDN, k, j, i));
        },
        Kokkos::Max<Real>(maximum));
    return maximum > threshold ? AmrTag::refine : AmrTag::derefine;
  }
  Real maximum;
  ParReduce(
      "PANGU MHD density-gradient AMR", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
        const Real center = fmax(fabs(primitive(IDN, k, j, i)), 1.0e-30);
        Real gradient =
            fabs(primitive(IDN, k, j, i + 1) - primitive(IDN, k, j, i - 1)) / (2.0 * center);
        if (ndim >= 2)
          gradient =
              fmax(gradient, fabs(primitive(IDN, k, j + 1, i) - primitive(IDN, k, j - 1, i)) /
                                 (2.0 * center));
        if (ndim >= 3)
          gradient =
              fmax(gradient, fabs(primitive(IDN, k + 1, j, i) - primitive(IDN, k - 1, j, i)) /
                                 (2.0 * center));
        local = fmax(local, gradient);
      },
      Kokkos::Max<Real>(maximum));
  if (maximum > refine)
    return AmrTag::refine;
  if (maximum < derefine)
    return AmrTag::derefine;
  return AmrTag::same;
}

void UpdateAMRSchedule(Mesh* mesh, const parthenon::SimTime& time) {
  const auto& packages = mesh->packages.AllPackages();
  const auto it = packages.find("mhd");
  if (it == packages.end())
    return;
  const auto& package = it->second;
  const int interval = package->Param<int>("amr_interval");
  const bool initial_refinement = package->Param<bool>("amr_initial_refinement");
  const bool enabled = initial_refinement || (time.ncycle + 1 >= interval);
  package->UpdateParam<bool>("amr_tag_enabled", enabled);
}

TaskStatus ComputeFaceCurlMeshTask(MeshData<Real>* current, MeshData<Real>* dudt) {
  for (int b = 0; b < current->NumBlocks(); ++b) {
    auto in = current->GetBlockData(b);
    auto out = dudt->GetBlockData(b);
    auto edge = in->Get("bnd_flux::mhd.b_face").data;
    auto rhs = out->Get("mhd.b_face").data;
    auto block = in->GetBlockPointer();
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    const auto coords = block->coords;
    const int ndim = block->pmy_mesh->ndim;
    block->par_for(
        "PANGU CT B1 RHS", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Real value = 0.0;
          if (ndim >= 2)
            value -= (edge(2, 0, 0, 0, k, j + 1, i) - edge(2, 0, 0, 0, k, j, i)) /
                     coords.Dxc<X2DIR>(k, j, i);
          if (ndim >= 3)
            value += (edge(1, 0, 0, 0, k + 1, j, i) - edge(1, 0, 0, 0, k, j, i)) /
                     coords.Dxc<X3DIR>(k, j, i);
          rhs(0, 0, 0, 0, k, j, i) = value;
        });
    block->par_for(
        "PANGU CT B2 RHS", kb.s, kb.e, jb.s, jb.e + (ndim >= 2), ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Real value = (edge(2, 0, 0, 0, k, j, i + 1) - edge(2, 0, 0, 0, k, j, i)) /
                       coords.Dxc<X1DIR>(k, j, i);
          if (ndim >= 3)
            value -= (edge(0, 0, 0, 0, k + 1, j, i) - edge(0, 0, 0, 0, k, j, i)) /
                     coords.Dxc<X3DIR>(k, j, i);
          rhs(1, 0, 0, 0, k, j, i) = value;
        });
    block->par_for(
        "PANGU CT B3 RHS", kb.s, kb.e + (ndim >= 3), jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Real value = -(edge(1, 0, 0, 0, k, j, i + 1) - edge(1, 0, 0, 0, k, j, i)) /
                       coords.Dxc<X1DIR>(k, j, i);
          if (ndim >= 2)
            value += (edge(0, 0, 0, 0, k, j + 1, i) - edge(0, 0, 0, 0, k, j, i)) /
                     coords.Dxc<X2DIR>(k, j, i);
          rhs(2, 0, 0, 0, k, j, i) = value;
        });
  }
  return TaskStatus::complete;
}

TaskStatus AverageFaceDataMeshTask(MeshData<Real>* current, MeshData<Real>* base,
                                   const Real weight) {
  for (int b = 0; b < current->NumBlocks(); ++b) {
    auto current_block = current->GetBlockData(b);
    auto base_block = base->GetBlockData(b);
    auto x = current_block->Get("mhd.b_face").data;
    auto y = base_block->Get("mhd.b_face").data;
    auto block = current_block->GetBlockPointer();
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    const int ndim = block->pmy_mesh->ndim;
    block->par_for(
        "PANGU average B1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(int k, int j, int i) {
          x(0, 0, 0, 0, k, j, i) =
              weight * x(0, 0, 0, 0, k, j, i) + (1.0 - weight) * y(0, 0, 0, 0, k, j, i);
        });
    block->par_for(
        "PANGU average B2", kb.s, kb.e, jb.s, jb.e + (ndim >= 2), ib.s, ib.e,
        KOKKOS_LAMBDA(int k, int j, int i) {
          x(1, 0, 0, 0, k, j, i) =
              weight * x(1, 0, 0, 0, k, j, i) + (1.0 - weight) * y(1, 0, 0, 0, k, j, i);
        });
    block->par_for(
        "PANGU average B3", kb.s, kb.e + (ndim >= 3), jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(int k, int j, int i) {
          x(2, 0, 0, 0, k, j, i) =
              weight * x(2, 0, 0, 0, k, j, i) + (1.0 - weight) * y(2, 0, 0, 0, k, j, i);
        });
  }
  return TaskStatus::complete;
}

TaskStatus UpdateFaceDataMeshTask(MeshData<Real>* current, MeshData<Real>* dudt, const Real dt,
                                  MeshData<Real>* next) {
  for (int b = 0; b < current->NumBlocks(); ++b) {
    auto current_block = current->GetBlockData(b);
    auto rhs_block = dudt->GetBlockData(b);
    auto next_block = next->GetBlockData(b);
    auto x = current_block->Get("mhd.b_face").data;
    auto rhs = rhs_block->Get("mhd.b_face").data;
    auto z = next_block->Get("mhd.b_face").data;
    auto block = current_block->GetBlockPointer();
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    const int ndim = block->pmy_mesh->ndim;
    block->par_for(
        "PANGU update B1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(int k, int j, int i) {
          z(0, 0, 0, 0, k, j, i) = x(0, 0, 0, 0, k, j, i) + dt * rhs(0, 0, 0, 0, k, j, i);
        });
    block->par_for(
        "PANGU update B2", kb.s, kb.e, jb.s, jb.e + (ndim >= 2), ib.s, ib.e,
        KOKKOS_LAMBDA(int k, int j, int i) {
          z(1, 0, 0, 0, k, j, i) = x(1, 0, 0, 0, k, j, i) + dt * rhs(1, 0, 0, 0, k, j, i);
        });
    block->par_for(
        "PANGU update B3", kb.s, kb.e + (ndim >= 3), jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(int k, int j, int i) {
          z(2, 0, 0, 0, k, j, i) = x(2, 0, 0, 0, k, j, i) + dt * rhs(2, 0, 0, 0, k, j, i);
        });
  }
  return TaskStatus::complete;
}

TaskStatus ApplyFirstOrderFluxCorrectionBlockTask(std::shared_ptr<MeshBlockData<Real>>& data,
                                                  std::shared_ptr<MeshBlockData<Real>>& base,
                                                  const Real gam0, const Real gam1,
                                                  const Real beta_dt) {
  const Real dt = beta_dt;
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("mhd");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian)
    return relativity::ApplyMHDFluxCorrectionBlockTask(data, base, gam0, gam1, beta_dt);
  if (!package->Param<bool>("fofc"))
    return TaskStatus::complete;
  const auto eos = GetEos(package);
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
  const auto base_conserved = base->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto base_face = base->Get("mhd.b_face").data;
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto face_emf = data->PackVariables(std::vector<std::string>{"mhd.face_emf"});
  auto bface = data->Get("mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  const int ndim = block->pmy_mesh->ndim;
  block->par_for(
      "PANGU MHD FOFC candidate", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real candidate[kIdealComponents]{};
        const int components = eos.HasEnergy() ? kIdealComponents : kIsothermalComponents;
        for (int n = 0; n < components; ++n) {
          candidate[n] =
              gam0 * conserved(n, k, j, i) + gam1 * base_conserved(n, k, j, i) -
              dt * (conserved.flux(X1DIR, n, k, j, i + 1) - conserved.flux(X1DIR, n, k, j, i)) /
                  coordinates.Dxc<X1DIR>(k, j, i);
          if (ndim >= 2)
            candidate[n] -=
                dt * (conserved.flux(X2DIR, n, k, j + 1, i) - conserved.flux(X2DIR, n, k, j, i)) /
                coordinates.Dxc<X2DIR>(k, j, i);
          if (ndim >= 3)
            candidate[n] -=
                dt * (conserved.flux(X3DIR, n, k + 1, j, i) - conserved.flux(X3DIR, n, k, j, i)) /
                coordinates.Dxc<X3DIR>(k, j, i);
        }
        Real magnetic[3]{
            gam0 * bcell(0, k, j, i) +
                0.5 * gam1 * (base_face(0, 0, 0, 0, k, j, i) + base_face(0, 0, 0, 0, k, j, i + 1)),
            gam0 * bcell(1, k, j, i) +
                0.5 * gam1 *
                    (base_face(1, 0, 0, 0, k, j, i) + base_face(1, 0, 0, 0, k, j + (ndim >= 2), i)),
            gam0 * bcell(2, k, j, i) + 0.5 * gam1 *
                                           (base_face(2, 0, 0, 0, k, j, i) +
                                            base_face(2, 0, 0, 0, k + (ndim >= 3), j, i))};
        magnetic[1] += dt * (face_emf(E3X1, k, j, i + 1) - face_emf(E3X1, k, j, i)) /
                       coordinates.Dxc<X1DIR>(k, j, i);
        magnetic[2] -= dt * (face_emf(E2X1, k, j, i + 1) - face_emf(E2X1, k, j, i)) /
                       coordinates.Dxc<X1DIR>(k, j, i);
        if (ndim >= 2) {
          magnetic[0] -= dt * (face_emf(E3X2, k, j + 1, i) - face_emf(E3X2, k, j, i)) /
                         coordinates.Dxc<X2DIR>(k, j, i);
          magnetic[2] += dt * (face_emf(E1X2, k, j + 1, i) - face_emf(E1X2, k, j, i)) /
                         coordinates.Dxc<X2DIR>(k, j, i);
        }
        if (ndim >= 3) {
          magnetic[0] += dt * (face_emf(E2X3, k + 1, j, i) - face_emf(E2X3, k, j, i)) /
                         coordinates.Dxc<X3DIR>(k, j, i);
          magnetic[1] -= dt * (face_emf(E1X3, k + 1, j, i) - face_emf(E1X3, k, j, i)) /
                         coordinates.Dxc<X3DIR>(k, j, i);
        }
        const Real density = candidate[IDN];
        bool invalid = !(density >= eos.density_floor);
        if (eos.HasEnergy() && !invalid) {
          const Real kinetic = 0.5 *
                               (candidate[IM1] * candidate[IM1] + candidate[IM2] * candidate[IM2] +
                                candidate[IM3] * candidate[IM3]) /
                               density;
          const Real magnetic_energy =
              0.5 *
              (magnetic[0] * magnetic[0] + magnetic[1] * magnetic[1] + magnetic[2] * magnetic[2]);
          const Real pressure = (eos.gamma - 1.0) * (candidate[IEN] - kinetic - magnetic_energy);
          invalid = !(pressure >= eos.pressure_floor);
        }
        flags(0, k, j, i) = invalid ? 1.0 : 0.0;
      });
  ReplaceMHDFlaggedFacesBlock<0>(block, primitive, bcell, flags, conserved, bface, face_emf, eos,
                                 ib, jb, kb);
  if (ndim >= 2)
    ReplaceMHDFlaggedFacesBlock<1>(block, primitive, bcell, flags, conserved, bface, face_emf, eos,
                                   ib, jb, kb);
  if (ndim >= 3)
    ReplaceMHDFlaggedFacesBlock<2>(block, primitive, bcell, flags, conserved, bface, face_emf, eos,
                                   ib, jb, kb);
  return CalculateCornerEMFBlock(data);
}

TaskStatus ApplySourcesBlockTask(std::shared_ptr<MeshBlockData<Real>>& data, const Real dt) {
  relativity::ApplyMHDSourcesBlockTask(data, dt);
  srcterms::ApplyMHDBlockTask(data, dt);
  return TaskStatus::complete;
}

TaskStatus ApplySourcesMeshTask(MeshData<Real>* data, const Real dt) {
  return relativity::ApplyMHDSourcesMeshTask(data, dt);
}

} // namespace pangu::mhd
