#include "hydro/hydro_package.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "driver/contributor.h"
#include "driver/stage_extension.h"
#include "eos/newtonian_eos.h"
#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "hydro/task_contribution.h"
#include "interface/metadata.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "z4c/coupling/sync_grhd.h"
#include "pangu.h"
#include "prolong_restrict/prolong_restrict.hpp"
#include "reconstruct/hydro_reconstruction.h"
#include "relativity/physics_assembly.h"
#include "relativity/relativistic_hydro.h"
#include "riemann/registry.h"
#include "srcterms/source_terms.h"
#include "utils/error_checking.hpp"

namespace pangu::hydro {
using namespace parthenon::package::prelude;

// First-order flux correction replaces flagged faces with the registered fallback solver.
using FirstOrderSolver =
    riemann::Implementation<riemann::Physics::newtonian_hydro, riemann::first_order_fallback>;

namespace {

EosMode ParseEos(const std::string& name) {
  if (name == "ideal" || name == "adiabatic")
    return EosMode::ideal;
  if (name == "isothermal")
    return EosMode::isothermal;
  PARTHENON_FAIL("hydro/eos must be ideal, adiabatic, or isothermal");
}

Reconstruction ParseReconstruction(const std::string& name) {
  const auto method = reconstruct::Parse(name);
  if (!method)
    PARTHENON_FAIL("hydro/reconstruct must be dc, plm, ppm, ppm4, ppmc, or wenoz");
  return *method;
}

riemann::Solver ParseRiemann(const std::string& name) {
  const auto solver = riemann::Parse(name);
  if (!solver)
    PARTHENON_FAIL("hydro/rsolver must name a registered Riemann solver, none, or advect");
  return *solver;
}

relativity::HydroMode ParsePhysics(const std::string& name) {
  if (name == "newtonian")
    return relativity::HydroMode::newtonian;
  if (name == "sr" || name == "special_relativity")
    return relativity::HydroMode::sr;
  if (name == "gr" || name == "fixed_gr")
    return relativity::HydroMode::gr;
  PARTHENON_FAIL("hydro/physics must be newtonian, sr, special_relativity, gr, or fixed_gr");
}

eos::NewtonianEOS GetEos(const std::shared_ptr<StateDescriptor>& package) {
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

template <Reconstruction Method, int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION void ReconstructPrimitive(const Pack& primitive, const int k, const int j,
                                                 const int i, const eos::NewtonianEOS& eos,
                                                 Primitive& left, Primitive& right) {
  Real left_values[kIdealComponents]{};
  Real right_values[kIdealComponents]{};
  const int components = eos.HasEnergy() ? kIdealComponents : kIsothermalComponents;
  for (int n = 0; n < components; ++n) {
    reconstruct::ReconstructFaceStencil<Method>(
        [&](const int offset) { return ReadOffset<Direction>(primitive, n, k, j, i, offset); },
        left_values[n], right_values[n]);
  }
  left.density = fmax(left_values[IDN], eos.density_floor);
  right.density = fmax(right_values[IDN], eos.density_floor);
  for (int axis = 0; axis < 3; ++axis) {
    left.velocity[axis] = left_values[IV1 + axis];
    right.velocity[axis] = right_values[IV1 + axis];
  }
  if (eos.HasEnergy()) {
    left.pressure = fmax(left_values[IPR], eos.pressure_floor);
    right.pressure = fmax(right_values[IPR], eos.pressure_floor);
  } else {
    left.pressure = eos.iso_sound_speed * eos.iso_sound_speed * left.density;
    right.pressure = eos.iso_sound_speed * eos.iso_sound_speed * right.density;
  }
}

template <Reconstruction Method, int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION void ReconstructComponent(const Pack& primitive, const int component,
                                                 const int k, const int j, const int i, Real& left,
                                                 Real& right) {
  reconstruct::ReconstructFaceStencil<Method>(
      [&](const int offset) {
        return ReadOffset<Direction>(primitive, component, k, j, i, offset);
      },
      left, right);
}

template <int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION Real ReadMeshOffset(const Pack& pack, const int block, const int component,
                                           const int k, const int j, const int i,
                                           const int offset) {
  if constexpr (Direction == 0)
    return pack(block, component, k, j, i + offset);
  if constexpr (Direction == 1)
    return pack(block, component, k, j + offset, i);
  return pack(block, component, k + offset, j, i);
}

template <Reconstruction Method, int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION void ReconstructPrimitiveMesh(const Pack& primitive, const int block,
                                                     const int k, const int j, const int i,
                                                     const eos::NewtonianEOS& eos, Primitive& left,
                                                     Primitive& right) {
  Real left_values[kIdealComponents]{};
  Real right_values[kIdealComponents]{};
  const int components = eos.HasEnergy() ? kIdealComponents : kIsothermalComponents;
  for (int n = 0; n < components; ++n) {
    reconstruct::ReconstructFaceStencil<Method>(
        [&](const int offset) {
          return ReadMeshOffset<Direction>(primitive, block, n, k, j, i, offset);
        },
        left_values[n], right_values[n]);
  }
  left.density = fmax(left_values[IDN], eos.density_floor);
  right.density = fmax(right_values[IDN], eos.density_floor);
  for (int axis = 0; axis < 3; ++axis) {
    left.velocity[axis] = left_values[IV1 + axis];
    right.velocity[axis] = right_values[IV1 + axis];
  }
  if (eos.HasEnergy()) {
    left.pressure = fmax(left_values[IPR], eos.pressure_floor);
    right.pressure = fmax(right_values[IPR], eos.pressure_floor);
  } else {
    left.pressure = eos.iso_sound_speed * eos.iso_sound_speed * left.density;
    right.pressure = eos.iso_sound_speed * eos.iso_sound_speed * right.density;
  }
}

template <Reconstruction Method, int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION void ReconstructComponentMesh(const Pack& primitive, const int block,
                                                     const int component, const int k, const int j,
                                                     const int i, Real& left, Real& right) {
  reconstruct::ReconstructFaceStencil<Method>(
      [&](const int offset) {
        return ReadMeshOffset<Direction>(primitive, block, component, k, j, i, offset);
      },
      left, right);
}

template <Reconstruction Method, riemann::Solver Solver, int Direction>
void CalculateDirectionalFluxesMesh(MeshData<Real>* data, const eos::NewtonianEOS& eos,
                                    const int scalars) {
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"hydro.cons"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int components = eos.HasEnergy() ? kIdealComponents : kIsothermalComponents;
  const int nblocks = data->NumBlocks();
  const auto run = [&](const char* label, const IndexRange& krange, const IndexRange& jrange,
                       const IndexRange& irange) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, label, parthenon::DevExecSpace(), 0, nblocks - 1, krange.s, krange.e,
        jrange.s, jrange.e, irange.s, irange.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          Primitive left{}, right{};
          ReconstructPrimitiveMesh<Method, Direction>(primitive, b, k, j, i, eos, left, right);
          Real flux[kIdealComponents]{};
          riemann::Implementation<riemann::Physics::newtonian_hydro, Solver>::Solve(
              left, right, eos, Direction, flux);
          const auto block_conserved = conserved(b);
          for (int n = 0; n < components; ++n)
            block_conserved.flux(Direction + 1, n, k, j, i) = flux[n];
          for (int scalar = 0; scalar < scalars; ++scalar) {
            Real scalar_left, scalar_right;
            ReconstructComponentMesh<Method, Direction>(primitive, b, components + scalar, k, j, i,
                                                        scalar_left, scalar_right);
            const Real fraction = flux[IDN] >= 0.0 ? scalar_left : scalar_right;
            block_conserved.flux(Direction + 1, components + scalar, k, j, i) =
                flux[IDN] * fmin(fmax(fraction, 0.0), 1.0);
          }
        });
  };

  if constexpr (Direction == 0)
    run("PANGU hydro packed x1 flux", kb, jb, {ib.s, ib.e + 1});
  else if constexpr (Direction == 1)
    run("PANGU hydro packed x2 flux", kb, {jb.s, jb.e + 1}, ib);
  else
    run("PANGU hydro packed x3 flux", {kb.s, kb.e + 1}, jb, ib);
}

template <Reconstruction Method, riemann::Solver Solver>
TaskStatus CalculateFluxesMeshImpl(MeshData<Real>* data) {
  const auto package = data->GetParentPointer()->packages.Get("hydro");
  const auto eos = GetEos(package);
  const int scalars = package->Param<int>("nscalars");
  CalculateDirectionalFluxesMesh<Method, Solver, 0>(data, eos, scalars);
  if (data->GetParentPointer()->ndim >= 2)
    CalculateDirectionalFluxesMesh<Method, Solver, 1>(data, eos, scalars);
  if (data->GetParentPointer()->ndim >= 3)
    CalculateDirectionalFluxesMesh<Method, Solver, 2>(data, eos, scalars);
  return TaskStatus::complete;
}

template <Reconstruction Method, riemann::Solver Solver, int Direction>
void CalculateDirectionalFluxesBlock(const std::shared_ptr<MeshBlockData<Real>>& data,
                                     const eos::NewtonianEOS& eos) {
  auto block = data->GetBlockPointer();
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"hydro.cons"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const int components = eos.HasEnergy() ? kIdealComponents : kIsothermalComponents;
  const int scalars = block->packages.Get("hydro")->Param<int>("nscalars");

  if constexpr (Direction == 0) {
    block->par_for(
        "PANGU hydro x1 flux", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Primitive left{}, right{};
          ReconstructPrimitive<Method, Direction>(primitive, k, j, i, eos, left, right);
          Real flux[kIdealComponents]{};
          riemann::Implementation<riemann::Physics::newtonian_hydro, Solver>::Solve(
              left, right, eos, Direction, flux);
          for (int n = 0; n < components; ++n)
            conserved.flux(X1DIR, n, k, j, i) = flux[n];
          for (int scalar = 0; scalar < scalars; ++scalar) {
            Real scalar_left, scalar_right;
            ReconstructComponent<Method, Direction>(primitive, components + scalar, k, j, i,
                                                    scalar_left, scalar_right);
            const Real fraction = flux[IDN] >= 0.0 ? scalar_left : scalar_right;
            conserved.flux(X1DIR, components + scalar, k, j, i) =
                flux[IDN] * fmin(fmax(fraction, 0.0), 1.0);
          }
        });
  } else if constexpr (Direction == 1) {
    block->par_for(
        "PANGU hydro x2 flux", kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Primitive left{}, right{};
          ReconstructPrimitive<Method, Direction>(primitive, k, j, i, eos, left, right);
          Real flux[kIdealComponents]{};
          riemann::Implementation<riemann::Physics::newtonian_hydro, Solver>::Solve(
              left, right, eos, Direction, flux);
          for (int n = 0; n < components; ++n)
            conserved.flux(X2DIR, n, k, j, i) = flux[n];
          for (int scalar = 0; scalar < scalars; ++scalar) {
            Real scalar_left, scalar_right;
            ReconstructComponent<Method, Direction>(primitive, components + scalar, k, j, i,
                                                    scalar_left, scalar_right);
            const Real fraction = flux[IDN] >= 0.0 ? scalar_left : scalar_right;
            conserved.flux(X2DIR, components + scalar, k, j, i) =
                flux[IDN] * fmin(fmax(fraction, 0.0), 1.0);
          }
        });
  } else {
    block->par_for(
        "PANGU hydro x3 flux", kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Primitive left{}, right{};
          ReconstructPrimitive<Method, Direction>(primitive, k, j, i, eos, left, right);
          Real flux[kIdealComponents]{};
          riemann::Implementation<riemann::Physics::newtonian_hydro, Solver>::Solve(
              left, right, eos, Direction, flux);
          for (int n = 0; n < components; ++n)
            conserved.flux(X3DIR, n, k, j, i) = flux[n];
          for (int scalar = 0; scalar < scalars; ++scalar) {
            Real scalar_left, scalar_right;
            ReconstructComponent<Method, Direction>(primitive, components + scalar, k, j, i,
                                                    scalar_left, scalar_right);
            const Real fraction = flux[IDN] >= 0.0 ? scalar_left : scalar_right;
            conserved.flux(X3DIR, components + scalar, k, j, i) =
                flux[IDN] * fmin(fmax(fraction, 0.0), 1.0);
          }
        });
  }
}

template <Reconstruction Method, riemann::Solver Solver>
TaskStatus CalculateFluxesBlockImpl(std::shared_ptr<MeshBlockData<Real>>& data) {
  const auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("hydro");
  const auto eos = GetEos(package);
  CalculateDirectionalFluxesBlock<Method, Solver, 0>(data, eos);
  if (block->pmy_mesh->ndim >= 2)
    CalculateDirectionalFluxesBlock<Method, Solver, 1>(data, eos);
  if (block->pmy_mesh->ndim >= 3)
    CalculateDirectionalFluxesBlock<Method, Solver, 2>(data, eos);
  return TaskStatus::complete;
}

template <int Component> Real ConservedHistoryMesh(MeshData<Real>* data) {
  const auto block = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  Real total = 0.0;
  ParReduce(
      "PANGU hydro conserved history", 0, conserved.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        local += conserved(b, Component, k, j, i) * conserved.GetCoords(b).CellVolume(k, j, i);
      },
      Kokkos::Sum<Real>(total));
  return total;
}

Real FOFCHistoryMesh(MeshData<Real>* data) {
  const auto block = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto flags = data->PackVariables(std::vector<std::string>{"hydro.fofc"});
  Real total = 0.0;
  ParReduce(
      "PANGU hydro FOFC history", 0, flags.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        local += flags(b, 0, k, j, i);
      },
      Kokkos::Sum<Real>(total));
  return total;
}

Real DensityMaximumHistoryMesh(MeshData<Real>* data) {
  const auto block = data->GetBlockData(0)->GetBlockPointer();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  Real maximum = 0.0;
  ParReduce(
      "PANGU hydro density maximum history", 0, primitive.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        local = fmax(local, primitive(b, IDN, k, j, i));
      },
      Kokkos::Max<Real>(maximum));
  return maximum;
}

template <int Direction, typename PrimitivePack, typename ConservedPack>
void ReplaceFlaggedFacesBlock(MeshBlock* block, const PrimitivePack& primitive,
                              const PrimitivePack& flags, ConservedPack& conserved,
                              const eos::NewtonianEOS& eos, const IndexRange& ib,
                              const IndexRange& jb, const IndexRange& kb) {
  const int components = eos.HasEnergy() ? kIdealComponents : kIsothermalComponents;
  const int scalars = block->packages.Get("hydro")->Param<int>("nscalars");
  if constexpr (Direction == 0) {
    block->par_for(
        "PANGU hydro FOFC x1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const bool left_bad = i > ib.s && flags(0, k, j, i - 1) > 0.5;
          const bool right_bad = i <= ib.e && flags(0, k, j, i) > 0.5;
          if (!(left_bad || right_bad))
            return;
          Primitive left{fmax(primitive(IDN, k, j, i - 1), eos.density_floor),
                         {primitive(IV1, k, j, i - 1), primitive(IV2, k, j, i - 1),
                          primitive(IV3, k, j, i - 1)},
                         0.0};
          Primitive right{
              fmax(primitive(IDN, k, j, i), eos.density_floor),
              {primitive(IV1, k, j, i), primitive(IV2, k, j, i), primitive(IV3, k, j, i)},
              0.0};
          left.pressure = eos.HasEnergy()
                              ? fmax(primitive(IPR, k, j, i - 1), eos.pressure_floor)
                              : eos.iso_sound_speed * eos.iso_sound_speed * left.density;
          right.pressure = eos.HasEnergy()
                               ? fmax(primitive(IPR, k, j, i), eos.pressure_floor)
                               : eos.iso_sound_speed * eos.iso_sound_speed * right.density;
          Real flux[kIdealComponents]{};
          FirstOrderSolver::Solve(left, right, eos, Direction, flux);
          for (int n = 0; n < components; ++n)
            conserved.flux(X1DIR, n, k, j, i) = flux[n];
          for (int scalar = 0; scalar < scalars; ++scalar) {
            const Real fraction = flux[IDN] >= 0.0 ? primitive(components + scalar, k, j, i - 1)
                                                   : primitive(components + scalar, k, j, i);
            conserved.flux(X1DIR, components + scalar, k, j, i) =
                flux[IDN] * fmin(fmax(fraction, 0.0), 1.0);
          }
        });
  } else if constexpr (Direction == 1) {
    block->par_for(
        "PANGU hydro FOFC x2", kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const bool left_bad = j > jb.s && flags(0, k, j - 1, i) > 0.5;
          const bool right_bad = j <= jb.e && flags(0, k, j, i) > 0.5;
          if (!(left_bad || right_bad))
            return;
          Primitive left{fmax(primitive(IDN, k, j - 1, i), eos.density_floor),
                         {primitive(IV1, k, j - 1, i), primitive(IV2, k, j - 1, i),
                          primitive(IV3, k, j - 1, i)},
                         0.0};
          Primitive right{
              fmax(primitive(IDN, k, j, i), eos.density_floor),
              {primitive(IV1, k, j, i), primitive(IV2, k, j, i), primitive(IV3, k, j, i)},
              0.0};
          left.pressure = eos.HasEnergy()
                              ? fmax(primitive(IPR, k, j - 1, i), eos.pressure_floor)
                              : eos.iso_sound_speed * eos.iso_sound_speed * left.density;
          right.pressure = eos.HasEnergy()
                               ? fmax(primitive(IPR, k, j, i), eos.pressure_floor)
                               : eos.iso_sound_speed * eos.iso_sound_speed * right.density;
          Real flux[kIdealComponents]{};
          FirstOrderSolver::Solve(left, right, eos, Direction, flux);
          for (int n = 0; n < components; ++n)
            conserved.flux(X2DIR, n, k, j, i) = flux[n];
          for (int scalar = 0; scalar < scalars; ++scalar) {
            const Real fraction = flux[IDN] >= 0.0 ? primitive(components + scalar, k, j - 1, i)
                                                   : primitive(components + scalar, k, j, i);
            conserved.flux(X2DIR, components + scalar, k, j, i) =
                flux[IDN] * fmin(fmax(fraction, 0.0), 1.0);
          }
        });
  } else {
    block->par_for(
        "PANGU hydro FOFC x3", kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const bool left_bad = k > kb.s && flags(0, k - 1, j, i) > 0.5;
          const bool right_bad = k <= kb.e && flags(0, k, j, i) > 0.5;
          if (!(left_bad || right_bad))
            return;
          Primitive left{fmax(primitive(IDN, k - 1, j, i), eos.density_floor),
                         {primitive(IV1, k - 1, j, i), primitive(IV2, k - 1, j, i),
                          primitive(IV3, k - 1, j, i)},
                         0.0};
          Primitive right{
              fmax(primitive(IDN, k, j, i), eos.density_floor),
              {primitive(IV1, k, j, i), primitive(IV2, k, j, i), primitive(IV3, k, j, i)},
              0.0};
          left.pressure = eos.HasEnergy()
                              ? fmax(primitive(IPR, k - 1, j, i), eos.pressure_floor)
                              : eos.iso_sound_speed * eos.iso_sound_speed * left.density;
          right.pressure = eos.HasEnergy()
                               ? fmax(primitive(IPR, k, j, i), eos.pressure_floor)
                               : eos.iso_sound_speed * eos.iso_sound_speed * right.density;
          Real flux[kIdealComponents]{};
          FirstOrderSolver::Solve(left, right, eos, Direction, flux);
          for (int n = 0; n < components; ++n)
            conserved.flux(X3DIR, n, k, j, i) = flux[n];
          for (int scalar = 0; scalar < scalars; ++scalar) {
            const Real fraction = flux[IDN] >= 0.0 ? primitive(components + scalar, k - 1, j, i)
                                                   : primitive(components + scalar, k, j, i);
            conserved.flux(X3DIR, components + scalar, k, j, i) =
                flux[IDN] * fmin(fmax(fraction, 0.0), 1.0);
          }
        });
  }
}

} // namespace

bool HasStageSources(const parthenon::Packages_t& packages) {
  const auto hydro = packages.Get("hydro");
  return srcterms::HasActiveSources(packages) || hydro->Param<Real>("accel1") != 0.0 ||
         hydro->Param<Real>("accel2") != 0.0 || hydro->Param<Real>("accel3") != 0.0 ||
         hydro->Param<int>("physics_mode") == 2;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput* pin) {
  auto package = std::make_shared<StateDescriptor>("hydro");
  package->AddParam<driver::StageContribution>(
      std::string(driver::stage_contribution_key), HydroStageContribution());
  package->AddParam<driver::FluidStageExtension>(
      std::string(driver::fluid_stage_extension_key),
      driver::FluidStageExtension{{"hydro.cons"}, {"hydro.prim"}, HasStageSources, nullptr});
  const auto eos_mode = ParseEos(pin->GetOrAddString("hydro", "eos", "ideal"));
  const auto physics = ParsePhysics(pin->GetOrAddString("hydro", "physics", "newtonian"));
  relativity::ValidateCompiledPhysics(physics, "hydro/physics");
  const auto reconstruction =
      ParseReconstruction(pin->GetOrAddString("hydro", "reconstruct", "plm"));
  const auto riemann_solver = ParseRiemann(pin->GetOrAddString("hydro", "rsolver", "hllc"));
  const auto gamma = pin->GetOrAddReal("hydro", "gamma", 1.4);
  const auto iso_sound_speed = pin->GetOrAddReal("hydro", "iso_sound_speed", 1.0);
  const auto cfl = pin->GetOrAddReal("hydro", "cfl", 0.4);
  const auto density_floor = pin->GetOrAddReal("hydro", "density_floor", 1.0e-12);
  const auto pressure_floor = pin->GetOrAddReal("hydro", "pressure_floor", 1.0e-12);
  const auto gamma_max = pin->GetOrAddReal("hydro", "gamma_max", 1000.0);
  const auto nscalars = pin->GetOrAddInteger("hydro", "nscalars", 0);
  const auto fofc = pin->GetOrAddBoolean("hydro", "fofc", true);
  const auto accel1 = pin->GetOrAddReal("hydro", "accel1", 0.0);
  const auto accel2 = pin->GetOrAddReal("hydro", "accel2", 0.0);
  const auto accel3 = pin->GetOrAddReal("hydro", "accel3", 0.0);
  const auto refine_tolerance = pin->GetOrAddReal("hydro", "refine_tolerance", 0.2);
  const auto derefine_tolerance = pin->GetOrAddReal("hydro", "derefine_tolerance", 0.05);
  Real bondi_k_adi = 1.0;
  Real bondi_r_crit = 8.0;
  const auto problem_id = pin->GetString("parthenon/job", "problem_id");
  if (problem_id == "bondi_cks" || problem_id == "bondi_mks") {
    bondi_k_adi = pin->GetOrAddReal("problem", "k_adi", bondi_k_adi);
    bondi_r_crit = pin->GetOrAddReal("problem", "r_crit", bondi_r_crit);
  }
  const auto nghost = pin->GetInteger("parthenon/mesh", "nghost");
  const auto& reconstruction_descriptor = reconstruct::Describe(reconstruction);
  const int required_ghost = physics == relativity::HydroMode::newtonian
                                 ? reconstruction_descriptor.ghost_zones
                                 : reconstruction_descriptor.relativistic_ghost_zones;

  if (gamma <= 1.0)
    PARTHENON_FAIL("hydro/gamma must be greater than one");
  if (iso_sound_speed <= 0.0)
    PARTHENON_FAIL("hydro/iso_sound_speed must be positive");
  if (cfl <= 0.0 || cfl > 1.0)
    PARTHENON_FAIL("hydro/cfl must be in (0, 1]");
  if (density_floor <= 0.0 || pressure_floor <= 0.0) {
    PARTHENON_FAIL("Hydro floors must be positive");
  }
  if (nscalars < 0)
    PARTHENON_FAIL("hydro/nscalars must be non-negative");
  if (nghost < required_ghost)
    PARTHENON_FAIL("Not enough ghost zones for reconstruction");
  const auto& riemann_descriptor = riemann::Describe(riemann_solver);
  const auto riemann_physics = physics == relativity::HydroMode::newtonian
                                   ? riemann::Physics::newtonian_hydro
                                   : riemann::Physics::relativistic_hydro;
  if (riemann_solver != riemann::Solver::none && !riemann_descriptor.Supports(riemann_physics)) {
    PARTHENON_FAIL("hydro/rsolver=" + std::string(riemann_descriptor.name) + " does not support " +
                   (physics == relativity::HydroMode::newtonian ? "Newtonian" : "relativistic") +
                   " Hydro");
  }
  if (eos_mode == EosMode::isothermal && !riemann_descriptor.isothermal) {
    std::string name(riemann_descriptor.name);
    std::transform(name.begin(), name.end(), name.begin(), ::toupper);
    PARTHENON_FAIL("Isothermal Hydro does not support the " + name + " Riemann solver");
  }
  if (physics != relativity::HydroMode::newtonian && eos_mode != EosMode::ideal)
    PARTHENON_FAIL("Relativistic Hydro currently requires the ideal EOS");
  if (physics != relativity::HydroMode::newtonian &&
      !reconstruction_descriptor.relativistic_hydro)
    PARTHENON_FAIL("Selected reconstruction does not support relativistic Hydro");
  PARTHENON_REQUIRE(gamma_max > 1.0, "hydro/gamma_max must be greater than one");
  if constexpr (geometry::ConfiguredGeometryIsSynchronized()) {
    PARTHENON_REQUIRE(physics == relativity::HydroMode::gr,
                      "sync Geometry requires hydro/physics=gr");
    PARTHENON_REQUIRE(!fofc,
                      "sync GRHD first-order flux correction is not enabled until SYNC-2");
  }

  package->AddParam<int>("eos_mode", static_cast<int>(eos_mode));
  package->AddParam<int>("physics_mode", static_cast<int>(physics));
  package->AddParam<int>("reconstruction", static_cast<int>(reconstruction));
  package->AddParam<int>("riemann", static_cast<int>(riemann_solver));
  package->AddParam<Real>("gamma", gamma);
  package->AddParam<Real>("iso_sound_speed", iso_sound_speed);
  package->AddParam<Real>("cfl", cfl);
  package->AddParam<Real>("density_floor", density_floor);
  package->AddParam<Real>("pressure_floor", pressure_floor);
  package->AddParam<Real>("gamma_max", gamma_max);
  package->AddParam<int>("nscalars", nscalars);
  package->AddParam<bool>("fofc", fofc);
  package->AddParam<Real>("accel1", accel1);
  package->AddParam<Real>("accel2", accel2);
  package->AddParam<Real>("accel3", accel3);
  package->AddParam<Real>("refine_tolerance", refine_tolerance);
  package->AddParam<Real>("derefine_tolerance", derefine_tolerance);
  package->AddParam<Real>("bondi_k_adi", bondi_k_adi);
  package->AddParam<Real>("bondi_r_crit", bondi_r_crit);

  const int hydro_components =
      eos_mode == EosMode::ideal ? kIdealComponents : kIsothermalComponents;
  const int components = hydro_components + nscalars;
  std::vector<std::string> conserved_labels{"density", "momentum_density_1", "momentum_density_2",
                                            "momentum_density_3"};
  std::vector<std::string> primitive_labels{"density", "velocity_1", "velocity_2", "velocity_3"};
  if (eos_mode == EosMode::ideal) {
    conserved_labels.emplace_back("total_energy_density");
    primitive_labels.emplace_back("pressure");
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
  package->AddField("hydro.cons", conserved);
  std::vector<parthenon::MetadataFlag> primitive_flags{Metadata::Cell,
                                                       Metadata::Derived};
  if constexpr (geometry::ConfiguredGeometryIsSynchronized())
    primitive_flags.emplace_back(Metadata::OneCopy);
  package->AddField("hydro.prim",
                    Metadata(primitive_flags, std::vector<int>{components}, primitive_labels));
  package->AddField("hydro.fofc", Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}));
  if constexpr (geometry::ConfiguredMetricSupportsExcision()) {
    package->AddField("hydro.excision",
                      Metadata({Metadata::Cell, Metadata::Independent, Metadata::OneCopy},
                               std::vector<int>{2}, std::vector<std::string>{"floor", "flux"}));
  }

  parthenon::HstVar_list history;
  history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                   ConservedHistoryMesh<IDN>, "hydro_mass"));
  history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                   ConservedHistoryMesh<IM1>, "hydro_momentum_1"));
  history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                   ConservedHistoryMesh<IM2>, "hydro_momentum_2"));
  history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                   ConservedHistoryMesh<IM3>, "hydro_momentum_3"));
  if (eos_mode == EosMode::ideal) {
    history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                     ConservedHistoryMesh<IEN>, "hydro_energy"));
  }
  history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                   FOFCHistoryMesh, "hydro_fofc_cells"));
  if (problem_id == "sync_grhd_tov") {
    history.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::max,
                                                     DensityMaximumHistoryMesh,
                                                     "hydro_density_max"));
  }
  package->AddParam<>(parthenon::hist_param_key, history);
  package->FillDerivedBlock = ConservedToPrimitiveBlock;
  package->FillDerivedMesh = ConservedToPrimitiveMesh;
  if constexpr (geometry::ConfiguredMetricSupportsExcision())
    package->PostInitializationMesh = relativity::InitializeExcisionMasksMesh;
  package->EstimateTimestepBlock = EstimateTimestepBlock;
  package->EstimateTimestepMesh = EstimateTimestepMesh;
  package->CheckRefinementBlock = CheckRefinementBlock;
  return package;
}

TaskStatus CalculateFluxesBlockTask(std::shared_ptr<MeshBlockData<Real>>& data) {
  const auto package = data->GetBlockPointer()->packages.Get("hydro");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian)
    return relativity::CalculateFluxesBlockTask(data);
  const auto reconstruction = static_cast<Reconstruction>(package->Param<int>("reconstruction"));
  const auto solver = static_cast<riemann::Solver>(package->Param<int>("riemann"));
  if (solver == riemann::Solver::none) {
    auto block = data->GetBlockPointer();
    auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"hydro.cons"});
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    const int components =
        (package->Param<int>("eos_mode") == 0 ? kIdealComponents : kIsothermalComponents) +
        package->Param<int>("nscalars");
    const int ndim = block->pmy_mesh->ndim;
    block->par_for(
        "PANGU kinematic zero x1 flux", 0, components - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
          conserved.flux(X1DIR, n, k, j, i) = 0.0;
        });
    if (ndim >= 2)
      block->par_for(
          "PANGU kinematic zero x2 flux", 0, components - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
          KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
            conserved.flux(X2DIR, n, k, j, i) = 0.0;
          });
    if (ndim >= 3)
      block->par_for(
          "PANGU kinematic zero x3 flux", 0, components - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
            conserved.flux(X3DIR, n, k, j, i) = 0.0;
          });
    return TaskStatus::complete;
  }
  return reconstruct::Visit(reconstruction, [&]<Reconstruction Method>() {
    return riemann::Visit<riemann::Physics::newtonian_hydro>(solver, [&]<riemann::Solver Solver>() {
      return CalculateFluxesBlockImpl<Method, Solver>(data);
    });
  });
}

TaskStatus CalculateFluxesMeshTask(MeshData<Real>* data) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized())
    return nr::CalculateSyncGRHDFluxesMeshTask(data);
  const auto package = data->GetParentPointer()->packages.Get("hydro");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian)
    return relativity::CalculateFluxesMeshTask(data);
  const auto reconstruction = static_cast<Reconstruction>(package->Param<int>("reconstruction"));
  const auto solver = static_cast<riemann::Solver>(package->Param<int>("riemann"));
  PARTHENON_REQUIRE(solver != riemann::Solver::none,
                    "The packed Hydro flux path requires an active Riemann solver");
  return reconstruct::Visit(reconstruction, [&]<Reconstruction Method>() {
    return riemann::Visit<riemann::Physics::newtonian_hydro>(solver, [&]<riemann::Solver Solver>() {
      return CalculateFluxesMeshImpl<Method, Solver>(data);
    });
  });
}

void ConservedToPrimitiveBlock(MeshBlockData<Real>* data) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized()) {
    nr::SyncGRHDConservedToPrimitiveBlock(data);
    return;
  }
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("hydro");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian) {
    relativity::ConservedToPrimitiveBlock(data);
    return;
  }
  const auto eos = GetEos(package);
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const int hydro_components = eos.HasEnergy() ? kIdealComponents : kIsothermalComponents;
  const int scalars = package->Param<int>("nscalars");
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  block->par_for(
      "PANGU hydro conserved to primitive", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real density = conserved(IDN, k, j, i);
        if (!(density >= eos.density_floor)) {
          density = eos.density_floor;
          conserved(IDN, k, j, i) = density;
          conserved(IM1, k, j, i) = 0.0;
          conserved(IM2, k, j, i) = 0.0;
          conserved(IM3, k, j, i) = 0.0;
        }
        primitive(IDN, k, j, i) = density;
        Real velocity_squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
          const Real velocity = conserved(IM1 + axis, k, j, i) / density;
          primitive(IV1 + axis, k, j, i) = velocity;
          velocity_squared += velocity * velocity;
        }
        if (eos.HasEnergy()) {
          const Real kinetic = 0.5 * density * velocity_squared;
          Real pressure = (eos.gamma - 1.0) * (conserved(IEN, k, j, i) - kinetic);
          if (!(pressure >= eos.pressure_floor)) {
            pressure = eos.pressure_floor;
            conserved(IEN, k, j, i) = kinetic + pressure / (eos.gamma - 1.0);
          }
          primitive(IPR, k, j, i) = pressure;
        }
        for (int scalar = 0; scalar < scalars; ++scalar) {
          primitive(hydro_components + scalar, k, j, i) =
              fmin(fmax(conserved(hydro_components + scalar, k, j, i) / density, 0.0), 1.0);
        }
      });
}

void ConservedToPrimitiveMesh(MeshData<Real>* data) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized()) {
    nr::SyncGRHDConservedToPrimitiveMesh(data);
    return;
  }
  const auto package = data->GetParentPointer()->packages.Get("hydro");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian) {
    relativity::ConservedToPrimitiveMesh(data);
    return;
  }
  const auto eos = GetEos(package);
  const auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const int hydro_components = eos.HasEnergy() ? kIdealComponents : kIsothermalComponents;
  const int scalars = package->Param<int>("nscalars");
  const auto ib = data->GetBoundsI(IndexDomain::entire);
  const auto jb = data->GetBoundsJ(IndexDomain::entire);
  const auto kb = data->GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU hydro packed conserved to primitive", parthenon::DevExecSpace(),
      0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real density = conserved(b, IDN, k, j, i);
        if (!(density >= eos.density_floor)) {
          density = eos.density_floor;
          conserved(b, IDN, k, j, i) = density;
          conserved(b, IM1, k, j, i) = 0.0;
          conserved(b, IM2, k, j, i) = 0.0;
          conserved(b, IM3, k, j, i) = 0.0;
        }
        primitive(b, IDN, k, j, i) = density;
        Real velocity_squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
          const Real velocity = conserved(b, IM1 + axis, k, j, i) / density;
          primitive(b, IV1 + axis, k, j, i) = velocity;
          velocity_squared += velocity * velocity;
        }
        if (eos.HasEnergy()) {
          const Real kinetic = 0.5 * density * velocity_squared;
          Real pressure = (eos.gamma - 1.0) * (conserved(b, IEN, k, j, i) - kinetic);
          if (!(pressure >= eos.pressure_floor)) {
            pressure = eos.pressure_floor;
            conserved(b, IEN, k, j, i) = kinetic + pressure / (eos.gamma - 1.0);
          }
          primitive(b, IPR, k, j, i) = pressure;
        }
        for (int scalar = 0; scalar < scalars; ++scalar) {
          primitive(b, hydro_components + scalar, k, j, i) =
              fmin(fmax(conserved(b, hydro_components + scalar, k, j, i) / density, 0.0), 1.0);
        }
      });
}

Real EstimateTimestepBlock(MeshBlockData<Real>* data) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized())
    return nr::EstimateSyncGRHDTimestepBlock(data);
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("hydro");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian)
    return relativity::EstimateTimestepBlock(data);
  const auto eos = GetEos(package);
  const auto cfl = package->Param<Real>("cfl");
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  const auto ndim = block->pmy_mesh->ndim;
  Real minimum;
  ParReduce(
      "PANGU hydro timestep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
        Primitive state{primitive(IDN, k, j, i),
                        {primitive(IV1, k, j, i), primitive(IV2, k, j, i), primitive(IV3, k, j, i)},
                        eos.HasEnergy()
                            ? primitive(IPR, k, j, i)
                            : eos.iso_sound_speed * eos.iso_sound_speed * primitive(IDN, k, j, i)};
        const Real sound = eos.SoundSpeed(state);
        Real cell_dt = coordinates.Dxc<X1DIR>(k, j, i) / (fabs(state.velocity[0]) + sound);
        if (ndim >= 2)
          cell_dt =
              fmin(cell_dt, coordinates.Dxc<X2DIR>(k, j, i) / (fabs(state.velocity[1]) + sound));
        if (ndim >= 3)
          cell_dt =
              fmin(cell_dt, coordinates.Dxc<X3DIR>(k, j, i) / (fabs(state.velocity[2]) + sound));
        local = fmin(local, cell_dt);
      },
      Kokkos::Min<Real>(minimum));
  return cfl * minimum;
}

Real EstimateTimestepMesh(MeshData<Real>* data) {
  if constexpr (geometry::ConfiguredGeometryIsSynchronized())
    return nr::EstimateSyncGRHDTimestepMesh(data);
  const auto package = data->GetParentPointer()->packages.Get("hydro");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian)
    return relativity::EstimateTimestepMesh(data);
  const auto eos = GetEos(package);
  const auto cfl = package->Param<Real>("cfl");
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const auto ndim = data->GetParentPointer()->ndim;
  Real minimum = std::numeric_limits<Real>::max();
  ParReduce(
      "PANGU hydro packed timestep", 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        Primitive state{
            primitive(b, IDN, k, j, i),
            {primitive(b, IV1, k, j, i), primitive(b, IV2, k, j, i), primitive(b, IV3, k, j, i)},
            eos.HasEnergy()
                ? primitive(b, IPR, k, j, i)
                : eos.iso_sound_speed * eos.iso_sound_speed * primitive(b, IDN, k, j, i)};
        const Real sound = eos.SoundSpeed(state);
        const auto& coordinates = primitive.GetCoords(b);
        Real cell_dt = coordinates.Dxc<X1DIR>(k, j, i) / (fabs(state.velocity[0]) + sound);
        if (ndim >= 2)
          cell_dt =
              fmin(cell_dt, coordinates.Dxc<X2DIR>(k, j, i) / (fabs(state.velocity[1]) + sound));
        if (ndim >= 3)
          cell_dt =
              fmin(cell_dt, coordinates.Dxc<X3DIR>(k, j, i) / (fabs(state.velocity[2]) + sound));
        local = fmin(local, cell_dt);
      },
      Kokkos::Min<Real>(minimum));
  return cfl * minimum;
}

AmrTag CheckRefinementBlock(MeshBlockData<Real>* data) {
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("hydro");
  const auto refine = package->Param<Real>("refine_tolerance");
  const auto derefine = package->Param<Real>("derefine_tolerance");
  const auto density = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto ndim = block->pmy_mesh->ndim;
  Real maximum;
  ParReduce(
      "PANGU hydro density-gradient AMR", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
        const Real center = fmax(fabs(density(IDN, k, j, i)), 1.0e-30);
        Real gradient =
            fabs(density(IDN, k, j, i + 1) - density(IDN, k, j, i - 1)) / (2.0 * center);
        if (ndim >= 2) {
          gradient = fmax(gradient, fabs(density(IDN, k, j + 1, i) - density(IDN, k, j - 1, i)) /
                                        (2.0 * center));
        }
        if (ndim >= 3) {
          gradient = fmax(gradient, fabs(density(IDN, k + 1, j, i) - density(IDN, k - 1, j, i)) /
                                        (2.0 * center));
        }
        local = fmax(local, gradient);
      },
      Kokkos::Max<Real>(maximum));
  if (maximum > refine)
    return AmrTag::refine;
  if (maximum < derefine)
    return AmrTag::derefine;
  return AmrTag::same;
}

TaskStatus ApplySourcesBlockTask(std::shared_ptr<MeshBlockData<Real>>& data, const Real dt) {
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("hydro");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian)
    return relativity::ApplySourcesBlockTask(data, dt);
  const auto eos = GetEos(package);
  const Real accel1 = package->Param<Real>("accel1");
  const Real accel2 = package->Param<Real>("accel2");
  const Real accel3 = package->Param<Real>("accel3");
  if (accel1 != 0.0 || accel2 != 0.0 || accel3 != 0.0) {
    auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
    const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
    block->par_for(
        "PANGU hydro constant acceleration", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real density = conserved(IDN, k, j, i);
          const Real work = conserved(IM1, k, j, i) * accel1 + conserved(IM2, k, j, i) * accel2 +
                            conserved(IM3, k, j, i) * accel3;
          const Real acceleration_squared = accel1 * accel1 + accel2 * accel2 + accel3 * accel3;
          conserved(IM1, k, j, i) += dt * density * accel1;
          conserved(IM2, k, j, i) += dt * density * accel2;
          conserved(IM3, k, j, i) += dt * density * accel3;
          if (eos.HasEnergy()) {
            conserved(IEN, k, j, i) += dt * work + 0.5 * dt * dt * density * acceleration_squared;
          }
        });
  }
  srcterms::ApplyHydroBlockTask(data, dt);
  return TaskStatus::complete;
}

TaskStatus ApplySourcesMeshTask(MeshData<Real>* data, const Real dt) {
  const auto package = data->GetParentPointer()->packages.Get("hydro");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian)
    return relativity::ApplySourcesMeshTask(data, dt);
  PARTHENON_FAIL("Packed Newtonian Hydro sources are not enabled for this driver path");
}

TaskStatus ApplyFirstOrderFluxCorrectionBlockTask(std::shared_ptr<MeshBlockData<Real>>& data,
                                                  std::shared_ptr<MeshBlockData<Real>>& base,
                                                  const Real gam0, const Real gam1,
                                                  const Real beta_dt) {
  const Real dt = beta_dt;
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("hydro");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian)
    return relativity::ApplyHydroFluxCorrectionBlockTask(data, base, gam0, gam1, beta_dt);
  if (!package->Param<bool>("fofc"))
    return TaskStatus::complete;
  if (static_cast<riemann::Solver>(package->Param<int>("riemann")) == riemann::Solver::none)
    return TaskStatus::complete;
  const auto eos = GetEos(package);
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"hydro.cons"});
  const auto base_conserved = base->PackVariables(std::vector<std::string>{"hydro.cons"});
  const auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  const auto flags = data->PackVariables(std::vector<std::string>{"hydro.fofc"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  const auto ndim = block->pmy_mesh->ndim;
  block->par_for(
      "PANGU hydro FOFC candidate", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real density =
            gam0 * conserved(IDN, k, j, i) + gam1 * base_conserved(IDN, k, j, i) -
            dt * (conserved.flux(X1DIR, IDN, k, j, i + 1) - conserved.flux(X1DIR, IDN, k, j, i)) /
                coordinates.Dxc<X1DIR>(k, j, i);
        Real momentum[3];
        for (int axis = 0; axis < 3; ++axis) {
          momentum[axis] = gam0 * conserved(IM1 + axis, k, j, i) +
                           gam1 * base_conserved(IM1 + axis, k, j, i) -
                           dt *
                               (conserved.flux(X1DIR, IM1 + axis, k, j, i + 1) -
                                conserved.flux(X1DIR, IM1 + axis, k, j, i)) /
                               coordinates.Dxc<X1DIR>(k, j, i);
        }
        Real energy = eos.HasEnergy()
                          ? gam0 * conserved(IEN, k, j, i) + gam1 * base_conserved(IEN, k, j, i) -
                                dt *
                                    (conserved.flux(X1DIR, IEN, k, j, i + 1) -
                                     conserved.flux(X1DIR, IEN, k, j, i)) /
                                    coordinates.Dxc<X1DIR>(k, j, i)
                          : 0.0;
        if (ndim >= 2) {
          density -=
              dt * (conserved.flux(X2DIR, IDN, k, j + 1, i) - conserved.flux(X2DIR, IDN, k, j, i)) /
              coordinates.Dxc<X2DIR>(k, j, i);
          for (int axis = 0; axis < 3; ++axis) {
            momentum[axis] -= dt *
                              (conserved.flux(X2DIR, IM1 + axis, k, j + 1, i) -
                               conserved.flux(X2DIR, IM1 + axis, k, j, i)) /
                              coordinates.Dxc<X2DIR>(k, j, i);
          }
          if (eos.HasEnergy()) {
            energy -=
                dt *
                (conserved.flux(X2DIR, IEN, k, j + 1, i) - conserved.flux(X2DIR, IEN, k, j, i)) /
                coordinates.Dxc<X2DIR>(k, j, i);
          }
        }
        if (ndim >= 3) {
          density -=
              dt * (conserved.flux(X3DIR, IDN, k + 1, j, i) - conserved.flux(X3DIR, IDN, k, j, i)) /
              coordinates.Dxc<X3DIR>(k, j, i);
          for (int axis = 0; axis < 3; ++axis) {
            momentum[axis] -= dt *
                              (conserved.flux(X3DIR, IM1 + axis, k + 1, j, i) -
                               conserved.flux(X3DIR, IM1 + axis, k, j, i)) /
                              coordinates.Dxc<X3DIR>(k, j, i);
          }
          if (eos.HasEnergy()) {
            energy -=
                dt *
                (conserved.flux(X3DIR, IEN, k + 1, j, i) - conserved.flux(X3DIR, IEN, k, j, i)) /
                coordinates.Dxc<X3DIR>(k, j, i);
          }
        }
        bool invalid = !(density >= eos.density_floor);
        if (eos.HasEnergy() && !invalid) {
          const Real kinetic =
              0.5 *
              (momentum[0] * momentum[0] + momentum[1] * momentum[1] + momentum[2] * momentum[2]) /
              density;
          const Real pressure = (eos.gamma - 1.0) * (energy - kinetic);
          invalid = !(pressure >= eos.pressure_floor);
        }
        flags(0, k, j, i) = invalid ? 1.0 : 0.0;
      });
  ReplaceFlaggedFacesBlock<0>(block, primitive, flags, conserved, eos, ib, jb, kb);
  if (ndim >= 2)
    ReplaceFlaggedFacesBlock<1>(block, primitive, flags, conserved, eos, ib, jb, kb);
  if (ndim >= 3)
    ReplaceFlaggedFacesBlock<2>(block, primitive, flags, conserved, eos, ib, jb, kb);
  return TaskStatus::complete;
}

TaskStatus ApplyFirstOrderFluxCorrectionMeshTask(MeshData<Real>* data, MeshData<Real>* base,
                                                 const Real gam0, const Real gam1,
                                                 const Real beta_dt) {
  const auto package = data->GetParentPointer()->packages.Get("hydro");
  if (static_cast<relativity::HydroMode>(package->Param<int>("physics_mode")) !=
      relativity::HydroMode::newtonian)
    return relativity::ApplyHydroFluxCorrectionMeshTask(data, base, gam0, gam1, beta_dt);
  PARTHENON_FAIL("Packed Newtonian Hydro FOFC is not enabled for this driver path");
}

TaskStatus CarryStateMeshTask(MeshData<Real>* current, MeshData<Real>* next) {
  return parthenon::Update::CopyData(std::vector<std::string>{"hydro.cons", "hydro.prim"}, current,
                                     next);
}

} // namespace pangu::hydro
