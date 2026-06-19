// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "initialization/timestep_estimation.h"

#include <limits>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"

parthenon::Real EstimateTimestepMesh(
    parthenon::MeshData<parthenon::Real> *resource) {
  using namespace parthenon;
  const auto pmb0 = resource->GetBlockData(0)->GetBlockPointer();
  const auto package_core = pmb0->packages.Get("core");
  const auto kCflNumber = package_core->Param<Real>("cfl_number");

  const auto meshgrid_size_x1 =
      pmb0->cellbounds.ncellsi(IndexDomain::entire);
  const auto meshgrid_size_x2 =
      pmb0->cellbounds.ncellsj(IndexDomain::entire);
  const auto meshgrid_size_x3 =
      pmb0->cellbounds.ncellsk(IndexDomain::entire);

  const int offset_x1 = (meshgrid_size_x1 > 1) ? 1 : 0;
  const int offset_x2 = (meshgrid_size_x2 > 1) ? 1 : 0;
  const int offset_x3 = (meshgrid_size_x3 > 1) ? 1 : 0;

  Real globalMinimumOfTimestep = std::numeric_limits<Real>::max();

  for (int b = 0; b < resource->NumBlocks(); ++b) {
    auto &mbd = resource->GetBlockData(b);
    const auto pmb = mbd->GetBlockPointer();

    const auto bound_x1_interior =
        pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto bound_x2_interior =
        pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto bound_x3_interior =
        pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    const auto &Coords = pmb->coords;

    Real minimumOfTimestepX1 = std::numeric_limits<Real>::max();
    Real minimumOfTimestepX2 = std::numeric_limits<Real>::max();
    Real minimumOfTimestepX3 = std::numeric_limits<Real>::max();

    PackIndexMap fastMsIndexMap;
    const std::vector<std::string> fast_ms_tags = {"alfven"};
    const auto FastMagnetosonicSpeed =
        mbd->PackVariables(fast_ms_tags, fastMsIndexMap);

    if (offset_x1)
      pmb->par_reduce(
          PARTHENON_AUTO_LABEL,
          bound_x3_interior.s, bound_x3_interior.e,
          bound_x2_interior.s, bound_x2_interior.e,
          bound_x1_interior.s, bound_x1_interior.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i, Real &timestepX1) {
            const Real c = FastMagnetosonicSpeed(Vector3D::X1, k, j, i);
            const Real dt = kCflNumber * Coords.Dx<X1DIR>() / c;
            if (timestepX1 > dt) timestepX1 = dt;
          },
          Kokkos::Min<Real>(minimumOfTimestepX1));

    if (offset_x2)
      pmb->par_reduce(
          PARTHENON_AUTO_LABEL,
          bound_x3_interior.s, bound_x3_interior.e,
          bound_x2_interior.s, bound_x2_interior.e,
          bound_x1_interior.s, bound_x1_interior.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i, Real &timestepX2) {
            const Real c = FastMagnetosonicSpeed(Vector3D::X2, k, j, i);
            const Real dt = kCflNumber * Coords.Dx<X2DIR>() / c;
            if (timestepX2 > dt) timestepX2 = dt;
          },
          Kokkos::Min<Real>(minimumOfTimestepX2));

    if (offset_x3)
      pmb->par_reduce(
          PARTHENON_AUTO_LABEL,
          bound_x3_interior.s, bound_x3_interior.e,
          bound_x2_interior.s, bound_x2_interior.e,
          bound_x1_interior.s, bound_x1_interior.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i, Real &timestepX3) {
            const Real c = FastMagnetosonicSpeed(Vector3D::X3, k, j, i);
            const Real dt = kCflNumber * Coords.Dx<X3DIR>() / c;
            if (timestepX3 > dt) timestepX3 = dt;
          },
          Kokkos::Min<Real>(minimumOfTimestepX3));

    const Real MinimumOfTimestep =
        1.0 / (1.0 / minimumOfTimestepX1 + 1.0 / minimumOfTimestepX2 +
               1.0 / minimumOfTimestepX3);
    if (MinimumOfTimestep < globalMinimumOfTimestep)
      globalMinimumOfTimestep = MinimumOfTimestep;
  }

  return globalMinimumOfTimestep;
}
