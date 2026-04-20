// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/numerical_correction module defines primitive_fixer.cc
// responsibilities for the Pangu runtime. It centers on numerical_correction to express
// core data flow, keep interfaces readable, and preserve predictable behavior across task
// coordination, recovery paths, and performance-sensitive execution.

#include "numerical_correction/primitive_fixer.h"

#include <memory>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"
#include "physics_transformation/state_calculation.h"

parthenon::TaskStatus FixPrimitiveSRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
  using namespace parthenon::package::prelude;
  PARTHENON_INSTRUMENT
  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
  const auto density_floor = package_core->Param<Real>("density_floor");
  const auto energy_floor = package_core->Param<Real>("energy_floor");
  const auto sigma_max = package_core->Param<Real>("sigma_max");
  const auto lorentz_max = package_core->Param<Real>("lorentz_max");

  const auto bound_x1_entire =
      pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto bound_x2_entire =
      pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto bound_x3_entire =
      pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field"};
  auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);

  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_entire.s, bound_x3_entire.e, bound_x2_entire.s, bound_x2_entire.e,
      bound_x1_entire.s, bound_x1_entire.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        constexpr Real small = 1.0e-20;
        const Real sigma_max_safe = Kokkos::max(sigma_max, small);

        Real rho_floor = density_floor;
        const Real ug_floor = energy_floor;

        const Real rho = primitive(RHO, k, j, i);
        const Real ug = primitive(ENY, k, j, i);

        const Real primitive_array[NPRIM] = {
            primitive(RHO, k, j, i), primitive(ENY, k, j, i),
            primitive(UX1, k, j, i), primitive(UX2, k, j, i),
            primitive(UX3, k, j, i), primitive(BX1, k, j, i),
            primitive(BX2, k, j, i), primitive(BX3, k, j, i),
        };
        State state;
        CalculateSRMHDState(primitive_array, state);
        const Real bsq = Kokkos::max(dot4(state.bcon, state.bcov), 0.0);

        if (rho_floor < ug / sigma_max_safe) {
          rho_floor = ug / sigma_max_safe;
        }
        const Real sigma = bsq / Kokkos::max(rho, small);
        if (sigma > sigma_max) {
          rho_floor = Kokkos::max(rho_floor, bsq / sigma_max_safe);
        }

        if (primitive(RHO, k, j, i) < rho_floor)
          primitive(RHO, k, j, i) = rho_floor;
        if (primitive(ENY, k, j, i) < ug_floor)
          primitive(ENY, k, j, i) = ug_floor;

        const Real umax2 = lorentz_max * lorentz_max - 1.0;
        if (umax2 > 0.0) {
          const Real u1 = primitive(UX1, k, j, i);
          const Real u2 = primitive(UX2, k, j, i);
          const Real u3 = primitive(UX3, k, j, i);
          const Real uvsq = u1 * u1 + u2 * u2 + u3 * u3;
          if (uvsq > umax2) {
            const Real factor = Kokkos::sqrt(umax2 / Kokkos::max(uvsq, small));
            primitive(UX1, k, j, i) *= factor;
            primitive(UX2, k, j, i) *= factor;
            primitive(UX3, k, j, i) *= factor;
          }
        }
      });

  return TaskStatus::complete;
}

parthenon::TaskStatus FixPrimitiveGRMHD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource) {
  using namespace parthenon::package::prelude;
  PARTHENON_INSTRUMENT
  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
  const auto density_floor = package_core->Param<Real>("density_floor");
  const auto density_floor_pow = package_core->Param<Real>("density_floor_pow");
  const auto energy_floor = package_core->Param<Real>("energy_floor");
  const auto energy_floor_pow = package_core->Param<Real>("energy_floor_pow");
  const auto sigma_max = package_core->Param<Real>("sigma_max");
  const auto lorentz_max = package_core->Param<Real>("lorentz_max");

  const auto bound_x1_entire =
      pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto bound_x2_entire =
      pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto bound_x3_entire =
      pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto coords = pmb->coords;

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field"};
  auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);
  auto covariant_metric = geom_resource->Get("covariant_metric").data;
  auto contravariant_metric = geom_resource->Get("contravariant_metric").data;

  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_entire.s, bound_x3_entire.e, bound_x2_entire.s, bound_x2_entire.e,
      bound_x1_entire.s, bound_x1_entire.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        constexpr Real small = 1.0e-20;
        const Real sigma_max_safe = Kokkos::max(sigma_max, small);

        const auto x1 = coords.Xc<X1DIR>(i);
        const auto r = Kokkos::max(Kokkos::exp(x1), 1.0e-20);

        Real rho_floor = density_floor * Kokkos::pow(r, density_floor_pow);
        const Real eng_floor = energy_floor * Kokkos::pow(r, energy_floor_pow);

        Real gcov[4][4], gcon[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcov[row][col] = covariant_metric(CENTER, col, row, k, j, i);
            gcon[row][col] = contravariant_metric(CENTER, col, row, k, j, i);
          }
        }

        const Real rho = primitive(RHO, k, j, i);
        const Real ug = primitive(ENY, k, j, i);
        const Real primitive_array[NPRIM] = {
            primitive(RHO, k, j, i), primitive(ENY, k, j, i),
            primitive(UX1, k, j, i), primitive(UX2, k, j, i),
            primitive(UX3, k, j, i), primitive(BX1, k, j, i),
            primitive(BX2, k, j, i), primitive(BX3, k, j, i),
        };
        State state;
        CalculateGRMHDState(primitive_array, gcov, gcon, state);
        const Real bsq = Kokkos::max(dot4(state.bcon, state.bcov), 0.0);

        if (rho_floor < ug / sigma_max_safe) {
          rho_floor = ug / sigma_max_safe;
        }
        const Real sigma = bsq / Kokkos::max(rho, small);
        if (sigma > sigma_max) {
          rho_floor = Kokkos::max(rho_floor, bsq / sigma_max_safe);
        }

        if (primitive(RHO, k, j, i) < rho_floor)
          primitive(RHO, k, j, i) = rho_floor;
        if (primitive(ENY, k, j, i) < eng_floor)
          primitive(ENY, k, j, i) = eng_floor;

        const Real u1 = primitive(UX1, k, j, i);
        const Real u2 = primitive(UX2, k, j, i);
        const Real u3 = primitive(UX3, k, j, i);
        const Real uvsq = gcov[1][1] * u1 * u1 + gcov[2][2] * u2 * u2 +
                          gcov[3][3] * u3 * u3 +
                          2.0 * (gcov[1][2] * u1 * u2 + gcov[1][3] * u1 * u3 +
                                 gcov[2][3] * u2 * u3);
        const Real umax2 = lorentz_max * lorentz_max - 1.0;
        if (umax2 > 0.0 && uvsq > umax2) {
          const Real factor = Kokkos::sqrt(umax2 / Kokkos::max(uvsq, small));
          primitive(UX1, k, j, i) *= factor;
          primitive(UX2, k, j, i) *= factor;
          primitive(UX3, k, j, i) *= factor;
        }
      });

  return TaskStatus::complete;
}
