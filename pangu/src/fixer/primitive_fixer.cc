// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "fixer/primitive_fixer.h"

#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"
#include "physics/state_calculation.h"
#include "physics/heating_model.h"

parthenon::TaskStatus FixPrimitive(parthenon::MeshData<parthenon::Real> *md) {
  using namespace parthenon::package::prelude;
  PARTHENON_INSTRUMENT
  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  const auto package_core = pmb0->packages.Get("core");
  const auto kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");
  const auto density_floor = package_core->Param<Real>("density_floor");
  const auto density_floor_pow = package_core->Param<Real>("density_floor_pow");
  const auto energy_floor = package_core->Param<Real>("energy_floor");
  const auto energy_floor_pow = package_core->Param<Real>("energy_floor_pow");
  const auto sigma_max = package_core->Param<Real>("sigma_max");
  const auto lorentz_max = package_core->Param<Real>("lorentz_max");
  const RatioLimits ratio = {
      package_core->Param<Real>("ratio_min"), package_core->Param<Real>("ratio_max")};

  const auto bound_x1_entire = md->GetBoundsI(IndexDomain::entire);
  const auto bound_x2_entire = md->GetBoundsJ(IndexDomain::entire);
  const auto bound_x3_entire = md->GetBoundsK(IndexDomain::entire);
  auto block = IndexRange{0, md->NumBlocks() - 1};

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
      "electron_entropy"
  };
  auto primitive = md->PackVariables(primitive_tags, primitiveIndexMap);
  auto covariant_metric =
      md->PackVariables(std::vector<std::string>{"covariant_metric"});
  auto contravariant_metric =
      md->PackVariables(std::vector<std::string>{"contravariant_metric"});

  const Real umax2 = lorentz_max * lorentz_max - 1.0;

  pmb0->par_for(
      PARTHENON_AUTO_LABEL, block.s, block.e,
      bound_x3_entire.s, bound_x3_entire.e, bound_x2_entire.s, bound_x2_entire.e,
      bound_x1_entire.s, bound_x1_entire.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        constexpr Real small = 1.0e-20;
        const Real sigma_max_safe = Kokkos::max(sigma_max, small);
        const Real rho_before_floor = Kokkos::max(primitive(b, RHO, k, j, i), small);

        const auto &coords = primitive.GetCoords(b);
        const auto x = coords.Xc<X1DIR>(i);
        const auto y = coords.Xc<X2DIR>(j);
        const auto z = coords.Xc<X3DIR>(k);
        const auto r = Kokkos::sqrt(x * x + y * y + z * z);

        Real rho_floor = density_floor;
        const Real eng_floor = energy_floor;

        Real gcov[4][4], gcon[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcov[row][col] =
                covariant_metric(b, CENTER * 16 + col * 4 + row, k, j, i);
            gcon[row][col] =
                contravariant_metric(b, CENTER * 16 + col * 4 + row, k, j, i);
          }
        }

        const Real rho = primitive(b, RHO, k, j, i);
        const Real ug = primitive(b, ENY, k, j, i);
        Real primitive_array[NPRIM];
        for (int index = 0; index < NPRIM; ++index) {
          primitive_array[index] = primitive(b, index, k, j, i);
        }
        State state;
        CalculateState(primitive_array, gcov, gcon, state);

        if (rho_floor < ug / sigma_max_safe) {
          rho_floor = ug / sigma_max_safe;
        }
        const Real sigma = state.bsq / rho;
        if (sigma > sigma_max) {
          rho_floor = state.bsq / sigma_max_safe;
        }

        if (primitive(b, RHO, k, j, i) < rho_floor)
          primitive(b, RHO, k, j, i) = rho_floor;
        if (primitive(b, ENY, k, j, i) < eng_floor)
          primitive(b, ENY, k, j, i) = eng_floor;

        const Real u1 = primitive(b, UX1, k, j, i);
        const Real u2 = primitive(b, UX2, k, j, i);
        const Real u3 = primitive(b, UX3, k, j, i);
        const Real uvsq = gcov[1][1] * u1 * u1 + gcov[2][2] * u2 * u2 +
                          gcov[3][3] * u3 * u3 +
                          2.0 * (gcov[1][2] * u1 * u2 + gcov[1][3] * u1 * u3 +
                                 gcov[2][3] * u2 * u3);
        if (umax2 > 0.0 && uvsq > umax2) {
          const Real factor = Kokkos::sqrt(umax2 / Kokkos::max(uvsq, small));
          primitive(b, UX1, k, j, i) *= factor;
          primitive(b, UX2, k, j, i) *= factor;
          primitive(b, UX3, k, j, i) *= factor;
        }

        const Real rho_after_floor = Kokkos::max(primitive(b, RHO, k, j, i), small);
        if (rho_after_floor != rho_before_floor) {
          primitive(b, KEL, k, j, i) *=
              Kokkos::pow(rho_after_floor / rho_before_floor, -kAdiabaticIndex);
        }

        primitive(b, ENT, k, j, i) =
            (kAdiabaticIndex - 1.0) * primitive(b, ENY, k, j, i) *
            Kokkos::pow(primitive(b, RHO, k, j, i), -kAdiabaticIndex);
        primitive(b, KEL, k, j, i) =
            clampByRatio(ratio, primitive(b, ENT, k, j, i), primitive(b, KEL, k, j, i));
      });

  return parthenon::TaskStatus::complete;
}
