// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "riemann_solver/electron_heating.h"

#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"
#include "physics/state_calculation.h"
#include "physics/heating_model.h"

parthenon::TaskStatus ApplyElectronHeating(
    parthenon::MeshData<parthenon::Real> *md) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  const auto package_core = pmb0->packages.Get("core");
  const auto enable_B = package_core->Param<bool>("enable_B");
  const auto enable_heating = package_core->Param<bool>("enable_heating");
  const auto& fnames = package_core->Param<std::vector<std::string>>("primitive_field_names");
  const auto kModelName = package_core->Param<std::string>("model_name");
  const Parameters heating_model_params = {
      parseModel(kModelName),
      package_core->Param<Real>("adiabatic_index"),
      package_core->Param<Real>("gamma_p"),
      package_core->Param<Real>("gamma_e"),
      package_core->Param<Real>("fel_constant"),
      {package_core->Param<Real>("ratio_min"), package_core->Param<Real>("ratio_max")},
      package_core->Param<bool>("limit_kel"),
      package_core->Param<bool>("suppress_highb_heat"),
      package_core->Param<bool>("enforce_positive_dissipation")};

  const auto bound_x1_interior = md->GetBoundsI(IndexDomain::interior);
  const auto bound_x2_interior = md->GetBoundsJ(IndexDomain::interior);
  const auto bound_x3_interior = md->GetBoundsK(IndexDomain::interior);
  auto block = IndexRange{0, md->NumBlocks() - 1};

  PackIndexMap idxMap;
  auto primitive = md->PackVariables(fnames, idxMap);

  const int iRHO = idxMap["density"].first;
  const int iENY = idxMap["energy"].first;
  const int iUX  = idxMap["weighted_velocity"].first;
  const int iENT = idxMap["entropy"].first;
  const int iBX  = enable_B ? idxMap["magnetic_field"].first : -1;
  const int iKEL = enable_heating ? idxMap["electron_entropy"].first : -1;

  auto covariant_metric =
      md->PackVariables(std::vector<std::string>{"covariant_metric"});
  auto contravariant_metric =
      md->PackVariables(std::vector<std::string>{"contravariant_metric"});

  PackIndexMap conservativeIndexMap;
  const std::vector<std::string> conservative_tags = {"conservative"};
  const auto conservative =
      md->PackVariables(conservative_tags, conservativeIndexMap);

  pmb0->par_for(
      PARTHENON_AUTO_LABEL, block.s, block.e,
      bound_x3_interior.s, bound_x3_interior.e,
      bound_x2_interior.s, bound_x2_interior.e, bound_x1_interior.s,
      bound_x1_interior.e, KOKKOS_LAMBDA(const int b, const int k, const int j,
                                          const int i) {
        Real gcov[4][4], gcon[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcov[row][col] =
                covariant_metric(b, CENTER * 16 + col * 4 + row, k, j, i);
            gcon[row][col] =
                contravariant_metric(b, CENTER * 16 + col * 4 + row, k, j, i);
          }
        }

        Real pcarr[NPRIM] = {0};
        pcarr[RHO] = primitive(b, iRHO, k, j, i);
        pcarr[ENY] = primitive(b, iENY, k, j, i);
        pcarr[UX1] = primitive(b, iUX,   k, j, i);
        pcarr[UX2] = primitive(b, iUX+1, k, j, i);
        pcarr[UX3] = primitive(b, iUX+2, k, j, i);
        if (enable_B) {
          pcarr[BX1] = primitive(b, iBX,   k, j, i);
          pcarr[BX2] = primitive(b, iBX+1, k, j, i);
          pcarr[BX3] = primitive(b, iBX+2, k, j, i);
        }
        pcarr[ENT] = primitive(b, iENT, k, j, i);
        if (enable_heating) {
          pcarr[KEL] = primitive(b, iKEL, k, j, i);
        }

        State state;
        CalculateState(pcarr, gcov, gcon, state);

        const Real cons_rho = conservative(b, iRHO, k, j, i);
        const CellState cell = {
            primitive(b, iRHO, k, j, i),
            primitive(b, iENY, k, j, i),
            state.bsq,
            conservative(b, iENT, k, j, i) / cons_rho,
            primitive(b, iENT, k, j, i),
            conservative(b, iKEL, k, j, i) / cons_rho};
        if (enable_heating) {
          primitive(b, iKEL, k, j, i) = apply(heating_model_params, cell);
        }
      });

  return TaskStatus::complete;
}
