// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "riemann_solver/electron_heating.h"

#include <memory>
#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"
#include "physics/state_calculation.h"
#include "physics/heating_model.h"

parthenon::TaskStatus ApplyElectronHeating(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
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

  const auto bound_x1_interior =
      pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto bound_x2_interior =
      pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto bound_x3_interior =
      pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
      "electron_entropy"};
  auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);

  auto covariant_metric = geom_resource->Get("covariant_metric").data;
  auto contravariant_metric = geom_resource->Get("contravariant_metric").data;

  PackIndexMap conservativeIndexMap;
  const std::vector<std::string> conservative_tags = {"conservative"};
  const auto conservative =
      resource->PackVariables(conservative_tags, conservativeIndexMap);

  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_interior.s, bound_x3_interior.e,
      bound_x2_interior.s, bound_x2_interior.e, bound_x1_interior.s,
      bound_x1_interior.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real gcov[4][4], gcon[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcov[row][col] = covariant_metric(CENTER, col, row, k, j, i);
            gcon[row][col] = contravariant_metric(CENTER, col, row, k, j, i);
          }
        }

        Real primitive_array[NPRIM];
        for (int index = 0; index < NPRIM; ++index) {
          primitive_array[index] = primitive(index, k, j, i);
        }
        State state;
        CalculateState(primitive_array, gcov, gcon, state);

        const Real cons_rho = conservative(RHO, k, j, i);
        const CellState cell = {
            primitive(RHO, k, j, i),
            primitive(ENY, k, j, i),
            state.bsq,
            conservative(ENT, k, j, i) / cons_rho,
            primitive(ENT, k, j, i),
            conservative(KEL, k, j, i) / cons_rho};
        primitive(KEL, k, j, i) = apply(heating_model_params, cell);
      });

  return TaskStatus::complete;
}
