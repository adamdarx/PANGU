// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "riemann_solver/electron_heating.h"

#include <memory>
#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"
#include "physics/state_calculation.h"
#include "physics/two_temperature.h"

parthenon::TaskStatus ApplyElectronHeating(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &geom_resource) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
  const auto kModelName = package_core->Param<std::string>("model_name");
  const auto kHeatingModel = two_temperature::StringToMODEL(kModelName);
  const auto kFelConstant = package_core->Param<Real>("fel_constant");
    const auto kGamma = package_core->Param<Real>("adiabatic_index");
    const auto kGammaP = package_core->Param<Real>("gamma_p");
    const auto kGammaE = package_core->Param<Real>("gamma_e");
    const auto kLimitKel = package_core->Param<bool>("limit_kel");
    const auto kRatioMin = package_core->Param<Real>("ratio_min");
    const auto kRatioMax = package_core->Param<Real>("ratio_max");
    const auto kSuppressHighbHeat = package_core->Param<bool>("suppress_highb_heat");
    const auto kEnforcePositiveDissipation =
        package_core->Param<bool>("enforce_positive_dissipation");

  const auto bound_x1_interior =
      pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto bound_x2_interior =
      pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto bound_x3_interior =
      pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
      "electron_entropy"
  };
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

        const Real advected_total_entropy = two_temperature::RecoverAdvectedScalar(
            conservative(RHO, k, j, i), conservative(ENT, k, j, i));
        const Real advected_electron_entropy = two_temperature::RecoverAdvectedScalar(
            conservative(RHO, k, j, i), conservative(KEL, k, j, i));
        const Real recovered_total_entropy = primitive(ENT, k, j, i);

                primitive(KEL, k, j, i) = two_temperature::ComputeModelElectronEntropy(
                        kHeatingModel, kGamma, kGammaP, kGammaE,
                        primitive(RHO, k, j, i), primitive(ENY, k, j, i), state.bsq,
                        advected_total_entropy, recovered_total_entropy,
                        advected_electron_entropy, kFelConstant, kLimitKel, kRatioMin,
                        kRatioMax, kSuppressHighbHeat, kEnforcePositiveDissipation);
      });

  return TaskStatus::complete;
}
