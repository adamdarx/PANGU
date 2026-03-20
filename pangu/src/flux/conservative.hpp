#pragma once

#include <memory>
#include <string>
#include <vector>

#include <basic_types.hpp>
#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"
#include "../physics/stress_tensor.hpp"

parthenon::TaskStatus CalculateConservative(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  const auto meshblock_pointer = resource->GetBlockPointer();
  const auto package = meshblock_pointer->packages.Get("PANGU");
  const auto adiabatic_index = package->Param<Real>("AdiabaticIndex");

  const auto bound_x1 =
      meshblock_pointer->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto bound_x2 =
      meshblock_pointer->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto bound_x3 =
      meshblock_pointer->cellbounds.GetBoundsK(IndexDomain::interior);

  PackIndexMap primitive_index_map;
  const std::vector<std::string> primitive_tags = {
      "Density", "Energy", "WeightedVelocity", "MagneticField"};
  const auto primitive = resource->PackVariables(primitive_tags, primitive_index_map);

  PackIndexMap conservative_index_map;
  const std::vector<std::string> conservative_tags = {"Conservative"};
  auto conservative =
      resource->PackVariablesAndFluxes(conservative_tags, conservative_index_map);

  meshblock_pointer->par_for(
      PARTHENON_AUTO_LABEL, bound_x3.s, bound_x3.e, bound_x2.s, bound_x2.e,
      bound_x1.s, bound_x1.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto squared_weighted_velocity =
            Kokkos::pow(primitive(WeightedVelocityX1, k, j, i), 2) +
            Kokkos::pow(primitive(WeightedVelocityX2, k, j, i), 2) +
            Kokkos::pow(primitive(WeightedVelocityX3, k, j, i), 2);
        const auto squared_lorentz_factor = 1 + squared_weighted_velocity;
        const auto lorentz_factor = Kokkos::sqrt(squared_lorentz_factor);
        parthenon::Real directed_energy_momentum_tensor[4];
        const parthenon::Real primitive_c_array[PrimitiveVariableNumber] = {
            primitive(DensityIndex, k, j, i), primitive(EnergyIndex, k, j, i),
            primitive(WeightedVelocityX1, k, j, i),
            primitive(WeightedVelocityX2, k, j, i),
            primitive(WeightedVelocityX3, k, j, i),
            primitive(MagneticFieldX1, k, j, i),
            primitive(MagneticFieldX2, k, j, i),
            primitive(MagneticFieldX3, k, j, i),
        };

        CalculateEnergyMomentumTensorInDir(
            adiabatic_index, primitive_c_array, X0DIR,
            directed_energy_momentum_tensor);
        conservative(DensityIndex, k, j, i) =
            primitive(DensityIndex, k, j, i) * lorentz_factor;
        conservative(EnergyIndex, k, j, i) =
            directed_energy_momentum_tensor[0] +
            conservative(DensityIndex, k, j, i);
        conservative(WeightedVelocityX1, k, j, i) =
            directed_energy_momentum_tensor[1];
        conservative(WeightedVelocityX2, k, j, i) =
            directed_energy_momentum_tensor[2];
        conservative(WeightedVelocityX3, k, j, i) =
            directed_energy_momentum_tensor[3];

        conservative(MagneticFieldX1, k, j, i) =
            primitive(MagneticFieldX1, k, j, i);
        conservative(MagneticFieldX2, k, j, i) =
            primitive(MagneticFieldX2, k, j, i);
        conservative(MagneticFieldX3, k, j, i) =
            primitive(MagneticFieldX3, k, j, i);
      });

  return TaskStatus::complete;
}