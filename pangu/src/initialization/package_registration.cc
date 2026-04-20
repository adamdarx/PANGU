// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/initialization module defines package_registration.cc
// responsibilities for the Pangu runtime. It centers on initialization to express core data
// flow, keep interfaces readable, and preserve predictable behavior across task
// coordination, recovery paths, and performance-sensitive execution.

#include "initialization/package_registration.h"

#include <cmath>
#include <memory>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "initialization/timestep_estimation.h"
#include "mesh/refinement_criteria.h"

namespace core {
std::shared_ptr<parthenon::StateDescriptor> Initialize(
    parthenon::ParameterInput *pin) {
  const auto package_core =
      std::make_shared<parthenon::StateDescriptor>("core");

  const auto kCflNumber = pin->GetOrAddReal("core", "cfl_number", 0.8);
  const auto kAdiabaticIndex =
      pin->GetOrAddReal("core", "adiabatic_index", 5. / 3.);
  const auto kQFactorFloor =
      pin->GetOrAddReal("core", "q_factor_floor", 0.3);
  const auto kQFactorCeiling =
      pin->GetOrAddReal("core", "q_factor_ceiling", 0.03);
  const auto kDensityFloor =
      pin->GetOrAddReal("core", "density_floor", 1.0e-6);
  const auto kEnergyFloor = 
      pin->GetOrAddReal("core", "energy_floor", 1.0e-8);
  const auto kDensityFloorPow =
      pin->GetOrAddReal("core", "density_floor_pow", -1.5);
  const auto kEnergyFloorPow =
      pin->GetOrAddReal("core", "energy_floor_pow", -2.5);
  const auto kSigmaMax = pin->GetOrAddReal("core", "sigma_max", 50);
  const auto kLorentzMax = pin->GetOrAddReal("core", "lorentz_max", 50);

  package_core->AddParam<>("cfl_number", kCflNumber);
  package_core->AddParam<>("adiabatic_index", kAdiabaticIndex);
  package_core->AddParam<>("q_factor_floor", kQFactorFloor);
  package_core->AddParam<>("q_factor_ceiling", kQFactorCeiling);
  package_core->AddParam<>("density_floor", kDensityFloor);
  package_core->AddParam<>("energy_floor", kEnergyFloor);
  package_core->AddParam<>("density_floor_pow", kDensityFloorPow);
  package_core->AddParam<>("energy_floor_pow", kEnergyFloorPow);
  package_core->AddParam<>("sigma_max", kSigmaMax);
  package_core->AddParam<>("lorentz_max", kLorentzMax);

  parthenon::Metadata m;
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
  package_core->AddField(std::string("density"), m);
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
  package_core->AddField(std::string("energy"), m);
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
  package_core->AddField(std::string("q_factor"), m);
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost,
       parthenon::Metadata::Vector},
      std::vector<int>({3}));
  package_core->AddField(std::string("weighted_velocity"), m);
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost},
      std::vector<int>({3}));
  package_core->AddField(std::string("alfven"), m);
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost,
       parthenon::Metadata::Vector},
      std::vector<int>({3}));
  package_core->AddField(std::string("magnetic_field"), m);
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost},
      std::vector<int>({3}));
  package_core->AddField(std::string("electric_field"), m);
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::WithFluxes,
       parthenon::Metadata::Independent, parthenon::Metadata::FillGhost},
      std::vector<int>({8}));
  package_core->AddField(std::string("conservative"), m);
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
  package_core->AddField(std::string("flag"), m);

  package_core->CheckRefinementBlock = CheckRefinement;
  package_core->EstimateTimestepBlock = EstimateTimestepBlock;
  return package_core;
}
}  

namespace metric {
std::shared_ptr<parthenon::StateDescriptor> Initialize(
    parthenon::ParameterInput *pin) {
  const auto package_metric =
      std::make_shared<parthenon::StateDescriptor>("metric");

  const auto kMetricName = pin->GetOrAddString("metric", "name", "Minkowski");
  package_metric->AddParam<>("metric_name", kMetricName);

  parthenon::Metadata m;
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost},
      std::vector<int>({4, 4, 4}));
  package_metric->AddField(std::string("covariant_metric"), m);

  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost},
      std::vector<int>({4, 4, 4}));
  package_metric->AddField(std::string("contravariant_metric"), m);

  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost},
      std::vector<int>({4}));
  package_metric->AddField(std::string("metric_determinant"), m);

  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost},
      std::vector<int>({4, 4, 4}));
  package_metric->AddField(std::string("connection"), m);

  return package_metric;
}
}  
