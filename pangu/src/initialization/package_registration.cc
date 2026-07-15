// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "initialization/package_registration.h"

#include <memory>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "initialization/variable_mnemonics.h"
#include "initialization/timestep_estimation.h"
#include "physics/heating_model.h"

namespace core {
std::shared_ptr<parthenon::StateDescriptor> Initialize(
    parthenon::ParameterInput *pin) {
  const auto package_core =
      std::make_shared<parthenon::StateDescriptor>("core");

  const auto kCflNumber = pin->GetOrAddReal("core", "cfl_number", 0.8);
  const auto kAdiabaticIndex =
      pin->GetOrAddReal("core", "adiabatic_index", 5. / 3.);
  const auto kRiemannSolver =
      pin->GetOrAddString("core", "riemann_solver", "laxf");
  const auto kLimiter =
      pin->GetOrAddString("core", "limiter", "ppm4");
  const auto kDensityFloor =
      pin->GetOrAddReal("core", "density_floor", 1.0e-6);
  const auto kEnergyFloor =
      pin->GetOrAddReal("core", "energy_floor", 1.0e-8);
  const auto kDensityFloorPow =
      pin->GetOrAddReal("core", "density_floor_pow", -1.5);
  const auto kEnergyFloorPow =
      pin->GetOrAddReal("core", "energy_floor_pow", -2.5);
  const auto kEnableB = pin->GetOrAddBoolean("core", "enable_B", false);
  const auto kEnableHeating = pin->GetOrAddBoolean("core", "enable_heating", false);
  const auto kSigmaMax = pin->GetOrAddReal("core", "sigma_max", 50);
  const auto kLorentzMax = pin->GetOrAddReal("core", "lorentz_max", 50);
  const auto kModelName =
      modelName(parseModel(
          pin->GetOrAddString("electron", "model", "constant")));
  const auto kFelConstant = pin->GetOrAddReal("electron", "fel_constant", 0.1);
  const auto kGammaE = pin->GetOrAddReal("electron", "gamma_e", 4. / 3.);
  const auto kGammaP = pin->GetOrAddReal("electron", "gamma_p", 5. / 3.);
  const auto kLimitKel = pin->GetOrAddBoolean("electron", "limit_kel", true);
  const auto kSuppressHighbHeat =
      pin->GetOrAddBoolean("electron", "suppress_highb_heat", false);
  const auto kEnforcePositiveDissipation =
      pin->GetOrAddBoolean("electron", "enforce_positive_dissipation", false);
  const auto kRatioMin = pin->GetOrAddReal("electron", "ratio_min", 0.001);
  const auto kRatioMax = pin->GetOrAddReal("electron", "ratio_max", 1000.0);
  const auto kFelInit = pin->GetOrAddReal("electron", "fel_0", 0.1);

  package_core->AddParam<>("cfl_number", kCflNumber);
  package_core->AddParam<>("adiabatic_index", kAdiabaticIndex);
  package_core->AddParam<>("riemann_solver", kRiemannSolver);
  package_core->AddParam<>("limiter", kLimiter);
  package_core->AddParam<>("density_floor", kDensityFloor);
  package_core->AddParam<>("energy_floor", kEnergyFloor);
  package_core->AddParam<>("density_floor_pow", kDensityFloorPow);
  package_core->AddParam<>("energy_floor_pow", kEnergyFloorPow);
  package_core->AddParam<>("enable_B", kEnableB);
  package_core->AddParam<>("enable_heating", kEnableHeating);
  package_core->AddParam<>("sigma_max", kSigmaMax);
  package_core->AddParam<>("lorentz_max", kLorentzMax);
  package_core->AddParam<>("model_name", kModelName);
  package_core->AddParam<>("fel_constant", kFelConstant);
  package_core->AddParam<>("gamma_e", kGammaE);
  package_core->AddParam<>("gamma_p", kGammaP);
  package_core->AddParam<>("limit_kel", kLimitKel);
  package_core->AddParam<>("suppress_highb_heat", kSuppressHighbHeat);
  package_core->AddParam<>("enforce_positive_dissipation", kEnforcePositiveDissipation);
  package_core->AddParam<>("ratio_min", kRatioMin);
  package_core->AddParam<>("ratio_max", kRatioMax);
  package_core->AddParam<>("fel_0", kFelInit);

  // Build primitive field name vector based on enabled modules.
  std::vector<std::string> fnames = {
      "density", "energy", "weighted_velocity"
  };
  if (kEnableB) fnames.push_back("magnetic_field");
  fnames.push_back("entropy");
  if (kEnableHeating) fnames.push_back("electron_entropy");
  package_core->AddParam<>("primitive_field_names", fnames);

  // Total number of components: density(1) + energy(1) + weighted_velocity(3)
  // + [magnetic_field(3)] + entropy(1) + [electron_entropy(1)]
  int n_components = 6;
  if (kEnableB) n_components += 3;
  if (kEnableHeating) n_components += 1;

  parthenon::Metadata m;
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
  package_core->AddField(std::string("density"), m);
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
  package_core->AddField(std::string("energy"), m);
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
  package_core->AddField(std::string("entropy"), m);
  if (kEnableHeating) {
    m = parthenon::Metadata(
        {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
    package_core->AddField(std::string("electron_entropy"), m);
  }
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost,
       parthenon::Metadata::Vector},
      std::vector<int>({3}));
  package_core->AddField(std::string("weighted_velocity"), m);
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost,
       parthenon::Metadata::OneCopy},
      std::vector<int>({3}));
  package_core->AddField(std::string("alfven"), m);
  if (kEnableB) {
    m = parthenon::Metadata(
        {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost,
         parthenon::Metadata::Vector},
        std::vector<int>({3}));
    package_core->AddField(std::string("magnetic_field"), m);
    m = parthenon::Metadata(
        {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost,
         parthenon::Metadata::OneCopy},
        std::vector<int>({3}));
    package_core->AddField(std::string("electric_field"), m);
  }
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::WithFluxes,
       parthenon::Metadata::Independent, parthenon::Metadata::FillGhost},
      std::vector<int>({n_components}));
  package_core->AddField(std::string("conservative"), m);
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::FillGhost,
       parthenon::Metadata::OneCopy});
  package_core->AddField(std::string("flag"), m);

  package_core->EstimateTimestepMesh = EstimateTimestepMesh;
  return package_core;
}
}

namespace metric {
std::shared_ptr<parthenon::StateDescriptor> Initialize(
    parthenon::ParameterInput *pin) {
  const auto package_metric =
      std::make_shared<parthenon::StateDescriptor>("metric");

  const auto kMetricType = pin->GetOrAddString("metric", "type", "minkowski");
  package_metric->AddParam<>("metric_type", kMetricType);

  // Kerr / excision parameters, read unconditionally by fixers and recovery.
  const auto kKerrA = pin->GetOrAddReal("metric", "a", 0.0);
  const auto kMksH = pin->GetOrAddReal("metric", "h", 0.0);
  const auto kExcise = pin->GetOrAddBoolean("metric", "excise", false);
  const auto kRExcise = pin->GetOrAddReal("metric", "r_excise", 1.0);
  const auto kDExcise = pin->GetOrAddReal("metric", "dexcise", 1.0e-8);
  const auto kPExcise = pin->GetOrAddReal("metric", "pexcise", 0.333e-12);
  package_metric->AddParam<>("a", kKerrA);
  package_metric->AddParam<>("h", kMksH);
  package_metric->AddParam<>("excise", kExcise);
  package_metric->AddParam<>("r_excise", kRExcise);
  package_metric->AddParam<>("dexcise", kDExcise);
  package_metric->AddParam<>("pexcise", kPExcise);

  parthenon::Metadata m;
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::OneCopy},
      std::vector<int>({4, 4, 4}));
  package_metric->AddField(std::string("covariant_metric"), m);

  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::OneCopy},
      std::vector<int>({4, 4, 4}));
  package_metric->AddField(std::string("contravariant_metric"), m);

  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::OneCopy},
      std::vector<int>({4}));
  package_metric->AddField(std::string("metric_determinant"), m);

  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::OneCopy},
      std::vector<int>({4, 4, 4}));
  package_metric->AddField(std::string("connection"), m);

  return package_metric;
}
}
