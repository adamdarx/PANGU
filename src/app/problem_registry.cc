#include "app/problem_registry.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <type_traits>

#include "geometry_assembly.h"
#include "hydro/hydro_package.h"
#include "mhd/mhd_package.h"
#include "pangu_config.h"
#include "pgen/hydro_problems.h"
#include "pgen/mhd_problems.h"
#include "pgen/numerical_relativity.h"
#include "pgen/radiation_beam.h"
#include "pgen/radiation_diffusion.h"
#include "pgen/radiation_hohlraum.h"
#include "pgen/radiation_linear_wave.h"
#include "pgen/radiation_relax.h"
#include "pgen/radiation_shadow.h"
#include "utils/error_checking.hpp"
#include "z4c/evolution/package.h"

namespace pangu::app {
namespace {

struct ProblemDefinition {
  void (*generator)(parthenon::MeshBlock *, parthenon::ParameterInput *);
  void (*after_loop)(parthenon::Mesh *, parthenon::ParameterInput *,
                     parthenon::SimTime &);
  BlockFluxTask flux_function;
  BlockFluxCorrectionTask first_order_correction;
  BlockSourceTask source_function;
  PackageInitializer package_initializer;
  PackageInitializer coupled_package_initializer = nullptr;
};

const std::map<std::string, ProblemDefinition> &Registry() {
  static std::map<std::string, ProblemDefinition> registry{
      {"hydro_advection",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"sod",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"linear_wave",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"blast",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"kh",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"rt",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"sr_shock",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"sr_linear_wave",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"sr_blast",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"gr_linear_wave",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"bondi_cks",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"bondi_mks",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"gr_torus",
       {pgen::HydroProblem, pgen::HydroAfterLoop,
        hydro::CalculateFluxesBlockTask,
        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
        hydro::ApplySourcesBlockTask, hydro::Initialize}},
      {"mhd_shock",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"brio_wu",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"sr_mhd_shock",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"gr_mhd_shock",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"sr_mhd_linear_wave",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"sr_mhd_modes",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"electron_hubble",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"electron_noh",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"gr_mhd_linear_wave",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"gr_monopole",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"magnetised_bondi_mks",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"gr_torus_sane",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"gr_chakrabarti_torus_sane",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"gr_chakrabarti_torus",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"cpaw",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"field_loop",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
      {"orszag_tang",
       {pgen::MHDProblem, pgen::MHDAfterLoop, mhd::CalculateFluxesBlockTask,
        mhd::ApplyFirstOrderFluxCorrectionBlockTask, mhd::ApplySourcesBlockTask,
        mhd::Initialize}},
  };
  registry.emplace(
      "rad_beam",
      ProblemDefinition{pgen::RadiationBeam, nullptr,
                        hydro::CalculateFluxesBlockTask,
                        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
                        hydro::ApplySourcesBlockTask, hydro::Initialize});
  registry.emplace(
      "rad_linear_wave",
      ProblemDefinition{pgen::RadiationLinearWave, nullptr,
                        hydro::CalculateFluxesBlockTask,
                        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
                        hydro::ApplySourcesBlockTask, hydro::Initialize});
  registry.emplace(
      "rad_diffusion",
      ProblemDefinition{pgen::RadiationDiffusion, nullptr,
                        hydro::CalculateFluxesBlockTask,
                        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
                        hydro::ApplySourcesBlockTask, hydro::Initialize});
  registry.emplace(
      "rad_hohlraum",
      ProblemDefinition{pgen::RadiationHohlraum, nullptr,
                        hydro::CalculateFluxesBlockTask,
                        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
                        hydro::ApplySourcesBlockTask, hydro::Initialize});
  registry.emplace(
      "rad_relax",
      ProblemDefinition{pgen::RadiationRelaxation, nullptr,
                        hydro::CalculateFluxesBlockTask,
                        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
                        hydro::ApplySourcesBlockTask, hydro::Initialize});
  registry.emplace(
      "rad_shadow",
      ProblemDefinition{pgen::RadiationShadow, nullptr,
                        hydro::CalculateFluxesBlockTask,
                        hydro::ApplyFirstOrderFluxCorrectionBlockTask,
                        hydro::ApplySourcesBlockTask, hydro::Initialize});
  // Make the condition dependent on a template parameter so fixed-background
  // assemblies do not emit references to pgen functions whose translation
  // unit is intentionally absent from those builds.
  [](auto synchronized, auto &registry) {
    if constexpr (decltype(synchronized)::value) {
      static const bool inserted_sync_grhd_shock =
          registry
              .emplace("sync_grhd_shock",
                       ProblemDefinition{
                           pgen::SyncGRHDProblem, pgen::HydroAfterLoop,
                           hydro::CalculateFluxesBlockTask,
                           hydro::ApplyFirstOrderFluxCorrectionBlockTask,
                           hydro::ApplySourcesBlockTask, nr::Initialize,
                           hydro::Initialize})
              .second;
      static const bool inserted_sync_grhd_linear_wave =
          registry
              .emplace("sync_grhd_linear_wave",
                       ProblemDefinition{
                           pgen::SyncGRHDProblem, pgen::HydroAfterLoop,
                           hydro::CalculateFluxesBlockTask,
                           hydro::ApplyFirstOrderFluxCorrectionBlockTask,
                           hydro::ApplySourcesBlockTask, nr::Initialize,
                           hydro::Initialize})
              .second;
      static const bool inserted_sync_grhd_tov =
          registry
              .emplace("sync_grhd_tov",
                       ProblemDefinition{
                           pgen::SyncGRHDTOV, nullptr,
                           hydro::CalculateFluxesBlockTask,
                           hydro::ApplyFirstOrderFluxCorrectionBlockTask,
                           hydro::ApplySourcesBlockTask, nr::Initialize,
                           hydro::Initialize})
              .second;
      static const bool inserted_sync_grmhd_linear_wave =
          registry
              .emplace(
                  "sync_grmhd_linear_wave",
                  ProblemDefinition{pgen::SyncGRMHDProblem, nullptr,
                                    mhd::CalculateFluxesBlockTask,
                                    mhd::ApplyFirstOrderFluxCorrectionBlockTask,
                                    mhd::ApplySourcesBlockTask, nr::Initialize,
                                    mhd::Initialize})
              .second;
      static const bool inserted_sync_grmhd_tov =
          registry
              .emplace(
                  "sync_grmhd_tov",
                  ProblemDefinition{pgen::SyncGRMHDTOV, nullptr,
                                    mhd::CalculateFluxesBlockTask,
                                    mhd::ApplyFirstOrderFluxCorrectionBlockTask,
                                    mhd::ApplySourcesBlockTask, nr::Initialize,
                                    mhd::Initialize})
              .second;
      static const bool inserted_sync_grmhd_boosted_puncture =
          registry
              .emplace(
                  "sync_grmhd_boosted_puncture",
                  ProblemDefinition{pgen::SyncGRMHDBoostedPuncture, nullptr,
                                    mhd::CalculateFluxesBlockTask,
                                    mhd::ApplyFirstOrderFluxCorrectionBlockTask,
                                    mhd::ApplySourcesBlockTask, nr::Initialize,
                                    mhd::Initialize})
              .second;
      static const bool inserted_minkowski =
          registry
              .emplace("nr_minkowski",
                       ProblemDefinition{pgen::NumericalRelativityMinkowski,
                                         pgen::NumericalRelativityAfterLoop,
                                         nullptr, nullptr, nullptr,
                                         nr::Initialize})
              .second;
      static const bool inserted_schwarzschild =
          registry
              .emplace("nr_schwarzschild",
                       ProblemDefinition{pgen::NumericalRelativitySchwarzschild,
                                         nullptr, nullptr, nullptr, nullptr,
                                         nr::Initialize})
              .second;
      static const bool inserted_schwarzschild_ks =
          registry
              .emplace("nr_schwarzschild_ks",
                       ProblemDefinition{
                           pgen::NumericalRelativitySchwarzschildKerrSchild,
                           nullptr, nullptr, nullptr, nullptr, nr::Initialize})
              .second;
      static const bool inserted_boosted_puncture =
          registry
              .emplace("nr_boosted_puncture",
                       ProblemDefinition{
                           pgen::NumericalRelativityBoostedPuncture, nullptr,
                           nullptr, nullptr, nullptr, nr::Initialize})
              .second;
      static const bool inserted_puncture =
          registry
              .emplace("nr_puncture",
                       ProblemDefinition{pgen::NumericalRelativityPuncture,
                                         nullptr, nullptr, nullptr, nullptr,
                                         nr::Initialize})
              .second;
      static const bool inserted_two_punctures =
          registry
              .emplace("nr_two_punctures",
                       ProblemDefinition{pgen::NumericalRelativityPuncture,
                                         nullptr, nullptr, nullptr, nullptr,
                                         nr::Initialize})
              .second;
      static const bool inserted_linear_wave =
          registry
              .emplace("nr_linear_wave",
                       ProblemDefinition{
                           pgen::NumericalRelativityLinearWave,
                           pgen::NumericalRelativityLinearWaveAfterLoop,
                           nullptr, nullptr, nullptr, nr::Initialize})
              .second;
      static const bool inserted_gauge_wave =
          registry
              .emplace(
                  "nr_gauge_wave",
                  ProblemDefinition{pgen::NumericalRelativityGaugeWave,
                                    pgen::NumericalRelativityGaugeWaveAfterLoop,
                                    nullptr, nullptr, nullptr, nr::Initialize})
              .second;
      static const bool inserted_robust_stability =
          registry
              .emplace("nr_robust_stability",
                       ProblemDefinition{
                           pgen::NumericalRelativityRobustStability,
                           pgen::NumericalRelativityRobustStabilityAfterLoop,
                           nullptr, nullptr, nullptr, nr::Initialize})
              .second;
      [[maybe_unused]] const bool all_registered =
          inserted_minkowski && inserted_sync_grhd_shock &&
          inserted_sync_grhd_linear_wave && inserted_sync_grhd_tov &&
          inserted_sync_grmhd_tov && inserted_sync_grmhd_boosted_puncture &&
          inserted_sync_grmhd_linear_wave && inserted_schwarzschild &&
          inserted_schwarzschild_ks && inserted_boosted_puncture &&
          inserted_puncture && inserted_two_punctures && inserted_linear_wave &&
          inserted_gauge_wave && inserted_robust_stability;
    }
  }(std::bool_constant<geometry::ConfiguredGeometryIsSynchronized()>{},
    registry);
  return registry;
}

template <bool Synchronized>
void ConfigureSynchronizedProblem(
    [[maybe_unused]] const std::string &problem_id,
    [[maybe_unused]] parthenon::ApplicationInput *app_input) {
  if constexpr (Synchronized) {
    pgen::RegisterNumericalRelativityReflectingBoundaries(app_input);
    pgen::RegisterNumericalRelativityExtrapolationBoundaries(app_input);
    pgen::RegisterNumericalRelativityOutflowBoundaries(app_input);
    // Parthenon resolves the adaptive hierarchy after each problem-generator
    // pass. Reinitialize analytic face fields once on the final hierarchy.
    if (problem_id == "sync_grmhd_linear_wave")
      app_input->PostInitialization = pgen::SyncGRMHDProblem;
    if (problem_id == "sync_grmhd_tov")
      app_input->PostInitialization = pgen::SyncGRMHDTOV;
  }
}

void ConfigureRadiationBoundaries(
    [[maybe_unused]] const std::string &problem_id,
    [[maybe_unused]] parthenon::ApplicationInput *app_input) {
  if (problem_id == "rad_hohlraum")
    pgen::RegisterRadiationHohlraumBoundaries(app_input);
  if (problem_id == "rad_shadow")
    pgen::RegisterRadiationShadowBoundaries(app_input);
}

} // namespace

void ConfigureProblem(const std::string &problem_id,
                      parthenon::ApplicationInput *app_input) {
  const auto it = Registry().find(problem_id);
  if (it == Registry().end()) {
    std::ostringstream msg;
    msg << "Unknown problem_id='" << problem_id << "'. Registered problems:";
    for (const auto &entry : Registry())
      msg << ' ' << entry.first;
    PARTHENON_FAIL(msg.str());
  }
  app_input->ProblemGenerator = it->second.generator;
  app_input->UserWorkAfterLoop = it->second.after_loop;
  const bool synchronized_geometry =
      geometry::ConfiguredGeometryIsSynchronized();
  const bool numerical_relativity_problem = problem_id.rfind("nr_", 0) == 0;
  const bool synchronized_matter_problem = problem_id.rfind("sync_", 0) == 0;
  const bool synchronized_problem =
      numerical_relativity_problem || synchronized_matter_problem;
  PARTHENON_REQUIRE(
      synchronized_problem == synchronized_geometry,
      synchronized_problem
          ? "synchronized-spacetime problems require -DMETRIC=z4c -DMODE=sync"
          : "a z4c+sync build accepts only numerical-relativity or "
            "synchronized-matter "
            "ProblemTypes");
  ConfigureSynchronizedProblem<geometry::ConfiguredGeometryIsSynchronized()>(
      problem_id, app_input);
  if (problem_id == "gr_torus")
    pgen::RegisterTorusBoundaries(app_input);
  if (problem_id == "bondi_cks" || problem_id == "bondi_mks")
    pgen::RegisterBondiBoundaries(app_input);
  if (problem_id == "gr_monopole")
    pgen::RegisterMonopoleBoundaries(app_input);
  if (problem_id == "magnetised_bondi_mks") {
    pgen::RegisterMagnetisedBondiBoundaries(app_input);
    app_input->MeshPostProblemGenerator = pgen::NormalizeMagnetisedBondi;
  }
  if (problem_id == "gr_torus_sane") {
    app_input->MeshPostProblemGenerator = pgen::NormalizeSaneTorus;
  }
  if (problem_id == "gr_chakrabarti_torus" ||
      problem_id == "gr_chakrabarti_torus_sane") {
    pgen::RegisterChakrabartiTorusBoundaries(app_input);
    app_input->MeshPostProblemGenerator = pgen::NormalizeSaneTorus;
  }
  ConfigureRadiationBoundaries(problem_id, app_input);
}

std::vector<std::string> RegisteredProblems() {
  std::vector<std::string> names;
  names.reserve(Registry().size());
  for (const auto &entry : Registry())
    names.push_back(entry.first);
  return names;
}

BlockFluxTask GetBlockFluxTask(const std::string &problem_id) {
  const auto definition = Registry().find(problem_id);
  if (definition == Registry().end()) {
    PARTHENON_FAIL("No flux function registered for problem_id='" + problem_id +
                   "'");
  }
  return definition->second.flux_function;
}

BlockFluxCorrectionTask
GetBlockFluxCorrectionTask(const std::string &problem_id) {
  const auto definition = Registry().find(problem_id);
  if (definition == Registry().end()) {
    PARTHENON_FAIL("No first-order correction registry entry for problem_id='" +
                   problem_id + "'");
  }
  return definition->second.first_order_correction;
}

BlockSourceTask GetBlockSourceTask(const std::string &problem_id) {
  const auto definition = Registry().find(problem_id);
  if (definition == Registry().end()) {
    PARTHENON_FAIL("No source registry entry for problem_id='" + problem_id +
                   "'");
  }
  return definition->second.source_function;
}

PackageInitializer GetPackageInitializer(const std::string &problem_id) {
  const auto definition = Registry().find(problem_id);
  if (definition == Registry().end()) {
    PARTHENON_FAIL("No package initializer registered for problem_id='" +
                   problem_id + "'");
  }
  return definition->second.package_initializer;
}

PackageInitializer GetCoupledPackageInitializer(const std::string &problem_id) {
  const auto definition = Registry().find(problem_id);
  if (definition == Registry().end()) {
    PARTHENON_FAIL(
        "No coupled package initializer registered for problem_id='" +
        problem_id + "'");
  }
  return definition->second.coupled_package_initializer;
}

} // namespace pangu::app
