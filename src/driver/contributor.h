#ifndef PANGU_DRIVER_CONTRIBUTOR_H_
#define PANGU_DRIVER_CONTRIBUTOR_H_

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <parthenon/parthenon.hpp>

#include "app/problem_registry.h"

namespace pangu::driver {

struct StageBuildContext {
  parthenon::Mesh* mesh;
  parthenon::LowStorageIntegrator* integrator;
  const parthenon::SimTime* time;
  parthenon::BlockList_t* blocks;
  int stage;
  app::BlockFluxTask flux;
  app::BlockFluxCorrectionTask flux_correction;
};

using StageBuildFunction = parthenon::TaskCollection (*)(StageBuildContext&);
using StageSupportFunction = bool (*)(const StageBuildContext&);
using StageConfigureFunction = void (*)(parthenon::Mesh*, parthenon::LowStorageIntegrator&);
using StagePostStepFunction = void (*)(parthenon::Mesh*, const parthenon::SimTime&);

struct StageContribution {
  std::string_view name;
  int priority;
  StageSupportFunction supports;
  StageBuildFunction build;
  // Optional: adjusts the integrator once before evolution starts.
  StageConfigureFunction configure = nullptr;
  // Optional: runs after every completed step.
  StagePostStepFunction post_step = nullptr;
};

inline constexpr std::string_view stage_contribution_key = "driver/stage_contribution";

// Chooses the highest-priority contribution that supports the stage, or nullptr when the generic
// fallback graph should be built. Two supporting contributions sharing the highest priority make the
// assembly ambiguous regardless of registration order.
inline const StageContribution*
SelectStageContribution(const std::vector<const StageContribution*>& contributions,
                        const StageBuildContext& context) {
  const StageContribution* selected = nullptr;
  const StageContribution* tied = nullptr;
  for (const auto* candidate : contributions) {
    if (!candidate->supports(context))
      continue;
    if (selected == nullptr || candidate->priority > selected->priority) {
      selected = candidate;
      tied = nullptr;
    } else if (candidate->priority == selected->priority) {
      tied = candidate;
    }
  }
  if (tied != nullptr)
    throw std::logic_error("task contributions '" + std::string(selected->name) + "' and '" +
                           std::string(tied->name) + "' have the same stage priority");
  return selected;
}

} // namespace pangu::driver

#endif
