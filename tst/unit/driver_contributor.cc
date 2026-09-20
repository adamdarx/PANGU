#include <stdexcept>
#include <string>
#include <vector>

#include "driver/contributor.h"

namespace {

using pangu::driver::SelectStageContribution;
using pangu::driver::StageBuildContext;
using pangu::driver::StageContribution;

bool SupportsFirstTwoStages(const StageBuildContext& context) { return context.stage <= 2; }
bool SupportsFirstStage(const StageBuildContext& context) { return context.stage == 1; }

StageBuildContext Stage(int stage) { return {nullptr, nullptr, nullptr, nullptr, stage, {}, {}}; }

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void RequireTie(const std::vector<const StageContribution*>& contributions, int stage) {
  try {
    (void)SelectStageContribution(contributions, Stage(stage));
  } catch (const std::logic_error& error) {
    if (std::string(error.what()).find("same stage priority") != std::string::npos)
      return;
    throw;
  }
  throw std::runtime_error("equal-priority contributions were accepted");
}

} // namespace

int main() {
  const StageContribution fluid{"fluid", 50, SupportsFirstTwoStages, nullptr};
  const StageContribution spacetime{"spacetime", 100, SupportsFirstStage, nullptr};
  const StageContribution twin{"twin", 50, SupportsFirstTwoStages, nullptr};

  const std::vector<const StageContribution*> registered{&fluid, &spacetime};
  Require(SelectStageContribution(registered, Stage(1)) == &spacetime,
          "higher-priority contribution was not selected");
  Require(SelectStageContribution(registered, Stage(2)) == &fluid,
          "only supporting contribution was not selected");
  Require(SelectStageContribution(registered, Stage(3)) == nullptr,
          "unsupported stage did not fall back to the generic graph");

  // A tie below the winning priority is not ambiguous, whatever the registration order.
  Require(SelectStageContribution({&fluid, &twin, &spacetime}, Stage(1)) == &spacetime,
          "lower-priority tie blocked the winning contribution");
  RequireTie({&fluid, &twin, &spacetime}, 2);
  return 0;
}
