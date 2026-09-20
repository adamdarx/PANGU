#ifndef PANGU_RADIATION_TASK_CONTRIBUTION_H_
#define PANGU_RADIATION_TASK_CONTRIBUTION_H_

#include "driver/contributor.h"

namespace pangu::radiation {

parthenon::TaskCollection ContributeTransportStage(driver::StageBuildContext& context);
driver::StageContribution TransportStageContribution();

} // namespace pangu::radiation

#endif
