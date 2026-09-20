#ifndef PANGU_Z4C_TASK_CONTRIBUTION_H_
#define PANGU_Z4C_TASK_CONTRIBUTION_H_

#include "driver/contributor.h"

namespace pangu::nr {

// Z4c with optional synchronized GRHD or GRMHD matter.
driver::StageContribution SyncStageContribution();

} // namespace pangu::nr

#endif
