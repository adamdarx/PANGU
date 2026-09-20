#ifndef PANGU_MHD_TASK_CONTRIBUTION_H_
#define PANGU_MHD_TASK_CONTRIBUTION_H_

#include "driver/contributor.h"

namespace pangu::mhd {

// Fixed-background GRMHD, or non-GR MHD on one uniform partition.
driver::StageContribution MHDStageContribution();

} // namespace pangu::mhd

#endif
