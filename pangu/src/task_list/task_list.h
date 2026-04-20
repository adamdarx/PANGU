// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

// This file in the src/task_list module defines task_list.h responsibilities for the Pangu
// runtime. It centers on amr_criteria to express core data flow, keep interfaces readable,
// and preserve predictable behavior across task coordination, recovery paths, and
// performance-sensitive execution.

#ifndef PANGU_SRC_TASKLIST_TASKLIST_H
#define PANGU_SRC_TASKLIST_TASKLIST_H

#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "constrained_transport/constrained_transport.h"
#include "initialization/package_registration.h"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "numerical_correction/primitive_fixer.h"
#include "numerical_correction/recovery_fixer.h"
#include "prolong_restrict/prolong_restrict.hpp"
#include "recovery/invertor.h"
#include "riemann_solver/conservative.h"
#include "riemann_solver/lax_friedrichs_flux.h"

#include <memory>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon::driver::prelude;

// Multi-stage simulation driver for ideal SRMHD and GRMHD task graphs.
class Simulator : public MultiStageDriver {
 public:
  // Constructs the simulator driver from Parthenon runtime objects.
  Simulator(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm);

  // Builds the staged task collection for one RK stage.
  TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);
};

// Initializes one mesh block with problem-specific primitive variables.
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);

// Creates and registers runtime packages used by the simulation.
parthenon::Packages_t ProcessPackages(
    std::unique_ptr<parthenon::ParameterInput> &pin);

#endif  // PANGU_SRC_TASKLIST_TASKLIST_H
