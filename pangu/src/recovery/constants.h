// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_RECOVERY_CONSTANTS_H
#define PANGU_SRC_RECOVERY_CONSTANTS_H

#include <cstdint>
#include <parthenon/package.hpp>

#include "initialization/variable_mnemonics.h"

using namespace parthenon::package::prelude;

inline constexpr int NPRIM_RECV = 8;
inline constexpr int64_t MAX_NEWT_ITER = 30;
inline constexpr double NEWT_TOL = 1e-10;
inline constexpr double MIN_NEWT_TOL = 1e-10;
inline constexpr int64_t EXTRA_NEWT_ITER = 2;
inline constexpr double NEWT_TOL2 = 1.0e-15;
inline constexpr double MIN_NEWT_TOL2 = 1.0e-10;
inline constexpr double W_TOO_BIG = 1.e20;
inline constexpr double UTSQ_TOO_BIG = 1.e20;
inline constexpr double FAIL_VAL = 1.e30;

#endif  // PANGU_SRC_RECOVERY_CONSTANTS_H
