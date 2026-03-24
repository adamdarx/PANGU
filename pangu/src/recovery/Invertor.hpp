
#pragma once
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

enum Mnemonic { RHO, UU, U1, U2, U3, B1, B2, B3 };
const int64_t MAX_NEWT_ITER = (30);
const double NEWT_TOL = (1e-10);
const double MIN_NEWT_TOL = (1e-10);
const int64_t EXTRA_NEWT_ITER = (2);
const double NEWT_TOL2 = (1.0e-15);
const double MIN_NEWT_TOL2 = (1.0e-10);
const double W_TOO_BIG = (1.e20);
const double UTSQ_TOO_BIG = (1.e20);
const double FAIL_VAL = (1.e30);