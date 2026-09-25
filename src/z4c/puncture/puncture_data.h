#ifndef PANGU_Z4C_PUNCTURE_PUNCTURE_DATA_H_
#define PANGU_Z4C_PUNCTURE_PUNCTURE_DATA_H_

#include <parthenon/parthenon.hpp>

namespace pangu::nr::puncture {

// Freely specifiable Bowen--York data for one asymptotically flat end.  Mass
// is the bare puncture mass; momentum and spin are Cartesian contravariant
// components in the conformally flat background metric.
struct Puncture {
  parthenon::Real mass = 0.0;
  parthenon::Real center[3]{};
  parthenon::Real momentum[3]{};
  parthenon::Real spin[3]{};
};

struct FreeData {
  parthenon::Real singular_psi = 1.0;
  parthenon::Real conformal_extrinsic[6]{};
  parthenon::Real conformal_extrinsic_squared = 0.0;
};

struct HamiltonianPoint {
  FreeData free{};
  parthenon::Real regular_correction = 0.0;
  parthenon::Real psi = 1.0;
  parthenon::Real source = 0.0;
};

} // namespace pangu::nr::puncture

#endif
