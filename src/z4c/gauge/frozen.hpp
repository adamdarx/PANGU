#ifndef PANGU_Z4C_GAUGE_FROZEN_HPP_
#define PANGU_Z4C_GAUGE_FROZEN_HPP_

#include <parthenon/parthenon.hpp>

namespace pangu::nr::gauge::frozen {
KOKKOS_INLINE_FUNCTION parthenon::Real LapseRhs() { return 0.0; }
KOKKOS_INLINE_FUNCTION parthenon::Real ShiftRhs() { return 0.0; }
} // namespace pangu::nr::gauge::frozen

#endif
