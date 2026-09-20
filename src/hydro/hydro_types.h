#ifndef PANGU_HYDRO_HYDRO_TYPES_H_
#define PANGU_HYDRO_HYDRO_TYPES_H_

#include <parthenon/parthenon.hpp>

#include "reconstruct/registry.h"

namespace pangu::hydro {

using parthenon::Real;

inline constexpr int IDN = 0;
inline constexpr int IM1 = 1;
inline constexpr int IM2 = 2;
inline constexpr int IM3 = 3;
inline constexpr int IEN = 4;
inline constexpr int IV1 = 1;
inline constexpr int IV2 = 2;
inline constexpr int IV3 = 3;
inline constexpr int IPR = 4;
inline constexpr int kIdealComponents = 5;
inline constexpr int kIsothermalComponents = 4;

enum class EosMode : int { ideal = 0, isothermal = 1 };
using Reconstruction = reconstruct::Method;

struct Primitive {
  Real density;
  Real velocity[3];
  Real pressure;
};

struct FluxState {
  Real conserved[kIdealComponents];
  Real flux[kIdealComponents];
};

KOKKOS_INLINE_FUNCTION
int MomentumIndex(const int direction) { return IM1 + direction; }

} // namespace pangu::hydro

#endif
