#ifndef PANGU_MHD_MHD_TYPES_H_
#define PANGU_MHD_MHD_TYPES_H_

#include <string>
#include <utility>

#include <parthenon/parthenon.hpp>

namespace pangu::mhd {

using parthenon::Real;

enum Component : int { IDN = 0, IM1 = 1, IM2 = 2, IM3 = 3, IEN = 4 };
enum PrimitiveComponent : int { IV1 = 1, IV2 = 2, IV3 = 3, IPR = 4 };
constexpr int kIsothermalComponents = 4;
constexpr int kIdealComponents = 5;

enum class EosMode : int { ideal = 0, isothermal = 1 };

KOKKOS_INLINE_FUNCTION
constexpr int MomentumIndex(const int direction) { return IM1 + direction; }

struct Primitive {
  Real density;
  Real velocity[3];
  Real pressure;
  Real magnetic[3];
};

struct ConservedFlux {
  Real conserved[kIdealComponents];
  Real flux[kIdealComponents];
};

struct InterfaceFlux {
  Real fluid[kIdealComponents];
  // Physical electric field E = -(v x B), cyclic transverse components.
  Real electric_t1;
  Real electric_t2;
};

#define PANGU_MHD_VARIABLE(varname, label)                                                         \
  struct varname : public parthenon::variable_names::base_t<false> {                               \
    template <class... Ts>                                                                         \
    KOKKOS_INLINE_FUNCTION varname(Ts&&... args)                                                   \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}                   \
    static std::string name() { return label; }                                                    \
  }

PANGU_MHD_VARIABLE(BFace, "mhd.b_face");

#undef PANGU_MHD_VARIABLE

} // namespace pangu::mhd

#endif
