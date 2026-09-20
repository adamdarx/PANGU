#ifndef PANGU_RIEMANN_NEWTONIAN_HYDRO_LLF_H_
#define PANGU_RIEMANN_NEWTONIAN_HYDRO_LLF_H_

#include <cmath>

#include "eos/newtonian_eos.h"

namespace pangu::riemann::newtonian_hydro {

using hydro::Primitive;
using parthenon::Real;

// Local Lax-Friedrichs flux with the larger one-sided acoustic speed.
struct LLF {
  KOKKOS_INLINE_FUNCTION static void Solve(const Primitive& left, const Primitive& right,
                                           const eos::NewtonianEOS& eos, const int direction,
                                           Real* interface_flux) {
    const auto left_state = eos.PrimitiveToConservedAndFlux(left, direction);
    const auto right_state = eos.PrimitiveToConservedAndFlux(right, direction);
    const int components = eos.HasEnergy() ? hydro::kIdealComponents : hydro::kIsothermalComponents;
    const Real left_sound = eos.SoundSpeed(left);
    const Real right_sound = eos.SoundSpeed(right);
    const Real left_normal = left.velocity[direction];
    const Real right_normal = right.velocity[direction];
    const Real speed = fmax(fabs(left_normal) + left_sound, fabs(right_normal) + right_sound);
    for (int n = 0; n < components; ++n) {
      interface_flux[n] = 0.5 * (left_state.flux[n] + right_state.flux[n]) -
                          0.5 * speed * (right_state.conserved[n] - left_state.conserved[n]);
    }
  }
};

} // namespace pangu::riemann::newtonian_hydro

#endif
