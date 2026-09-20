#ifndef PANGU_RIEMANN_NEWTONIAN_MHD_LLF_H_
#define PANGU_RIEMANN_NEWTONIAN_MHD_LLF_H_

// Copyright (c) 2020 James M. Stone and the AthenaK team.
// LLF algebra is derived from AthenaK under BSD-3-Clause; see LICENSE/NOTICE.

#include <cmath>

#include "eos/newtonian_mhd_eos.h"
#include "riemann/newtonian_mhd/electric_field.h"

namespace pangu::riemann::newtonian_mhd {

using mhd::InterfaceFlux;
using mhd::Primitive;
using parthenon::Real;

// Local Lax-Friedrichs flux with the larger one-sided fast magnetosonic speed.
struct LLF {
  KOKKOS_INLINE_FUNCTION static void Solve(const Primitive& left, const Primitive& right,
                                           const eos::NewtonianMHDEOS& eos, const int direction,
                                           InterfaceFlux& result) {
    const int tangent1 = (direction + 1) % 3;
    const int tangent2 = (direction + 2) % 3;
    const auto ul = eos.PrimitiveToConservedAndFlux(left, direction);
    const auto ur = eos.PrimitiveToConservedAndFlux(right, direction);
    const Real speed = fmax(fabs(left.velocity[direction]) + eos.FastSpeed(left, direction),
                            fabs(right.velocity[direction]) + eos.FastSpeed(right, direction));
    const int components = eos.HasEnergy() ? mhd::kIdealComponents : mhd::kIsothermalComponents;
    for (int n = 0; n < components; ++n) {
      result.fluid[n] =
          0.5 * (ul.flux[n] + ur.flux[n] - speed * (ur.conserved[n] - ul.conserved[n]));
    }
    const Real left_by_flux = left.magnetic[tangent1] * left.velocity[direction] -
                              left.magnetic[direction] * left.velocity[tangent1];
    const Real right_by_flux = right.magnetic[tangent1] * right.velocity[direction] -
                               right.magnetic[direction] * right.velocity[tangent1];
    const Real left_bz_flux = left.magnetic[tangent2] * left.velocity[direction] -
                              left.magnetic[direction] * left.velocity[tangent2];
    const Real right_bz_flux = right.magnetic[tangent2] * right.velocity[direction] -
                               right.magnetic[direction] * right.velocity[tangent2];
    const Real by_flux = 0.5 * (left_by_flux + right_by_flux -
                                speed * (right.magnetic[tangent1] - left.magnetic[tangent1]));
    const Real bz_flux = 0.5 * (left_bz_flux + right_bz_flux -
                                speed * (right.magnetic[tangent2] - left.magnetic[tangent2]));
    StoreElectricField(by_flux, bz_flux, result);
  }
};

} // namespace pangu::riemann::newtonian_mhd

#endif
