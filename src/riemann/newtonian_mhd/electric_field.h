#ifndef PANGU_RIEMANN_NEWTONIAN_MHD_ELECTRIC_FIELD_H_
#define PANGU_RIEMANN_NEWTONIAN_MHD_ELECTRIC_FIELD_H_

// Copyright (c) 2020 James M. Stone and the AthenaK team.
// LLF and HLLE algebra is derived from AthenaK under BSD-3-Clause; see LICENSE/NOTICE.

#include <cmath>

#include "eos/newtonian_mhd_eos.h"

namespace pangu::riemann::newtonian_mhd {

using mhd::InterfaceFlux;
using mhd::Primitive;
using parthenon::Real;

// Stores the face electric field from the transverse induction fluxes.
KOKKOS_INLINE_FUNCTION
void StoreElectricField(const Real by_flux, const Real bz_flux, InterfaceFlux& result) {
  // In local normal/tangent coordinates: F(By)=-E_t2 and F(Bz)=E_t1.
  result.electric_t1 = bz_flux;
  result.electric_t2 = -by_flux;
}

} // namespace pangu::riemann::newtonian_mhd

#endif
