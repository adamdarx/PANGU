#pragma once

#include <parthenon/parthenon.hpp>

#include "reconstruct/hydro_reconstruction.h"

namespace pangu::mhd {

struct PassiveTransportCoefficients {
  parthenon::Real left = 0.0;
  parthenon::Real right = 0.0;
};

template <int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION parthenon::Real ReadPassive(const Pack& pack, const int component,
                                                   const int k, const int j, const int i,
                                                   const int offset) {
  if constexpr (Direction == 0)
    return pack(component, k, j, i + offset);
  if constexpr (Direction == 1)
    return pack(component, k, j + offset, i);
  return pack(component, k + offset, j, i);
}

template <int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION parthenon::Real ReadPassiveMesh(const Pack& pack, const int block,
                                                       const int component, const int k,
                                                       const int j, const int i, const int offset) {
  if constexpr (Direction == 0)
    return pack(block, component, k, j, i + offset);
  if constexpr (Direction == 1)
    return pack(block, component, k, j + offset, i);
  return pack(block, component, k + offset, j, i);
}

template <int Direction, hydro::Reconstruction Method, typename Pack>
KOKKOS_INLINE_FUNCTION void ReconstructPassive(const Pack& primitive, const int component,
                                               const int k, const int j, const int i,
                                               parthenon::Real& left, parthenon::Real& right) {
  reconstruct::ReconstructFaceStencil<Method>(
      [&](const int offset) { return ReadPassive<Direction>(primitive, component, k, j, i, offset); },
      left, right);
}

template <int Direction, hydro::Reconstruction Method, typename Pack>
KOKKOS_INLINE_FUNCTION void
ReconstructPassiveMesh(const Pack& primitive, const int block, const int component, const int k,
                       const int j, const int i, parthenon::Real& left, parthenon::Real& right) {
  reconstruct::ReconstructFaceStencil<Method>(
      [&](const int offset) {
        return ReadPassiveMesh<Direction>(primitive, block, component, k, j, i, offset);
      },
      left, right);
}

KOKKOS_INLINE_FUNCTION PassiveTransportCoefficients
LLFPassiveTransport(const parthenon::Real density_left, const parthenon::Real density_right,
                    const parthenon::Real mass_flux_left, const parthenon::Real mass_flux_right,
                    const parthenon::Real speed) {
  return {0.5 * (mass_flux_left + speed * density_left),
          0.5 * (mass_flux_right - speed * density_right)};
}

KOKKOS_INLINE_FUNCTION PassiveTransportCoefficients
HLLEPassiveTransport(const parthenon::Real density_left, const parthenon::Real density_right,
                     const parthenon::Real mass_flux_left, const parthenon::Real mass_flux_right,
                     const parthenon::Real lambda_left, const parthenon::Real lambda_right) {
  if (lambda_left >= 0.0)
    return {mass_flux_left, 0.0};
  if (lambda_right <= 0.0)
    return {0.0, mass_flux_right};
  const parthenon::Real wave_product = lambda_right * lambda_left;
  const parthenon::Real inverse = 1.0 / (lambda_right - lambda_left);
  return {(lambda_right * mass_flux_left - wave_product * density_left) * inverse,
          (-lambda_left * mass_flux_right + wave_product * density_right) * inverse};
}

} // namespace pangu::mhd
