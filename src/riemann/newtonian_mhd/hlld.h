#ifndef PANGU_RIEMANN_NEWTONIAN_MHD_HLLD_H_
#define PANGU_RIEMANN_NEWTONIAN_MHD_HLLD_H_

// Copyright (c) 2020 James M. Stone and the AthenaK team.
// HLLD algebra is derived from AthenaK under BSD-3-Clause; see LICENSE/NOTICE.

#include <cmath>

#include "eos/newtonian_mhd_eos.h"

namespace pangu::riemann::newtonian_mhd {

using mhd::InterfaceFlux;
using mhd::Primitive;
using parthenon::Real;

struct HLLDState {
  Real d;
  Real mx;
  Real my;
  Real mz;
  Real e;
  Real by;
  Real bz;
};

KOKKOS_INLINE_FUNCTION
void StoreHLLDFlux(const HLLDState& flux, const int direction, const bool has_energy,
                   InterfaceFlux& result) {
  const int tangent1 = (direction + 1) % 3;
  const int tangent2 = (direction + 2) % 3;
  result.fluid[mhd::IDN] = flux.d;
  result.fluid[mhd::MomentumIndex(direction)] = flux.mx;
  result.fluid[mhd::IM1 + tangent1] = flux.my;
  result.fluid[mhd::IM1 + tangent2] = flux.mz;
  if (has_energy)
    result.fluid[mhd::IEN] = flux.e;
  result.electric_t1 = flux.bz;
  result.electric_t2 = -flux.by;
}

KOKKOS_INLINE_FUNCTION
void SolveHLLDIdeal(const Primitive& left, const Primitive& right, const eos::NewtonianMHDEOS& eos,
                    const int direction, InterfaceFlux& result) {
  constexpr Real small_number = 1.0e-4;
  const int tangent1 = (direction + 1) % 3;
  const int tangent2 = (direction + 2) % 3;
  const Real bx = left.magnetic[direction];
  const Real bx2 = bx * bx;
  const Real rho_l = left.density;
  const Real rho_r = right.density;
  const Real vx_l = left.velocity[direction];
  const Real vx_r = right.velocity[direction];
  const Real vy_l = left.velocity[tangent1];
  const Real vy_r = right.velocity[tangent1];
  const Real vz_l = left.velocity[tangent2];
  const Real vz_r = right.velocity[tangent2];
  const Real by_l = left.magnetic[tangent1];
  const Real by_r = right.magnetic[tangent1];
  const Real bz_l = left.magnetic[tangent2];
  const Real bz_r = right.magnetic[tangent2];
  const Real pbl = 0.5 * (bx2 + (by_l * by_l + bz_l * bz_l));
  const Real pbr = 0.5 * (bx2 + (by_r * by_r + bz_r * bz_r));
  const Real kel = 0.5 * rho_l * (vx_l * vx_l + (vy_l * vy_l + vz_l * vz_l));
  const Real ker = 0.5 * rho_r * (vx_r * vx_r + (vy_r * vy_r + vz_r * vz_r));

  HLLDState ul{rho_l,
               rho_l * vx_l,
               rho_l * vy_l,
               rho_l * vz_l,
               left.pressure / (eos.gamma - 1.0) + kel + pbl,
               by_l,
               bz_l};
  HLLDState ur{rho_r,
               rho_r * vx_r,
               rho_r * vy_r,
               rho_r * vz_r,
               right.pressure / (eos.gamma - 1.0) + ker + pbr,
               by_r,
               bz_r};
  const Real sl =
      fmin(vx_l - eos.FastSpeed(left, direction), vx_r - eos.FastSpeed(right, direction));
  const Real sr =
      fmax(vx_l + eos.FastSpeed(left, direction), vx_r + eos.FastSpeed(right, direction));
  const Real ptl = left.pressure + pbl;
  const Real ptr = right.pressure + pbr;
  HLLDState fl{ul.mx,
               ul.mx * vx_l + ptl - bx2,
               ul.my * vx_l - bx * ul.by,
               ul.mz * vx_l - bx * ul.bz,
               vx_l * (ul.e + ptl - bx2) - bx * (vy_l * ul.by + vz_l * ul.bz),
               ul.by * vx_l - bx * vy_l,
               ul.bz * vx_l - bx * vz_l};
  HLLDState fr{ur.mx,
               ur.mx * vx_r + ptr - bx2,
               ur.my * vx_r - bx * ur.by,
               ur.mz * vx_r - bx * ur.bz,
               vx_r * (ur.e + ptr - bx2) - bx * (vy_r * ur.by + vz_r * ur.bz),
               ur.by * vx_r - bx * vy_r,
               ur.bz * vx_r - bx * vz_r};

  if (sl >= 0.0) {
    StoreHLLDFlux(fl, direction, true, result);
    return;
  }
  if (sr <= 0.0) {
    StoreHLLDFlux(fr, direction, true, result);
    return;
  }

  const Real sdl = sl - vx_l;
  const Real sdr = sr - vx_r;
  const Real sm = (sdr * ur.mx - sdl * ul.mx + (ptl - ptr)) / (sdr * ur.d - sdl * ul.d);
  const Real sdml = sl - sm;
  const Real sdmr = sr - sm;
  const Real ptstl = ptl + ul.d * sdl * (sm - vx_l);
  const Real ptstr = ptr + ur.d * sdr * (sm - vx_r);
  const Real ptst = 0.5 * (ptstl + ptstr);

  HLLDState ulst{};
  HLLDState urst{};
  ulst.d = ul.d * sdl / sdml;
  urst.d = ur.d * sdr / sdmr;
  ulst.mx = ulst.d * sm;
  urst.mx = urst.d * sm;
  if (fabs(ul.d * sdl * sdml - bx2) < small_number * ptst) {
    ulst.my = ulst.d * vy_l;
    ulst.mz = ulst.d * vz_l;
    ulst.by = ul.by;
    ulst.bz = ul.bz;
  } else {
    Real factor = bx * (sdl - sdml) / (ul.d * sdl * sdml - bx2);
    ulst.my = ulst.d * (vy_l - ul.by * factor);
    ulst.mz = ulst.d * (vz_l - ul.bz * factor);
    factor = (ul.d * sdl * sdl - bx2) / (ul.d * sdl * sdml - bx2);
    ulst.by = ul.by * factor;
    ulst.bz = ul.bz * factor;
  }
  if (fabs(ur.d * sdr * sdmr - bx2) < small_number * ptst) {
    urst.my = urst.d * vy_r;
    urst.mz = urst.d * vz_r;
    urst.by = ur.by;
    urst.bz = ur.bz;
  } else {
    Real factor = bx * (sdr - sdmr) / (ur.d * sdr * sdmr - bx2);
    urst.my = urst.d * (vy_r - ur.by * factor);
    urst.mz = urst.d * (vz_r - ur.bz * factor);
    factor = (ur.d * sdr * sdr - bx2) / (ur.d * sdr * sdmr - bx2);
    urst.by = ur.by * factor;
    urst.bz = ur.bz * factor;
  }

  const Real vbstl = (ulst.mx * bx + (ulst.my * ulst.by + ulst.mz * ulst.bz)) / ulst.d;
  const Real vbstr = (urst.mx * bx + (urst.my * urst.by + urst.mz * urst.bz)) / urst.d;
  ulst.e = (sdl * ul.e - ptl * vx_l + ptst * sm +
            bx * (vx_l * bx + (vy_l * ul.by + vz_l * ul.bz) - vbstl)) /
           sdml;
  urst.e = (sdr * ur.e - ptr * vx_r + ptst * sm +
            bx * (vx_r * bx + (vy_r * ur.by + vz_r * ur.bz) - vbstr)) /
           sdmr;

  const Real sqrtdl = sqrt(ulst.d);
  const Real sqrtdr = sqrt(urst.d);
  const Real sal = sm - fabs(bx) / sqrtdl;
  const Real sar = sm + fabs(bx) / sqrtdr;
  HLLDState uldst{};
  HLLDState urdst{};
  if (0.5 * bx2 < small_number * ptst) {
    uldst = ulst;
    urdst = urst;
  } else {
    const Real inverse_sum = 1.0 / (sqrtdl + sqrtdr);
    const Real sign_bx = bx > 0.0 ? 1.0 : -1.0;
    uldst.d = ulst.d;
    urdst.d = urst.d;
    uldst.mx = ulst.mx;
    urdst.mx = urst.mx;
    Real factor = inverse_sum * (sqrtdl * ulst.my / ulst.d + sqrtdr * urst.my / urst.d +
                                 sign_bx * (urst.by - ulst.by));
    uldst.my = uldst.d * factor;
    urdst.my = urdst.d * factor;
    factor = inverse_sum * (sqrtdl * ulst.mz / ulst.d + sqrtdr * urst.mz / urst.d +
                            sign_bx * (urst.bz - ulst.bz));
    uldst.mz = uldst.d * factor;
    urdst.mz = urdst.d * factor;
    factor = inverse_sum * (sqrtdl * urst.by + sqrtdr * ulst.by +
                            sign_bx * sqrtdl * sqrtdr * (urst.my / urst.d - ulst.my / ulst.d));
    uldst.by = factor;
    urdst.by = factor;
    factor = inverse_sum * (sqrtdl * urst.bz + sqrtdr * ulst.bz +
                            sign_bx * sqrtdl * sqrtdr * (urst.mz / urst.d - ulst.mz / ulst.d));
    uldst.bz = factor;
    urdst.bz = factor;
    const Real vbdst = sm * bx + (uldst.my * uldst.by + uldst.mz * uldst.bz) / uldst.d;
    uldst.e = ulst.e - sqrtdl * sign_bx * (vbstl - vbdst);
    urdst.e = urst.e + sqrtdr * sign_bx * (vbstr - vbdst);
  }

  HLLDState flux{};
  if (sal >= 0.0) {
    flux = {fl.d + sl * (ulst.d - ul.d),    fl.mx + sl * (ulst.mx - ul.mx),
            fl.my + sl * (ulst.my - ul.my), fl.mz + sl * (ulst.mz - ul.mz),
            fl.e + sl * (ulst.e - ul.e),    fl.by + sl * (ulst.by - ul.by),
            fl.bz + sl * (ulst.bz - ul.bz)};
  } else if (sm >= 0.0) {
    flux = {fl.d + sl * (ulst.d - ul.d) + sal * (uldst.d - ulst.d),
            fl.mx + sl * (ulst.mx - ul.mx) + sal * (uldst.mx - ulst.mx),
            fl.my + sl * (ulst.my - ul.my) + sal * (uldst.my - ulst.my),
            fl.mz + sl * (ulst.mz - ul.mz) + sal * (uldst.mz - ulst.mz),
            fl.e + sl * (ulst.e - ul.e) + sal * (uldst.e - ulst.e),
            fl.by + sl * (ulst.by - ul.by) + sal * (uldst.by - ulst.by),
            fl.bz + sl * (ulst.bz - ul.bz) + sal * (uldst.bz - ulst.bz)};
  } else if (sar > 0.0) {
    flux = {fr.d + sr * (urst.d - ur.d) + sar * (urdst.d - urst.d),
            fr.mx + sr * (urst.mx - ur.mx) + sar * (urdst.mx - urst.mx),
            fr.my + sr * (urst.my - ur.my) + sar * (urdst.my - urst.my),
            fr.mz + sr * (urst.mz - ur.mz) + sar * (urdst.mz - urst.mz),
            fr.e + sr * (urst.e - ur.e) + sar * (urdst.e - urst.e),
            fr.by + sr * (urst.by - ur.by) + sar * (urdst.by - urst.by),
            fr.bz + sr * (urst.bz - ur.bz) + sar * (urdst.bz - urst.bz)};
  } else {
    flux = {fr.d + sr * (urst.d - ur.d),    fr.mx + sr * (urst.mx - ur.mx),
            fr.my + sr * (urst.my - ur.my), fr.mz + sr * (urst.mz - ur.mz),
            fr.e + sr * (urst.e - ur.e),    fr.by + sr * (urst.by - ur.by),
            fr.bz + sr * (urst.bz - ur.bz)};
  }
  StoreHLLDFlux(flux, direction, true, result);
}

KOKKOS_INLINE_FUNCTION
void SolveHLLDIsothermal(const Primitive& left, const Primitive& right,
                         const eos::NewtonianMHDEOS& eos, const int direction,
                         InterfaceFlux& result) {
  constexpr Real small_number = 1.0e-4;
  const int tangent1 = (direction + 1) % 3;
  const int tangent2 = (direction + 2) % 3;
  const Real bx = left.magnetic[direction];
  const Real bx2 = bx * bx;
  const Real cs2 = eos.iso_sound_speed * eos.iso_sound_speed;
  HLLDState ul{left.density,
               left.density * left.velocity[direction],
               left.density * left.velocity[tangent1],
               left.density * left.velocity[tangent2],
               0.0,
               left.magnetic[tangent1],
               left.magnetic[tangent2]};
  HLLDState ur{right.density,
               right.density * right.velocity[direction],
               right.density * right.velocity[tangent1],
               right.density * right.velocity[tangent2],
               0.0,
               right.magnetic[tangent1],
               right.magnetic[tangent2]};
  const Real sl = fmin(left.velocity[direction] - eos.FastSpeed(left, direction),
                       right.velocity[direction] - eos.FastSpeed(right, direction));
  const Real sr = fmax(left.velocity[direction] + eos.FastSpeed(left, direction),
                       right.velocity[direction] + eos.FastSpeed(right, direction));
  const Real ptl = cs2 * ul.d + 0.5 * (bx2 + ul.by * ul.by + ul.bz * ul.bz);
  const Real ptr = cs2 * ur.d + 0.5 * (bx2 + ur.by * ur.by + ur.bz * ur.bz);
  HLLDState fl{ul.mx,
               ul.mx * left.velocity[direction] + ptl - bx2,
               ul.my * left.velocity[direction] - bx * ul.by,
               ul.mz * left.velocity[direction] - bx * ul.bz,
               0.0,
               ul.by * left.velocity[direction] - bx * left.velocity[tangent1],
               ul.bz * left.velocity[direction] - bx * left.velocity[tangent2]};
  HLLDState fr{ur.mx,
               ur.mx * right.velocity[direction] + ptr - bx2,
               ur.my * right.velocity[direction] - bx * ur.by,
               ur.mz * right.velocity[direction] - bx * ur.bz,
               0.0,
               ur.by * right.velocity[direction] - bx * right.velocity[tangent1],
               ur.bz * right.velocity[direction] - bx * right.velocity[tangent2]};
  if (sl >= 0.0) {
    StoreHLLDFlux(fl, direction, false, result);
    return;
  }
  if (sr <= 0.0) {
    StoreHLLDFlux(fr, direction, false, result);
    return;
  }
  const Real inverse_speed = 1.0 / (sr - sl);
  const Real dhll = fmax((sr * ur.d - sl * ul.d - fr.d + fl.d) * inverse_speed, eos.density_floor);
  const Real fdhll = (sr * fl.d - sl * fr.d + sr * sl * (ur.d - ul.d)) * inverse_speed;
  const Real fmxhll = (sr * fl.mx - sl * fr.mx + sr * sl * (ur.mx - ul.mx)) * inverse_speed;
  const Real ustar = fdhll / dhll;
  const Real mxhll = (sr * ur.mx - sl * ul.mx - fr.mx + fl.mx) * inverse_speed;
  const Real sqrt_dhll = sqrt(dhll);
  const Real sal = ustar - fabs(bx) / sqrt_dhll;
  const Real sar = ustar + fabs(bx) / sqrt_dhll;
  HLLDState ulst{dhll, mxhll, 0.0, 0.0, 0.0, 0.0, 0.0};
  HLLDState urst{dhll, mxhll, 0.0, 0.0, 0.0, 0.0, 0.0};
  Real factor = (sl - sal) * (sl - sar);
  if (fabs(sl - sal) < small_number * eos.iso_sound_speed) {
    ulst.my = ul.my;
    ulst.mz = ul.mz;
    ulst.by = ul.by;
    ulst.bz = ul.bz;
  } else {
    const Real momentum_factor = bx * (ustar - left.velocity[direction]) / factor;
    const Real magnetic_factor =
        (ul.d * (sl - left.velocity[direction]) * (sl - left.velocity[direction]) - bx2) /
        (dhll * factor);
    ulst.my = dhll * left.velocity[tangent1] - ul.by * momentum_factor;
    ulst.mz = dhll * left.velocity[tangent2] - ul.bz * momentum_factor;
    ulst.by = ul.by * magnetic_factor;
    ulst.bz = ul.bz * magnetic_factor;
  }
  factor = (sr - sal) * (sr - sar);
  if (fabs(sr - sar) < small_number * eos.iso_sound_speed) {
    urst.my = ur.my;
    urst.mz = ur.mz;
    urst.by = ur.by;
    urst.bz = ur.bz;
  } else {
    const Real momentum_factor = bx * (ustar - right.velocity[direction]) / factor;
    const Real magnetic_factor =
        (ur.d * (sr - right.velocity[direction]) * (sr - right.velocity[direction]) - bx2) /
        (dhll * factor);
    urst.my = dhll * right.velocity[tangent1] - ur.by * momentum_factor;
    urst.mz = dhll * right.velocity[tangent2] - ur.bz * momentum_factor;
    urst.by = ur.by * magnetic_factor;
    urst.bz = ur.bz * magnetic_factor;
  }
  const Real signed_root = sqrt_dhll * (bx > 0.0 ? 1.0 : -1.0);
  HLLDState center{dhll,
                   mxhll,
                   0.5 * (ulst.my + urst.my + (urst.by - ulst.by) * signed_root),
                   0.5 * (ulst.mz + urst.mz + (urst.bz - ulst.bz) * signed_root),
                   0.0,
                   0.5 * (ulst.by + urst.by + (urst.my - ulst.my) / signed_root),
                   0.5 * (ulst.bz + urst.bz + (urst.mz - ulst.mz) / signed_root)};
  HLLDState flux{};
  if (sal >= 0.0) {
    flux = {fl.d + sl * (ulst.d - ul.d),
            fl.mx + sl * (ulst.mx - ul.mx),
            fl.my + sl * (ulst.my - ul.my),
            fl.mz + sl * (ulst.mz - ul.mz),
            0.0,
            fl.by + sl * (ulst.by - ul.by),
            fl.bz + sl * (ulst.bz - ul.bz)};
  } else if (sar <= 0.0) {
    flux = {fr.d + sr * (urst.d - ur.d),
            fr.mx + sr * (urst.mx - ur.mx),
            fr.my + sr * (urst.my - ur.my),
            fr.mz + sr * (urst.mz - ur.mz),
            0.0,
            fr.by + sr * (urst.by - ur.by),
            fr.bz + sr * (urst.bz - ur.bz)};
  } else {
    flux = {dhll * ustar,
            fmxhll,
            center.my * ustar - bx * center.by,
            center.mz * ustar - bx * center.bz,
            0.0,
            center.by * ustar - bx * center.my / center.d,
            center.bz * ustar - bx * center.mz / center.d};
  }
  StoreHLLDFlux(flux, direction, false, result);
}

// HLLD flux resolving the rotational and contact waves.
struct HLLD {
  KOKKOS_INLINE_FUNCTION static void Solve(const Primitive& left, const Primitive& right,
                                           const eos::NewtonianMHDEOS& eos, const int direction,
                                           InterfaceFlux& result) {
    if (eos.HasEnergy())
      SolveHLLDIdeal(left, right, eos, direction, result);
    else
      SolveHLLDIsothermal(left, right, eos, direction, result);
  }
};

} // namespace pangu::riemann::newtonian_mhd

#endif
