// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#include "riemann_solver/hlld.h"

#include <memory>
#include <parthenon/package.hpp>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "initialization/variable_mnemonics.h"
#include "interpolation/interpolater_mc.h"
#include "metric/tetrad.h"

namespace {
struct Cons1D {
  parthenon::Real d, mx, my, mz, e, bx, by, bz;
};

KOKKOS_INLINE_FUNCTION
parthenon::Real Sqr(const parthenon::Real x) { return x * x; }

KOKKOS_INLINE_FUNCTION
void CalculateLocalState(const parthenon::Real gamma,
                         const parthenon::Real primitive[NPRIM],
                         Cons1D &conservative, Cons1D &flux,
                         parthenon::Real &lambda_minus,
                         parthenon::Real &lambda_plus) {
  using parthenon::Real;

  const Real rho = primitive[RHO];
  const Real internal_energy = primitive[ENY];
  const Real pgas = (gamma - 1.0) * internal_energy;
  const Real ux = primitive[UX1];
  const Real uy = primitive[UX2];
  const Real uz = primitive[UX3];
  const Real bx = primitive[BX1];
  const Real by = primitive[BX2];
  const Real bz = primitive[BX3];

  const Real u0 = Kokkos::sqrt(1.0 + Sqr(ux) + Sqr(uy) + Sqr(uz));
  const Real vx = ux / u0;
  const Real vy = uy / u0;
  const Real vz = uz / u0;

  const Real b0 = bx * ux + by * uy + bz * uz;
  const Real b1 = (bx + b0 * ux) / u0;
  const Real b2 = (by + b0 * uy) / u0;
  const Real b3 = (bz + b0 * uz) / u0;
  const Real bsq = -Sqr(b0) + Sqr(b1) + Sqr(b2) + Sqr(b3);

  const Real wgas = rho + gamma * internal_energy;
  const Real wtot = wgas + bsq;
  const Real ptot = pgas + 0.5 * bsq;

  conservative.d = rho * u0;
  conservative.mx = wtot * ux * u0 - b1 * b0;
  conservative.my = wtot * uy * u0 - b2 * b0;
  conservative.mz = wtot * uz * u0 - b3 * b0;
  conservative.e = wtot * Sqr(u0) - Sqr(b0) - ptot;
  conservative.bx = bx;
  conservative.by = by;
  conservative.bz = bz;

  flux.d = rho * ux;
  flux.mx = wtot * ux * ux - b1 * b1 + ptot;
  flux.my = wtot * uy * ux - b2 * b1;
  flux.mz = wtot * uz * ux - b3 * b1;
  flux.e = wtot * u0 * ux - b0 * b1;
  flux.bx = 0.0;
  flux.by = by * vx - bx * vy;
  flux.bz = bz * vx - bx * vz;

  const Real cs_sq = Kokkos::max(gamma * pgas / wgas, 0.0);
  const Real va_sq = Kokkos::max(bsq / wtot, 0.0);
  const Real cf_sq =
      Kokkos::min(cs_sq + va_sq - cs_sq * va_sq, 1.0);
  const Real cf = Kokkos::sqrt(Kokkos::max(cf_sq, 0.0));

  lambda_minus = (vx - cf) / (1.0 - vx * cf);
  lambda_plus = (vx + cf) / (1.0 + vx * cf);
}

KOKKOS_INLINE_FUNCTION
void AddScaledJump(const Cons1D &base_flux, const Cons1D &state,
                   const Cons1D &base_state, const parthenon::Real speed,
                   Cons1D &flux) {
  flux.d = base_flux.d + speed * (state.d - base_state.d);
  flux.mx = base_flux.mx + speed * (state.mx - base_state.mx);
  flux.my = base_flux.my + speed * (state.my - base_state.my);
  flux.mz = base_flux.mz + speed * (state.mz - base_state.mz);
  flux.e = base_flux.e + speed * (state.e - base_state.e);
  flux.bx = 0.0;
  flux.by = base_flux.by + speed * (state.by - base_state.by);
  flux.bz = base_flux.bz + speed * (state.bz - base_state.bz);
}

KOKKOS_INLINE_FUNCTION
void StoreLocalFlux(const Cons1D &flux, const parthenon::Real mass_scalar,
                    const parthenon::Real electron_scalar,
                    parthenon::Real local_flux[NPRIM]) {
  local_flux[RHO] = flux.d;
  local_flux[ENY] = flux.e;
  local_flux[UX1] = flux.mx;
  local_flux[UX2] = flux.my;
  local_flux[UX3] = flux.mz;
  local_flux[BX1] = 0.0;
  local_flux[BX2] = flux.by;
  local_flux[BX3] = flux.bz;
  local_flux[ENT] = mass_scalar * flux.d;
  local_flux[KEL] = electron_scalar * flux.d;
}

KOKKOS_INLINE_FUNCTION
void CalculateLocalHlldFlux(const parthenon::Real gamma,
                            const parthenon::Real primitive_left[NPRIM],
                            const parthenon::Real primitive_right[NPRIM],
                            parthenon::Real local_flux[NPRIM],
                            parthenon::Real &maximum_signal_speed) {
  using parthenon::Real;
  constexpr Real small_number = 1.0e-12;

  Cons1D ul, ur, fl, fr;
  Real lambda_minus_left, lambda_plus_left;
  Real lambda_minus_right, lambda_plus_right;
  CalculateLocalState(gamma, primitive_left, ul, fl, lambda_minus_left,
                      lambda_plus_left);
  CalculateLocalState(gamma, primitive_right, ur, fr, lambda_minus_right,
                      lambda_plus_right);

  const Real sl = Kokkos::min(lambda_minus_left, lambda_minus_right);
  const Real sr = Kokkos::max(lambda_plus_left, lambda_plus_right);
  maximum_signal_speed = Kokkos::max(Kokkos::fabs(sl), Kokkos::fabs(sr));

  const Real mass_scalar =
      (fl.d >= 0.0) ? primitive_left[ENT] : primitive_right[ENT];
  const Real electron_scalar =
      (fl.d >= 0.0) ? primitive_left[KEL] : primitive_right[KEL];

  if (sl >= 0.0) {
    StoreLocalFlux(fl, primitive_left[ENT], primitive_left[KEL], local_flux);
    return;
  }
  if (sr <= 0.0) {
    StoreLocalFlux(fr, primitive_right[ENT], primitive_right[KEL], local_flux);
    return;
  }

  const Real vxl = primitive_left[UX1] /
                   Kokkos::sqrt(1.0 + Sqr(primitive_left[UX1]) +
                                Sqr(primitive_left[UX2]) +
                                Sqr(primitive_left[UX3]));
  const Real vxr = primitive_right[UX1] /
                   Kokkos::sqrt(1.0 + Sqr(primitive_right[UX1]) +
                                Sqr(primitive_right[UX2]) +
                                Sqr(primitive_right[UX3]));
  const Real ptl = fl.mx - ul.mx * vxl + Sqr(ul.bx);
  const Real ptr = fr.mx - ur.mx * vxr + Sqr(ur.bx);
  const Real sdl = sl - vxl;
  const Real sdr = sr - vxr;
  const Real sm =
      (sdr * ur.mx - sdl * ul.mx + ptl - ptr) /
      (sdr * ur.d - sdl * ul.d);
  const Real sdml = sl - sm;
  const Real sdmr = sr - sm;

  Cons1D ulst, urst, uldst, urdst;
  ulst.d = ul.d * sdl / sdml;
  urst.d = ur.d * sdr / sdmr;
  const Real sqrtdl = Kokkos::sqrt(Kokkos::max(ulst.d, small_number));
  const Real sqrtdr = Kokkos::sqrt(Kokkos::max(urst.d, small_number));
  const Real bx = 0.5 * (ul.bx + ur.bx);
  const Real bxsq = Sqr(bx);
  const Real ptst = 0.5 * (ptl + ul.d * sdl * (sm - vxl) + ptr +
                           ur.d * sdr * (sm - vxr));

  ulst.mx = ulst.d * sm;
  urst.mx = urst.d * sm;
  if (Kokkos::fabs(ul.d * sdl * sdml - bxsq) <
      small_number * Kokkos::fabs(ptst)) {
    ulst.my = ulst.d * primitive_left[UX2] /
              Kokkos::sqrt(1.0 + Sqr(primitive_left[UX1]) +
                           Sqr(primitive_left[UX2]) + Sqr(primitive_left[UX3]));
    ulst.mz = ulst.d * primitive_left[UX3] /
              Kokkos::sqrt(1.0 + Sqr(primitive_left[UX1]) +
                           Sqr(primitive_left[UX2]) + Sqr(primitive_left[UX3]));
    ulst.by = ul.by;
    ulst.bz = ul.bz;
  } else {
    Real tmp = bx * (sdl - sdml) / (ul.d * sdl * sdml - bxsq);
    ulst.my = ulst.d * (primitive_left[UX2] /
                            Kokkos::sqrt(1.0 + Sqr(primitive_left[UX1]) +
                                         Sqr(primitive_left[UX2]) +
                                         Sqr(primitive_left[UX3])) -
                        ul.by * tmp);
    ulst.mz = ulst.d * (primitive_left[UX3] /
                            Kokkos::sqrt(1.0 + Sqr(primitive_left[UX1]) +
                                         Sqr(primitive_left[UX2]) +
                                         Sqr(primitive_left[UX3])) -
                        ul.bz * tmp);
    tmp = (ul.d * Sqr(sdl) - bxsq) / (ul.d * sdl * sdml - bxsq);
    ulst.by = ul.by * tmp;
    ulst.bz = ul.bz * tmp;
  }

  if (Kokkos::fabs(ur.d * sdr * sdmr - bxsq) <
      small_number * Kokkos::fabs(ptst)) {
    urst.my = urst.d * primitive_right[UX2] /
              Kokkos::sqrt(1.0 + Sqr(primitive_right[UX1]) +
                           Sqr(primitive_right[UX2]) +
                           Sqr(primitive_right[UX3]));
    urst.mz = urst.d * primitive_right[UX3] /
              Kokkos::sqrt(1.0 + Sqr(primitive_right[UX1]) +
                           Sqr(primitive_right[UX2]) +
                           Sqr(primitive_right[UX3]));
    urst.by = ur.by;
    urst.bz = ur.bz;
  } else {
    Real tmp = bx * (sdr - sdmr) / (ur.d * sdr * sdmr - bxsq);
    urst.my = urst.d * (primitive_right[UX2] /
                            Kokkos::sqrt(1.0 + Sqr(primitive_right[UX1]) +
                                         Sqr(primitive_right[UX2]) +
                                         Sqr(primitive_right[UX3])) -
                        ur.by * tmp);
    urst.mz = urst.d * (primitive_right[UX3] /
                            Kokkos::sqrt(1.0 + Sqr(primitive_right[UX1]) +
                                         Sqr(primitive_right[UX2]) +
                                         Sqr(primitive_right[UX3])) -
                        ur.bz * tmp);
    tmp = (ur.d * Sqr(sdr) - bxsq) / (ur.d * sdr * sdmr - bxsq);
    urst.by = ur.by * tmp;
    urst.bz = ur.bz * tmp;
  }

  const Real vbstl =
      (ulst.mx * bx + ulst.my * ulst.by + ulst.mz * ulst.bz) / ulst.d;
  const Real vbstr =
      (urst.mx * bx + urst.my * urst.by + urst.mz * urst.bz) / urst.d;
  ulst.e = (sdl * ul.e - ptl * vxl + ptst * sm +
            bx * (vxl * bx + primitive_left[UX2] / Kokkos::sqrt(
                                      1.0 + Sqr(primitive_left[UX1]) +
                                      Sqr(primitive_left[UX2]) +
                                      Sqr(primitive_left[UX3])) *
                                      ul.by +
                  primitive_left[UX3] / Kokkos::sqrt(
                                      1.0 + Sqr(primitive_left[UX1]) +
                                      Sqr(primitive_left[UX2]) +
                                      Sqr(primitive_left[UX3])) *
                                      ul.bz -
                  vbstl)) /
           sdml;
  urst.e = (sdr * ur.e - ptr * vxr + ptst * sm +
            bx * (vxr * bx + primitive_right[UX2] / Kokkos::sqrt(
                                      1.0 + Sqr(primitive_right[UX1]) +
                                      Sqr(primitive_right[UX2]) +
                                      Sqr(primitive_right[UX3])) *
                                      ur.by +
                  primitive_right[UX3] / Kokkos::sqrt(
                                      1.0 + Sqr(primitive_right[UX1]) +
                                      Sqr(primitive_right[UX2]) +
                                      Sqr(primitive_right[UX3])) *
                                      ur.bz -
                  vbstr)) /
           sdmr;
  ulst.bx = bx;
  urst.bx = bx;

  const Real invsumd = 1.0 / (sqrtdl + sqrtdr);
  const Real bxsig = (bx >= 0.0) ? 1.0 : -1.0;
  uldst.d = ulst.d;
  urdst.d = urst.d;
  uldst.mx = ulst.mx;
  urdst.mx = urst.mx;
  Real tmp = invsumd * (sqrtdl * ulst.my / ulst.d +
                        sqrtdr * urst.my / urst.d +
                        bxsig * (urst.by - ulst.by));
  uldst.my = uldst.d * tmp;
  urdst.my = urdst.d * tmp;
  tmp = invsumd * (sqrtdl * ulst.mz / ulst.d +
                   sqrtdr * urst.mz / urst.d +
                   bxsig * (urst.bz - ulst.bz));
  uldst.mz = uldst.d * tmp;
  urdst.mz = urdst.d * tmp;
  tmp = invsumd * (sqrtdl * urst.by + sqrtdr * ulst.by +
                   bxsig * sqrtdl * sqrtdr *
                       (urst.my / urst.d - ulst.my / ulst.d));
  uldst.by = tmp;
  urdst.by = tmp;
  tmp = invsumd * (sqrtdl * urst.bz + sqrtdr * ulst.bz +
                   bxsig * sqrtdl * sqrtdr *
                       (urst.mz / urst.d - ulst.mz / ulst.d));
  uldst.bz = tmp;
  urdst.bz = tmp;
  uldst.bx = bx;
  urdst.bx = bx;
  tmp = sm * bx + (uldst.my * uldst.by + uldst.mz * uldst.bz) / uldst.d;
  uldst.e = ulst.e - sqrtdl * bxsig * (vbstl - tmp);
  urdst.e = urst.e + sqrtdr * bxsig * (vbstr - tmp);

  const Real sal = sm - Kokkos::fabs(bx) / sqrtdl;
  const Real sar = sm + Kokkos::fabs(bx) / sqrtdr;

  Cons1D flux_interface;
  if (sal >= 0.0) {
    AddScaledJump(fl, ulst, ul, sl, flux_interface);
  } else if (sar <= 0.0) {
    AddScaledJump(fr, urst, ur, sr, flux_interface);
  } else if (sm >= 0.0) {
    Cons1D flux_star;
    AddScaledJump(fl, ulst, ul, sl, flux_star);
    AddScaledJump(flux_star, uldst, ulst, sal, flux_interface);
  } else {
    Cons1D flux_star;
    AddScaledJump(fr, urst, ur, sr, flux_star);
    AddScaledJump(flux_star, urdst, urst, sar, flux_interface);
  }

  StoreLocalFlux(flux_interface, mass_scalar, electron_scalar, local_flux);
}

KOKKOS_INLINE_FUNCTION
void CalculateProjectedHlldFlux(const parthenon::Real gamma,
                                const parthenon::Real primitive_left[NPRIM],
                                const parthenon::Real primitive_right[NPRIM],
                                const parthenon::Real gcov[4][4],
                                const parthenon::Real metric_determinant,
                                const int dir,
                                parthenon::Real global_flux[NPRIM],
                                parthenon::Real &maximum_signal_speed) {
  Tetrad tetrad;
  BuildTetrad(gcov, metric_determinant, dir, tetrad);

  parthenon::Real local_primitive_left[NPRIM];
  parthenon::Real local_primitive_right[NPRIM];
  ProjectPrimitiveToTetrad(tetrad, primitive_left, local_primitive_left);
  ProjectPrimitiveToTetrad(tetrad, primitive_right, local_primitive_right);

  parthenon::Real local_flux[NPRIM];
  CalculateLocalHlldFlux(gamma, local_primitive_left, local_primitive_right,
                         local_flux, maximum_signal_speed);
  ProjectTetradFluxToGlobal(tetrad, local_flux, global_flux);
}
}  // namespace

parthenon::TaskStatus CalculateHLLD(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &init_resource) {
  using namespace parthenon;
  PARTHENON_INSTRUMENT

  const auto pmb = resource->GetBlockPointer();
  const auto package_core = pmb->packages.Get("core");
  const auto &kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");

  const auto bound_x1_interior =
      pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto bound_x2_interior =
      pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto bound_x3_interior =
      pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
	  "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
	  "electron_entropy"
    };
  const auto primitive =
      resource->PackVariables(primitive_tags, primitiveIndexMap);
  PackIndexMap conservativeIndexMap;
  const std::vector<std::string> conservative_tags = {"conservative"};
  auto conservative =
      resource->PackVariablesAndFluxes(conservative_tags, conservativeIndexMap);
  PackIndexMap alfvenVelocityIndexMap;
  const std::vector<std::string> alfven_tags = {"alfven"};
  auto AlfvenVelocity =
      resource->PackVariables(alfven_tags, alfvenVelocityIndexMap);

  auto covariant_metric = init_resource->Get("covariant_metric").data;
  auto metric_determinant = init_resource->Get("metric_determinant").data;

  const auto meshgrid_size_x1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
  const auto meshgrid_size_x2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
  const auto meshgrid_size_x3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

  const int offset_x1 = (meshgrid_size_x1 > 1) ? 1 : 0;
  const int offset_x2 = (meshgrid_size_x2 > 1) ? 1 : 0;
  const int offset_x3 = (meshgrid_size_x3 > 1) ? 1 : 0;

  pmb->par_for(
      PARTHENON_AUTO_LABEL, bound_x3_interior.s - offset_x3,
      bound_x3_interior.e + offset_x3, bound_x2_interior.s - offset_x2,
      bound_x2_interior.e + offset_x2, bound_x1_interior.s,
      bound_x1_interior.e + 1, KOKKOS_LAMBDA(const int k, const int j,
                                             const int i) {
        Real primitiveLeft[NPRIM];
        Real primitiveRight[NPRIM];
        for (int index = 0; index < NPRIM; ++index) {
          primitiveLeft[index] =
              primitive(index, k, j, i - 1) +
              0.5 * InterpolateMC(primitive(index, k, j, i - 2),
                                  primitive(index, k, j, i - 1),
                                  primitive(index, k, j, i));
          primitiveRight[index] =
              primitive(index, k, j, i) -
              0.5 * InterpolateMC(primitive(index, k, j, i - 1),
                                  primitive(index, k, j, i),
                                  primitive(index, k, j, i + 1));
        }

        Real gcovFace[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcovFace[row][col] = covariant_metric(FACEX1, col, row, k, j, i);
          }
        }

        Real flux[NPRIM];
        Real maximumSignalSpeed;
        CalculateProjectedHlldFlux(kAdiabaticIndex, primitiveLeft,
                                   primitiveRight, gcovFace,
                                   metric_determinant(FACEX1, k, j, i), X1DIR,
                                   flux, maximumSignalSpeed);
        AlfvenVelocity(Vector3D::X1, k, j, i) = maximumSignalSpeed;
        for (int index = 0; index < NPRIM; ++index) {
          conservative.flux(X1DIR, index, k, j, i) = flux[index];
        }
      });

  if (pmb->pmy_mesh->ndim >= 2)
    pmb->par_for(
        PARTHENON_AUTO_LABEL, bound_x3_interior.s - offset_x3,
        bound_x3_interior.e + offset_x3, bound_x1_interior.s - offset_x1,
        bound_x1_interior.e + offset_x1, bound_x2_interior.s,
        bound_x2_interior.e + 1, KOKKOS_LAMBDA(const int k, const int i,
                                               const int j) {
          Real primitiveLeft[NPRIM];
          Real primitiveRight[NPRIM];
          for (int index = 0; index < NPRIM; ++index) {
            primitiveLeft[index] =
                primitive(index, k, j - 1, i) +
                0.5 * InterpolateMC(primitive(index, k, j - 2, i),
                                    primitive(index, k, j - 1, i),
                                    primitive(index, k, j, i));
            primitiveRight[index] =
                primitive(index, k, j, i) -
                0.5 * InterpolateMC(primitive(index, k, j - 1, i),
                                    primitive(index, k, j, i),
                                    primitive(index, k, j + 1, i));
          }

          Real gcovFace[4][4];
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
              gcovFace[row][col] = covariant_metric(FACEX2, col, row, k, j, i);
            }
          }

          Real flux[NPRIM];
          Real maximumSignalSpeed;
          CalculateProjectedHlldFlux(
              kAdiabaticIndex, primitiveLeft, primitiveRight, gcovFace,
              metric_determinant(FACEX2, k, j, i), X2DIR, flux,
              maximumSignalSpeed);
          AlfvenVelocity(Vector3D::X2, k, j, i) = maximumSignalSpeed;
          for (int index = 0; index < NPRIM; ++index) {
            conservative.flux(X2DIR, index, k, j, i) = flux[index];
          }
        });

  if (pmb->pmy_mesh->ndim == 3)
    pmb->par_for(
        PARTHENON_AUTO_LABEL, bound_x2_interior.s, bound_x2_interior.e,
        bound_x1_interior.s, bound_x1_interior.e, bound_x3_interior.s,
        bound_x3_interior.e + 1, KOKKOS_LAMBDA(const int j, const int i,
                                               const int k) {
          Real primitiveLeft[NPRIM];
          Real primitiveRight[NPRIM];
          for (int index = 0; index < NPRIM; ++index) {
            primitiveLeft[index] =
                primitive(index, k - 1, j, i) +
                0.5 * InterpolateMC(primitive(index, k - 2, j, i),
                                    primitive(index, k - 1, j, i),
                                    primitive(index, k, j, i));
            primitiveRight[index] =
                primitive(index, k, j, i) -
                0.5 * InterpolateMC(primitive(index, k - 1, j, i),
                                    primitive(index, k, j, i),
                                    primitive(index, k + 1, j, i));
          }

          Real gcovFace[4][4];
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
              gcovFace[row][col] = covariant_metric(FACEX3, col, row, k, j, i);
            }
          }

          Real flux[NPRIM];
          Real maximumSignalSpeed;
          CalculateProjectedHlldFlux(
              kAdiabaticIndex, primitiveLeft, primitiveRight, gcovFace,
              metric_determinant(FACEX3, k, j, i), X3DIR, flux,
              maximumSignalSpeed);
          AlfvenVelocity(Vector3D::X3, k, j, i) = maximumSignalSpeed;
          for (int index = 0; index < NPRIM; ++index) {
            conservative.flux(X3DIR, index, k, j, i) = flux[index];
          }
        });

  return TaskStatus::complete;
}
