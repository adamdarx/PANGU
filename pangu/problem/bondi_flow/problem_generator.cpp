#include <memory>
#include <string>
#include <vector>

#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include "initialization/variable_mnemonics.h"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "metric/MKS.h"
#include "metric/tensor_algebra.h"
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "task_list/task_list.h"
#include <parthenon/package.hpp>

//----------------------------------------------------------------------------------------
// Bondi accretion analytic solution — Hawley, Smarr, & Wilson 1984 (HSW).
// Inlined from the former bondi_solver.h.

KOKKOS_INLINE_FUNCTION
parthenon::Real BondiTemperatureResidual(const parthenon::Real T,
                                         const parthenon::Real r,
                                         const parthenon::Real C1,
                                         const parthenon::Real C2,
                                         const parthenon::Real n) {
  const parthenon::Real T_safe =
      Kokkos::max(T, static_cast<parthenon::Real>(1.0e-14));
  const parthenon::Real A = 1.0 + (1.0 + n) * T_safe;
  const parthenon::Real B = C1 / (r * r * Kokkos::pow(T_safe, n));
  return A * A * (1.0 - 2.0 / r + B * B) - C2;
}

KOKKOS_INLINE_FUNCTION
parthenon::Real SolveBondiTemperature(const parthenon::Real r,
                                      const parthenon::Real C1,
                                      const parthenon::Real C2,
                                      const parthenon::Real n,
                                      const parthenon::Real rs) {
  const parthenon::Real Tinf = (Kokkos::sqrt(C2) - 1.0) / (n + 1.0);
  const parthenon::Real Tnear =
      Kokkos::pow(C1 * Kokkos::sqrt(2.0 / (r * r * r)), 1.0 / n);
  const parthenon::Real Tmin = (r < rs) ? Tinf : Kokkos::max(Tnear, Tinf);
  const parthenon::Real Tmax = (r < rs) ? Tnear : 1.0;
  parthenon::Real T0 = Tmin;
  parthenon::Real T1 = Tmax;
  parthenon::Real f0 = BondiTemperatureResidual(T0, r, C1, C2, n);
  parthenon::Real f1 = BondiTemperatureResidual(T1, r, C1, C2, n);
  if (f0 * f1 > 0.0) {
    return Kokkos::max(Tinf, static_cast<parthenon::Real>(1.0e-12));
  }
  const parthenon::Real rtol = 1.0e-12;
  const parthenon::Real ftol = 1.0e-14;
  const parthenon::Real epsT = rtol * (Tmin + Tmax);
  for (int iter = 0; iter < 128; ++iter) {
    const parthenon::Real Th = 0.5 * (T0 + T1);
    const parthenon::Real fh = BondiTemperatureResidual(Th, r, C1, C2, n);
    if (Kokkos::abs(fh) <= ftol || Kokkos::abs(Th - T0) <= epsT ||
        Kokkos::abs(Th - T1) <= epsT) {
      return Kokkos::max(Th, static_cast<parthenon::Real>(1.0e-12));
    }
    if (fh * f0 > 0.0) { T0 = Th; f0 = fh; }
    else { T1 = Th; f1 = fh; }
  }
  return Kokkos::max(0.5 * (T0 + T1), static_cast<parthenon::Real>(1.0e-12));
}

KOKKOS_INLINE_FUNCTION
void SolveBondiSolution(const parthenon::Real r,
                        const parthenon::Real rs,
                        const parthenon::Real /* mdot */,
                        const parthenon::Real adiabaticIndex,
                        parthenon::Real &rho,
                        parthenon::Real &u,
                        parthenon::Real &ur) {
  const parthenon::Real n = 1.0 / (adiabaticIndex - 1.0);
  const parthenon::Real uc = Kokkos::sqrt(1.0 / (2.0 * rs));
  const parthenon::Real Vc = Kokkos::sqrt(uc * uc / (1.0 - 3.0 * uc * uc));
  const parthenon::Real Tc = -n * Vc * Vc / ((n + 1.0) * (n * Vc * Vc - 1.0));
  const parthenon::Real C1 = uc * rs * rs * Kokkos::pow(Tc, n);
  const parthenon::Real A = 1.0 + (1.0 + n) * Tc;
  const parthenon::Real C2 = A * A * (1.0 - 2.0 / rs + uc * uc);
  const parthenon::Real T = SolveBondiTemperature(r, C1, C2, n, rs);
  const parthenon::Real Tn = Kokkos::pow(T, n);
  rho = Tn;
  u = rho * T * n;
  ur = -C1 / (Tn * r * r);
}

void ProblemGenerator(parthenon::MeshBlock *pmb,
                      parthenon::ParameterInput *pin) {
  using namespace parthenon;

  const auto package_core = pmb->packages.Get("core");
  auto &resource = pmb->meshblock_data.Get();
  const auto kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");
  const auto kFelInit = package_core->Param<Real>("fel_0");
  const auto enable_B = package_core->Param<bool>("enable_B");
  const auto enable_heating = package_core->Param<bool>("enable_heating");
  const auto& fnames = package_core->Param<std::vector<std::string>>("primitive_field_names");

  // Bondi parameters follow the Sisyphus defaults when not explicitly provided.
  const Real kBondiMdot = pin->GetReal("bondi", "mdot");
  const Real kBondiSonicRadius = pin->GetReal("bondi", "rs");
  const Real kBondiInnerAtmosphereRadius =
      pin->GetOrAddReal("bondi", "rin", 10.0);
  const Real kBondiAtmosphereFactor =
      pin->GetOrAddReal("bondi", "atmosphere_factor", 1.0e-7);

  // MKS metric parameters used by device-side metric functions.
  const Real mks_h = pin->GetOrAddReal("metric", "h", 0.0);
  const Real mks_a = pin->GetOrAddReal("metric", "a", 0.0);

  PackIndexMap idxMap;
  auto primitive = resource->PackVariables(fnames, idxMap);

  const int iRHO = idxMap["density"].first;
  const int iENY = idxMap["energy"].first;
  const int iUX  = idxMap["weighted_velocity"].first;
  const int iENT = idxMap["entropy"].first;
  const int iBX  = enable_B ? idxMap["magnetic_field"].first : -1;
  const int iKEL = enable_heating ? idxMap["electron_entropy"].first : -1;

  auto cellbounds = pmb->cellbounds;
  const auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
  auto coords = pmb->coords;

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x1 = coords.Xc<X1DIR>(i);
        const Real x2 = coords.Xc<X2DIR>(j);
        const Real x3 = coords.Xc<X3DIR>(k);

        const Real x_code[4] = {0.0, x1, x2, x3};
        Real y[4];
        MKS::CalculatePhysicalCoordinates(x_code, y, mks_h, mks_a);
        Real gcov[4][4];
        MKS::CalculateCodeMetric(x_code, gcov, mks_h, mks_a);

        Real gcon[4][4];
        invert(gcov, gcon);

        const Real r = y[1];
        const Real alpha = 1.0 / Kokkos::sqrt(-gcon[0][0]);
        const Real beta1 = gcon[0][1] * alpha * alpha;
        const Real beta2 = gcon[0][2] * alpha * alpha;
        const Real beta3 = gcon[0][3] * alpha * alpha;

        Real rho = kBondiAtmosphereFactor;
        Real eint = kBondiAtmosphereFactor * 1.0e-3;
        Real wvx1 = 0.0;
        Real wvx2 = 0.0;
        Real wvx3 = 0.0;

        if (r >= kBondiInnerAtmosphereRadius) {
          Real ur = 0.0;
          SolveBondiSolution(r, kBondiSonicRadius, kBondiMdot, kAdiabaticIndex,
                             rho, eint, ur);

          // Bondi solution provides u^r in physical radius; convert to internal
          // x1 velocity.
          const Real u1 = ur / r;

          const Real AA = gcov[0][0];
          const Real BB = 2.0 * gcov[0][1] * u1;
          const Real CC = 1.0 + gcov[1][1] * u1 * u1;
          const Real discr = Kokkos::max(BB * BB - 4.0 * AA * CC, 0.0);
          const Real u0 = (-BB - Kokkos::sqrt(discr)) / (2.0 * AA);
          const Real Gamma = alpha * u0;

          wvx1 = u1 + Gamma * beta1 / alpha;
          wvx2 = Gamma * beta2 / alpha;
          wvx3 = Gamma * beta3 / alpha;
        }

        primitive(iRHO, k, j, i) =
            Kokkos::max(rho, kBondiAtmosphereFactor);
        primitive(iENY, k, j, i) =
            Kokkos::max(eint, kBondiAtmosphereFactor * 1.0e-6);
        primitive(iUX,   k, j, i) = wvx1;
        primitive(iUX+1, k, j, i) = wvx2;
        primitive(iUX+2, k, j, i) = wvx3;
        if (enable_B) {
          primitive(iBX,   k, j, i) = 0.0;
          primitive(iBX+1, k, j, i) = 0.0;
          primitive(iBX+2, k, j, i) = 0.0;
        }
        primitive(iENT, k, j, i) =
            (kAdiabaticIndex - 1.0) * primitive(iENY, k, j, i) *
            Kokkos::pow(primitive(iRHO, k, j, i), -kAdiabaticIndex);
        if (enable_heating) {
          primitive(iKEL, k, j, i) = kFelInit * primitive(iENT, k, j, i);
        }
      });
}

void MeshPostInitialization(parthenon::Mesh *pmesh,
                            parthenon::ParameterInput *pin,
                            parthenon::MeshData<Real> *md) {
  using namespace parthenon;
}
