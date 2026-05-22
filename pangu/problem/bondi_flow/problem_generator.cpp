#include <memory>
#include <string>
#include <vector>

#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "initialization/variable_mnemonics.h"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "metric/MKS.h"
#include "metric/tensor_algebra.h"
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "task_list/task_list.h"

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

    if (fh * f0 > 0.0) {
      T0 = Th;
      f0 = fh;
    } else {
      T1 = Th;
      f1 = fh;
    }
  }

  return Kokkos::max(0.5 * (T0 + T1), static_cast<parthenon::Real>(1.0e-12));
}

KOKKOS_INLINE_FUNCTION
void SolveBondiSolution(const parthenon::Real r, const parthenon::Real rs,
                        const parthenon::Real mdot,
                        const parthenon::Real adiabaticIndex,
                        parthenon::Real &rho, parthenon::Real &u,
                        parthenon::Real &ur) {
  const parthenon::Real n = 1.0 / (adiabaticIndex - 1.0);
  const parthenon::Real uc = Kokkos::sqrt(1.0 / (2.0 * rs));
  const parthenon::Real Vc = Kokkos::sqrt(uc * uc / (1.0 - 3.0 * uc * uc));
  const parthenon::Real Tc = -n * Vc * Vc / ((n + 1.0) * (n * Vc * Vc - 1.0));
  const parthenon::Real C1 = uc * rs * rs * Kokkos::pow(Tc, n);
  const parthenon::Real A = 1.0 + (1.0 + n) * Tc;
  const parthenon::Real C2 = A * A * (1.0 - 2.0 / rs + uc * uc);
  const parthenon::Real K = Kokkos::pow(4.0 * M_PI * C1 / mdot, 1.0 / n);
  const parthenon::Real Kn = Kokkos::pow(K, n);

  const parthenon::Real T = SolveBondiTemperature(r, C1, C2, n, rs);
  const parthenon::Real Tn = Kokkos::pow(T, n);

  rho = Tn / Kn;
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

  // Bondi parameters follow the Sisyphus defaults when not explicitly provided.
  const Real kBondiMdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
  const Real kBondiSonicRadius = pin->GetOrAddReal("bondi", "rs", 8.0);
  const Real kBondiInnerAtmosphereRadius =
      pin->GetOrAddReal("bondi", "rin", 10.0);
  const Real kBondiAtmosphereFactor =
      pin->GetOrAddReal("bondi", "atmosphere_factor", 1.0e-7);

  // MKS metric parameters used by device-side metric functions.
  const Real mks_h = pin->GetOrAddReal("metric", "h", 0.0);
  const Real mks_a = pin->GetOrAddReal("metric", "a", 0.0);

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
      "electron_entropy"};
  auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);

  auto covariant_metric = resource->Get("covariant_metric").data;
  auto contravariant_metric = resource->Get("contravariant_metric").data;
  auto metric_determinant = resource->Get("metric_determinant").data;
  auto connection = resource->Get("connection").data;

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

        const Real x_code_loc[4][4] = {
            {0.0, coords.Xc<X1DIR>(i), coords.Xc<X2DIR>(j),
             coords.Xc<X3DIR>(k)},
            {0.0, coords.Xf<X1DIR, X1DIR>(k, j, i),
             coords.Xf<X2DIR, X1DIR>(k, j, i),
             coords.Xf<X3DIR, X1DIR>(k, j, i)},
            {0.0, coords.Xf<X1DIR, X2DIR>(k, j, i),
             coords.Xf<X2DIR, X2DIR>(k, j, i),
             coords.Xf<X3DIR, X2DIR>(k, j, i)},
            {0.0, coords.Xf<X1DIR, X3DIR>(k, j, i),
             coords.Xf<X2DIR, X3DIR>(k, j, i),
             coords.Xf<X3DIR, X3DIR>(k, j, i)}};

        for (int loc = 0; loc < 4; ++loc) {
          Real gcov_loc[4][4];
          Real gcon_loc[4][4];
          MKS::CalculateCodeMetric(x_code_loc[loc], gcov_loc, mks_h, mks_a);
          invert(gcov_loc, gcon_loc);
          metric_determinant(loc, k, j, i) = determinant(gcov_loc);
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
              covariant_metric(loc, col, row, k, j, i) = gcov_loc[row][col];
              contravariant_metric(loc, col, row, k, j, i) = gcon_loc[row][col];
            }
          }
        }

        constexpr Real MetricDiffDelta = 1.0e-5;

        Real dgcov[4][4][4];
        for (int dir = 0; dir < 4; ++dir) {
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
              dgcov[dir][row][col] = 0.0;
            }
          }
          if (dir > 0) {
            Real gp[4][4], gm[4][4];
            Real x_plus[4] = {0.0, x1, x2, x3};
            Real x_minus[4] = {0.0, x1, x2, x3};
            x_plus[dir] += MetricDiffDelta;
            x_minus[dir] -= MetricDiffDelta;

            MKS::CalculateCodeMetric(x_plus, gp, mks_h, mks_a);
            MKS::CalculateCodeMetric(x_minus, gm, mks_h, mks_a);
            for (int row = 0; row < 4; ++row) {
              for (int col = 0; col < 4; ++col) {
                dgcov[dir][row][col] =
                    (gp[row][col] - gm[row][col]) / (2.0 * MetricDiffDelta);
              }
            }
          }
        }

        Real conn_cov[4][4][4];
        for (int ii = 0; ii < 4; ++ii) {
          for (int jj = 0; jj < 4; ++jj) {
            for (int kk = 0; kk < 4; ++kk) {
              conn_cov[ii][jj][kk] =
                  0.5 *
                  (dgcov[jj][ii][kk] + dgcov[kk][ii][jj] - dgcov[ii][jj][kk]);
            }
          }
        }

        for (int ii = 0; ii < 4; ++ii) {
          for (int jj = 0; jj < 4; ++jj) {
            for (int kk = 0; kk < 4; ++kk) {
              Real conn_val = 0.0;
              for (int ll = 0; ll < 4; ++ll) {
                conn_val += gcon[ii][ll] * conn_cov[ll][jj][kk];
              }
              connection(ii, jj, kk, k, j, i) = conn_val;
            }
          }
        }

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

        primitive(RHO, k, j, i) =
            Kokkos::max(rho, kBondiAtmosphereFactor);
        primitive(ENY, k, j, i) =
            Kokkos::max(eint, kBondiAtmosphereFactor * 1.0e-6);
        primitive(UX1, k, j, i) = wvx1;
        primitive(UX2, k, j, i) = wvx2;
        primitive(UX3, k, j, i) = wvx3;
        primitive(BX1, k, j, i) = 0.0;
        primitive(BX2, k, j, i) = 0.0;
        primitive(BX3, k, j, i) = 0.0;
        primitive(ENT, k, j, i) =
            (kAdiabaticIndex - 1.0) * primitive(ENY, k, j, i) *
            Kokkos::pow(primitive(RHO, k, j, i), -kAdiabaticIndex);
        primitive(KEL, k, j, i) = kFelInit * primitive(ENT, k, j, i);
      });
}
