#include <memory>
#include <string>
#include <vector>

#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "initialization/variable_mnemonics.h"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "metric/CKS.h"
#include "metric/tensor_algebra.h"
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "task_list/task_list.h"

//----------------------------------------------------------------------------------------
// ProblemGenerator — Bondi accretion in CKS coordinates.
// Uses simple atmosphere + zero velocity initialization (no fixed analytic boundary).

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

  const auto package_metric = pmb->packages.Get("metric");
  const Real kerr_a = package_metric->Param<Real>("a");
  const Real kerr_a2 = kerr_a * kerr_a;
  const Real dexcise = package_metric->Param<Real>("dexcise");
  const Real pexcise = package_metric->Param<Real>("pexcise");
  const Real e_excise = pexcise / (kAdiabaticIndex - 1.0);

  PackIndexMap idxMap;
  auto primitive = resource->PackVariables(fnames, idxMap);

  const int iRHO = idxMap["density"].first;
  const int iENY = idxMap["energy"].first;
  const int iUX  = idxMap["weighted_velocity"].first;
  const int iENT = idxMap["entropy"].first;
  const int iBX  = enable_B ? idxMap["magnetic_field"].first : -1;
  const int iKEL = enable_heating ? idxMap["electron_entropy"].first : -1;

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
        const Real x = x_code[1];
        const Real y = x_code[2];
        const Real z = x_code[3];

        // Boyer-Lindquist radius from Cartesian KS coordinates.
        const Real rad2 = x * x + y * y + z * z;
        const Real r = Kokkos::fmax(
            Kokkos::sqrt((rad2 - kerr_a2 +
                          Kokkos::sqrt(SQR(rad2 - kerr_a2) + 4.0 * kerr_a2 * z * z)) /
                         2.0),
            1.0);

        // Metric and connection coefficients at this cell and its faces.
        Real gcov[4][4];
        CKS::CalculateCodeMetric(x_code, gcov, kerr_a);
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
          CKS::CalculateCodeMetric(x_code_loc[loc], gcov_loc, kerr_a);
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
            CKS::CalculateCodeMetric(x_plus, gp, kerr_a);
            CKS::CalculateCodeMetric(x_minus, gm, kerr_a);
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
                  0.5 * (dgcov[jj][ii][kk] + dgcov[kk][ii][jj] - dgcov[ii][jj][kk]);
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

        // Simple atmosphere + zero velocity initialization
        Real rho, eint, wvx1, wvx2, wvx3;
        if (r > 1.0) {
          rho = dexcise;
          eint = e_excise;
          wvx1 = 0.0;
          wvx2 = 0.0;
          wvx3 = 0.0;
        } else {
          rho = dexcise;
          eint = e_excise;
          wvx1 = 0.0;
          wvx2 = 0.0;
          wvx3 = 0.0;
        }

        primitive(iRHO, k, j, i) = rho;
        primitive(iENY, k, j, i) = eint;
        primitive(iUX,   k, j, i) = wvx1;
        primitive(iUX+1, k, j, i) = wvx2;
        primitive(iUX+2, k, j, i) = wvx3;
        if (enable_B) {
          primitive(iBX,   k, j, i) = 0.0;
          primitive(iBX+1, k, j, i) = 0.0;
          primitive(iBX+2, k, j, i) = 0.0;
        }
        primitive(iENT, k, j, i) =
            (kAdiabaticIndex - 1.0) * eint *
            Kokkos::pow(rho, -kAdiabaticIndex);
        if (enable_heating) {
          primitive(iKEL, k, j, i) = kFelInit * primitive(iENT, k, j, i);
        }
      });
}

void MeshPostInitialization(parthenon::Mesh * /* pm */,
                            parthenon::ParameterInput * /* pin */,
                            parthenon::MeshData<parthenon::Real> * /* md */) {
}
