#include <Kokkos_Random.hpp>
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
#include "physics/state_calculation.h"
#include "prolong_restrict/prolong_restrict.hpp"
#include "task_list/task_list.h"

KOKKOS_FUNCTION
Real lfish_calc(Real r, Real a) {
  return (
      ((Kokkos::pow(a, 2) - 2. * a * Kokkos::sqrt(r) + Kokkos::pow(r, 2)) *
       ((-2. * a * r *
         (Kokkos::pow(a, 2) - 2. * a * Kokkos::sqrt(r) + Kokkos::pow(r, 2))) /
            Kokkos::sqrt(2. * a * Kokkos::sqrt(r) + (-3. + r) * r) +
        ((a + (-2. + r) * Kokkos::sqrt(r)) *
         (Kokkos::pow(r, 3) + Kokkos::pow(a, 2) * (2. + r))) /
            Kokkos::sqrt(1 + (2. * a) / Kokkos::pow(r, 1.5) - 3. / r))) /
      (Kokkos::pow(r, 3) *
       Kokkos::sqrt(2. * a * Kokkos::sqrt(r) + (-3. + r) * r) *
       (Kokkos::pow(a, 2) + (-2. + r) * r)));
}

KOKKOS_INLINE_FUNCTION
void CalculateBoyerLindquistMetric(const Real r, const Real th, const Real a,
                                   Real gcov[4][4]) {
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      gcov[row][col] = 0.0;
    }
  }

  const Real cth = Kokkos::cos(th);
  const Real sth = Kokkos::sin(th);
  const Real sth2 = sth * sth;
  const Real r2 = r * r;
  const Real a2 = a * a;
  const Real mu = 1.0 + a2 * cth * cth / r2;
  const Real delta = 1.0 - 2.0 / r + a2 / r2;
  const Real inv_r_mu = 1.0 / (r * mu);

  gcov[0][0] = -1.0 + 2.0 * inv_r_mu;
  gcov[0][3] = -2.0 * a * sth2 * inv_r_mu;
  gcov[1][1] = mu / delta;
  gcov[2][2] = r2 / mu;
  gcov[3][0] = gcov[0][3];
  gcov[3][3] = r2 * sth2 * (1.0 + a2 / r2 + 2.0 * a2 * sth2 / (r * r2 * mu));
}

KOKKOS_INLINE_FUNCTION
Real SolveTemporalVelocity(const Real gcov[4][4], const Real u1, const Real u2,
                           const Real u3) {
  const Real AA = gcov[0][0];
  const Real BB = 2.0 * (gcov[0][1] * u1 + gcov[0][2] * u2 + gcov[0][3] * u3);
  const Real CC =
      1.0 + gcov[1][1] * u1 * u1 + gcov[2][2] * u2 * u2 + gcov[3][3] * u3 * u3 +
      2.0 *
          (gcov[1][2] * u1 * u2 + gcov[1][3] * u1 * u3 + gcov[2][3] * u2 * u3);
  const Real discr = BB * BB - 4.0 * AA * CC;
  return (-BB - Kokkos::sqrt(discr)) / (2.0 * AA);
}

KOKKOS_INLINE_FUNCTION
void TransformBLToCodeFourVelocity(const Real x_code[4], const Real a,
                                   const Real ucon_bl_in[4],
                                   Real ucon_code[4]) {
  const Real x = x_code[1];
  const Real y = x_code[2];
  const Real z = x_code[3];
  const Real rho_xy = Kokkos::sqrt(x * x + y * y);
  const Real r = Kokkos::sqrt(rho_xy * rho_xy + z * z);
  const Real th = Kokkos::atan2(rho_xy, z);
  const Real phi = Kokkos::atan2(y, x) - Kokkos::atan2(a, r);

  Real gcov_bl[4][4];
  CalculateBoyerLindquistMetric(r, th, a, gcov_bl);

  Real ucon_bl[4];
  for (int row = 0; row < 4; ++row) {
    ucon_bl[row] = ucon_bl_in[row];
  }
  ucon_bl[0] = SolveTemporalVelocity(gcov_bl, ucon_bl[1], ucon_bl[2],
                                     ucon_bl[3]);

  const Real sth = Kokkos::sin(th);
  const Real cth = Kokkos::cos(th);
  const Real cphi = Kokkos::cos(phi);
  const Real sphi = Kokkos::sin(phi);
  const Real rc = r * cphi - a * sphi;
  const Real rs = r * sphi + a * cphi;

  Real trans[4][4];
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      trans[row][col] = 0.0;
    }
  }
  trans[0][0] = 1.0;
  trans[1][1] = sth * cphi;
  trans[1][2] = rc * cth;
  trans[1][3] = -rs * sth;
  trans[2][1] = sth * sphi;
  trans[2][2] = rs * cth;
  trans[2][3] = rc * sth;
  trans[3][1] = cth;
  trans[3][2] = -r * sth;

  for (int row = 0; row < 4; ++row) {
    ucon_code[row] = 0.0;
    for (int col = 0; col < 4; ++col) {
      ucon_code[row] += trans[row][col] * ucon_bl[col];
    }
  }
}

void ProblemGenerator(parthenon::MeshBlock *pmb,
                      parthenon::ParameterInput *pin) {
  using namespace parthenon;

  const auto package_core = pmb->packages.Get("core");
  auto &resource = pmb->meshblock_data.Get();
  const Real kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");
  const Real kFelInit = package_core->Param<Real>("fel_0");

  const Real kerr_a = pin->GetOrAddReal("metric", "a", 0.9375);

  const Real rin = pin->GetOrAddReal("fm_torus", "rin", 6.0);
  const Real rmax = pin->GetOrAddReal("fm_torus", "rmax", 12.0);
  const Real kappa = pin->GetOrAddReal("fm_torus", "kappa", 1.0e-3);
  const Real perturbation =
      pin->GetOrAddReal("fm_torus", "perturbation", 4.0e-2);
  const Real beta_target = pin->GetOrAddReal("fm_torus", "beta", 100.0);
  const Real aphi_rho_cut = pin->GetOrAddReal("fm_torus", "aphi_rho_cut", 0.2);

  const Real a2 = kerr_a * kerr_a;
  const Real l = lfish_calc(rmax, kerr_a);

  const Real thin = M_PI_2;  ///< 环的中心在赤道面
  const Real sthin = Kokkos::sin(thin);
  const Real cthin = Kokkos::cos(thin);
  const Real DDin = rin * rin - 2. * rin + kerr_a * kerr_a;
  const Real AAin =
      (rin * rin + kerr_a * kerr_a) * (rin * rin + kerr_a * kerr_a) -
      DDin * kerr_a * kerr_a * sthin * sthin;
  const Real SSin = rin * rin + kerr_a * kerr_a * cthin * cthin;

  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
      "electron_entropy"
  };
  auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);

  auto covariant_metric = resource->Get("covariant_metric").data;
  auto contravariant_metric = resource->Get("contravariant_metric").data;
  auto metric_determinant = resource->Get("metric_determinant").data;
  auto connection = resource->Get("connection").data;

  auto cellbounds = pmb->cellbounds;
  const auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
  const auto ib_interior = cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb_interior = cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb_interior = cellbounds.GetBoundsK(IndexDomain::interior);
  auto coords = pmb->coords;
  Kokkos::Random_XorShift64_Pool<> random_pool(0);

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto generator = random_pool.get_state();
        const Real random = generator.drand(-1.0, 1.0);
        random_pool.free_state(generator);

        const Real x1 = coords.Xc<X1DIR>(i);
        const Real x2 = coords.Xc<X2DIR>(j);
        const Real x3 = coords.Xc<X3DIR>(k);

        const Real x_code[4] = {0.0, x1, x2, x3};
        Real y[4];
        CKS::CalculatePhysicalCoordinates(x_code, y);
        const Real x = y[1];
        const Real yy = y[2];
        const Real z = y[3];
        const Real rho_xy = Kokkos::sqrt(x * x + yy * yy);
        const Real r = Kokkos::sqrt(rho_xy * rho_xy + z * z);
        const Real th = Kokkos::atan2(rho_xy, z);
        const Real sth = Kokkos::sin(th);
        const Real cth = Kokkos::cos(th);

        Real gcov[4][4];
        CKS::CalculateCodeMetric(x_code, gcov, kerr_a);

        Real gcov_bl[4][4];
        CalculateBoyerLindquistMetric(r, th, kerr_a, gcov_bl);

        Real gcon[4][4];
        invert(gcov, gcon);

        const Real gdet = determinant(gcov);
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

        const Real alpha = 1.0 / Kokkos::sqrt(-gcon[0][0]);
        const Real beta1 = gcon[0][1] * alpha * alpha;
        const Real beta2 = gcon[0][2] * alpha * alpha;
        const Real beta3 = gcon[0][3] * alpha * alpha;

        Real rho = 0.0;
        Real eint = 0.0;
        Real ent = 0.0;
        Real wvx1 = 0.0;
        Real wvx2 = 0.0;
        Real wvx3 = 0.0;

        const Real DD = r * r - 2.0 * r + a2;
        const Real AA = (r * r + a2) * (r * r + a2) - DD * a2 * sth * sth;
        const Real SS = r * r + a2 * cth * cth;

        Real lnh = 1.0;

        if (r >= rin) {
          lnh = 0.5 * Kokkos::log(
                          (1. + Kokkos::sqrt(1. + 4. * (l * l * SS * SS) * DD /
                                                      (AA * sth * AA * sth))) /
                          (SS * DD / AA)) -
                0.5 * Kokkos::sqrt(1. + 4. * (l * l * SS * SS) * DD /
                                            (AA * AA * sth * sth)) -
                2. * kerr_a * r * l / AA -
                (0.5 * Kokkos::log(
                           (1. + Kokkos::sqrt(
                                     1. + 4. * (l * l * SSin * SSin) * DDin /
                                              (AAin * AAin * sthin * sthin))) /
                           (SSin * DDin / AAin)) -
                 0.5 * Kokkos::sqrt(1. + 4. * (l * l * SSin * SSin) * DDin /
                                             (AAin * AAin * sthin * sthin)) -
                 2. * kerr_a * rin * l / AAin);
        }

        if (lnh >= 0.0 && r >= rin) {
          const Real hm1 = Kokkos::exp(lnh) - 1.0;
          if (hm1 > 0.0) {
            rho = Kokkos::pow(
                hm1 * (kAdiabaticIndex - 1.0) / (kappa * kAdiabaticIndex),
                1.0 / (kAdiabaticIndex - 1.0));
            eint = kappa * Kokkos::pow(rho, kAdiabaticIndex) /
                   (kAdiabaticIndex - 1.0);

            eint *= (1.0 + perturbation * random);
            ent = (kAdiabaticIndex - 1) * eint * Kokkos::pow(rho, -kAdiabaticIndex);

            const Real expm2chi = SS * SS * DD / (AA * AA * sth * sth);
            const Real up1 = Kokkos::sqrt(
                (-1.0 + Kokkos::sqrt(1.0 + 4.0 * l * l * expm2chi)) / 2.0);
            const Real up_bl = 2.0 * kerr_a * r *
                                   Kokkos::sqrt(1.0 + up1 * up1) /
                                   Kokkos::sqrt(AA * SS * DD) +
                               Kokkos::sqrt(SS / AA) * up1 / sth;
            const Real u0_bl = SolveTemporalVelocity(gcov_bl, 0.0, 0.0, up_bl);
            const Real ucon_bl[4] = {u0_bl, 0.0, 0.0, up_bl};
            Real ucon_code[4];
            TransformBLToCodeFourVelocity(x_code, kerr_a, ucon_bl,
                                          ucon_code);

            wvx1 = ucon_code[1] + beta1 * ucon_code[0];
            wvx2 = ucon_code[2] + beta2 * ucon_code[0];
            wvx3 = ucon_code[3] + beta3 * ucon_code[0];
          }
        }

        primitive(RHO, k, j, i) = rho;
        primitive(ENY, k, j, i) = eint;
        primitive(UX1, k, j, i) = wvx1;
        primitive(UX2, k, j, i) = wvx2;
        primitive(UX3, k, j, i) = wvx3;
        primitive(BX1, k, j, i) = 0.0;
        primitive(BX2, k, j, i) = 0.0;
        primitive(BX3, k, j, i) = 0.0;
        primitive(ENT, k, j, i) = ent;
        primitive(KEL, k, j, i) = kFelInit * ent;
      });

  Real rhomax = 0.0;
  Real umax = 0.0;

  pmb->par_reduce(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_rhomax) {
        local_rhomax =
            Kokkos::max(local_rhomax, primitive(RHO, k, j, i));
      },
      Kokkos::Max<Real>(rhomax));
  
  pmb->par_reduce(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_umax) {
        local_umax = Kokkos::max(local_umax, primitive(ENY, k, j, i));
      },
      Kokkos::Max<Real>(umax));
}

void MeshPostInitialization(parthenon::Mesh *pmesh,
                            parthenon::ParameterInput *pin,
                            parthenon::MeshData<Real> *md) {
  using namespace parthenon;

  (void)md;

  const auto package_core = pmesh->packages.Get("core");
  const Real kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");
  const Real beta_target = pin->GetOrAddReal("fm_torus", "beta", 100.0);
  const Real aphi_rho_cut = pin->GetOrAddReal("fm_torus", "aphi_rho_cut", 0.2);

  Real local_rhomax = 0.0;
  Real local_umax = 0.0;

  for (const auto &pmb : pmesh->block_list) {
    auto &resource = pmb->meshblock_data.Get();
    PackIndexMap primitiveIndexMap;
    const std::vector<std::string> primitive_tags = {
        "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
        "electron_entropy"};
    auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);

    auto cellbounds = pmb->cellbounds;
    const auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
    const auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
    const auto kb = cellbounds.GetBoundsK(IndexDomain::entire);

    Real block_rhomax = 0.0;
    Real block_umax = 0.0;

    pmb->par_reduce(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_max) {
          local_max = Kokkos::max(local_max, primitive(RHO, k, j, i));
        },
        Kokkos::Max<Real>(block_rhomax));

    pmb->par_reduce(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_max) {
          local_max = Kokkos::max(local_max, primitive(ENY, k, j, i));
        },
        Kokkos::Max<Real>(block_umax));

    local_rhomax = Kokkos::max(local_rhomax, block_rhomax);
    local_umax = Kokkos::max(local_umax, block_umax);
  }

#ifdef MPI_PARALLEL
  Real global_rhomax = local_rhomax;
  Real global_umax = local_umax;
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &global_rhomax, 1,
                                    MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &global_umax, 1,
                                    MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
#else
  const Real global_rhomax = local_rhomax;
  const Real global_umax = local_umax;
#endif

  if (Globals::my_rank == 0) {
    printf("Maximum initial density: %e\n", global_rhomax);
    printf("Maximum initial energy: %e\n", global_umax);
  }

  for (const auto &pmb : pmesh->block_list) {
    auto &resource = pmb->meshblock_data.Get();
    PackIndexMap primitiveIndexMap;
    const std::vector<std::string> primitive_tags = {
        "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
        "electron_entropy"};
    auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);

    auto covariant_metric = resource->Get("covariant_metric").data;
    auto contravariant_metric = resource->Get("contravariant_metric").data;
    auto metric_determinant = resource->Get("metric_determinant").data;

    auto cellbounds = pmb->cellbounds;
    const auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
    const auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
    const auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
    const auto ib_interior = cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb_interior = cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb_interior = cellbounds.GetBoundsK(IndexDomain::interior);
    auto coords = pmb->coords;

    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          primitive(RHO, k, j, i) /= global_rhomax;
          primitive(ENY, k, j, i) /= global_rhomax;
        });

    const int ni = ib.e - ib.s + 1;
    const int nj = jb.e - jb.s + 1;
    const int nk = kb.e - kb.s + 1;
    Kokkos::View<Real ***> vectorPotential("vectorPotential", nk, nj, ni);

    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          vectorPotential(k - kb.s, j - jb.s, i - ib.s) = 0.0;
        });

    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s + 1, jb.e, ib.s + 1, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real rho_average =
              0.25 * (primitive(RHO, k, j, i) + primitive(RHO, k, j, i - 1) +
                      primitive(RHO, k, j - 1, i) +
                      primitive(RHO, k, j - 1, i - 1));
          const Real expr = rho_average - aphi_rho_cut;
          vectorPotential(k - kb.s, j - jb.s, i - ib.s) =
              (expr > 0.0) ? expr : 0.0;
        });

    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb_interior.s, kb_interior.e, jb_interior.s,
        jb_interior.e, ib_interior.s, ib_interior.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const int kk = k - kb.s;
          const int jj = j - jb.s;
          const int ii = i - ib.s;

          const Real sqrt_abs_gdet = Kokkos::sqrt(
              Kokkos::fabs(metric_determinant(CENTER, k, j, i)));
          const Real dx1 = coords.Dxc<X1DIR>(i);
          const Real dx2 = coords.Dxc<X2DIR>(j);

          const Real a00 = vectorPotential(kk, jj - 1, ii - 1);
          const Real a01 = vectorPotential(kk, jj, ii - 1);
          const Real a10 = vectorPotential(kk, jj - 1, ii);
          const Real a11 = vectorPotential(kk, jj, ii);

          primitive(BX1, k, j, i) =
              -(a00 - a01 + a10 - a11) / (2.0 * dx2 * sqrt_abs_gdet);
          primitive(BX2, k, j, i) =
              (a00 + a01 - a10 - a11) / (2.0 * dx1 * sqrt_abs_gdet);
        });
  }

  Real local_bsq_max = 0.0;
  for (const auto &pmb : pmesh->block_list) {
    auto &resource = pmb->meshblock_data.Get();
    PackIndexMap primitiveIndexMap;
    const std::vector<std::string> primitive_tags = {
        "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
        "electron_entropy"};
    auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);

    auto covariant_metric = resource->Get("covariant_metric").data;
    auto contravariant_metric = resource->Get("contravariant_metric").data;

    auto cellbounds = pmb->cellbounds;
    const auto ib_interior = cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb_interior = cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb_interior = cellbounds.GetBoundsK(IndexDomain::interior);

    Real block_bsq_max = 0.0;
    pmb->par_reduce(
        PARTHENON_AUTO_LABEL, kb_interior.s, kb_interior.e, jb_interior.s,
        jb_interior.e, ib_interior.s, ib_interior.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i,
                      Real &local_max) {
          Real gcov[4][4];
          Real gcon[4][4];
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
              gcov[row][col] = covariant_metric(CENTER, col, row, k, j, i);
              gcon[row][col] = contravariant_metric(CENTER, col, row, k, j, i);
            }
          }

          Real primitive_c_array[NPRIM];
          for (int index = 0; index < NPRIM; ++index) {
            primitive_c_array[index] = primitive(index, k, j, i);
          }

          State state;
          CalculateState(primitive_c_array, gcov, gcon, state);
          local_max = Kokkos::max(local_max, state.bsq);
        },
        Kokkos::Max<Real>(block_bsq_max));

    local_bsq_max = Kokkos::max(local_bsq_max, block_bsq_max);
  }

#ifdef MPI_PARALLEL
  Real global_bsq_max = local_bsq_max;
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &global_bsq_max, 1,
                                    MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
#else
  const Real global_bsq_max = local_bsq_max;
#endif

  const Real beta_min =
      2.0 * (kAdiabaticIndex - 1.0) * (global_umax / global_rhomax) /
      (global_bsq_max + 1.0e-30);
  const Real magnetic_norm = Kokkos::sqrt(beta_min / beta_target);

  for (const auto &pmb : pmesh->block_list) {
    auto &resource = pmb->meshblock_data.Get();
    PackIndexMap primitiveIndexMap;
    const std::vector<std::string> primitive_tags = {
        "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
        "electron_entropy"};
    auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);

    auto cellbounds = pmb->cellbounds;
    const auto ib_interior = cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb_interior = cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb_interior = cellbounds.GetBoundsK(IndexDomain::interior);

    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb_interior.s, kb_interior.e, jb_interior.s,
        jb_interior.e, ib_interior.s, ib_interior.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          primitive(BX1, k, j, i) *= magnetic_norm;
          primitive(BX2, k, j, i) *= magnetic_norm;
          primitive(BX3, k, j, i) *= magnetic_norm;
        });
  }

  if (Globals::my_rank == 0) {
    printf("Maximum initial magnetic bsq: %e\n", global_bsq_max);
    printf("Target beta: %e, beta_min: %e, magnetic normalization: %e\n",
           beta_target, beta_min, magnetic_norm);
  }
}
