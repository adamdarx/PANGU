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
#include "metric/christoffel.h"
#include "metric/tensor_algebra.h"
#include "parthenon/driver.hpp"
#include "physics/state_calculation.h"
#include "prolong_restrict/prolong_restrict.hpp"
#include "task_list/task_list.h"
#include <parthenon/package.hpp>

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
  gcov[2][2] = r2 * mu;
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
  // 通过扁球面方程求解 KS 径向坐标: r⁴ - (r_cart² - a²) r² - a² z² = 0
  // 对齐 AthenaK GetKerrSchildCoordinates / GetBoyerLindquistCoordinates
  const Real rad2 = x*x + y*y + z*z;
  const Real a2 = a*a;
  const Real r = Kokkos::fmax(Kokkos::sqrt((rad2 - a2 + Kokkos::sqrt(SQR(rad2 - a2) +
                               4.0*a2*z*z)) / 2.0), 1.0);
  const Real th = (Kokkos::fabs(z/r) < 1.0) ? Kokkos::acos(z/r) :
                   ((z > 0) ? 0.0 : M_PI);

  // Step 1: 计算 BL 度规并求解 u^t（对齐 AthenaK CalculateVelocityInTorus）
  Real gcov_bl[4][4];
  CalculateBoyerLindquistMetric(r, th, a, gcov_bl);

  Real ucon_bl[4];
  for (int row = 0; row < 4; ++row) {
    ucon_bl[row] = ucon_bl_in[row];
  }
  ucon_bl[0] = SolveTemporalVelocity(gcov_bl, ucon_bl[1], ucon_bl[2],
                                     ucon_bl[3]);

  // Step 2: BL → KS 速度变换（对齐 AthenaK TransformVector: pa0 = a0_bl + 2r/Δ * a1_bl）
  // u^t_KS = u^t_BL + (2r/Δ) u^r_BL,  u^φ_KS = u^φ_BL + (a/Δ) u^r_BL
  const Real delta = r*r - 2.0*r + a2;
  Real ucon_ks[4];
  ucon_ks[0] = ucon_bl[0] + (2.0 * r / delta) * ucon_bl[1];
  ucon_ks[1] = ucon_bl[1];
  ucon_ks[2] = ucon_bl[2];
  ucon_ks[3] = ucon_bl[3] + (a / delta) * ucon_bl[1];

  // Step 3: KS → CKS 笛卡尔 Jacobian（对齐 AthenaK TransformVector 空间分量）
  // 所有 Jacobian 分量直接用 (x, y, r, θ, a) 表达，不显式计算 φ_KS
  // 这与 AthenaK GetKerrSchildCoordinates 将中间坐标视为 KS 坐标的方式一致
  const Real inv_r2pa2 = 1.0 / (r*r + a2);
  const Real rho2 = x*x + y*y;
  const Real sqrt_r2pa2_over_rho2 =
      (rho2 > 1e-30) ? Kokkos::sqrt((r*r + a2) / rho2) : 0.0;

  Real trans[4][4];
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      trans[row][col] = 0.0;
    }
  }
  trans[0][0] = 1.0;
  trans[1][1] = (r*x + a*y) * inv_r2pa2;          // ∂x/∂r_KS（对齐 AthenaK）
  trans[1][2] = x * z / r * sqrt_r2pa2_over_rho2;  // ∂x/∂θ_KS（对齐 AthenaK）
  trans[1][3] = -y;                                // ∂x/∂φ_KS（对齐 AthenaK）
  trans[2][1] = (r*y - a*x) * inv_r2pa2;          // ∂y/∂r_KS（对齐 AthenaK）
  trans[2][2] = y * z / r * sqrt_r2pa2_over_rho2;  // ∂y/∂θ_KS（对齐 AthenaK）
  trans[2][3] = x;                                 // ∂y/∂φ_KS（对齐 AthenaK）
  trans[3][1] = z / r;                             // ∂z/∂r（对齐 AthenaK）
  trans[3][2] = -r * Kokkos::sqrt(rho2 * inv_r2pa2);  // ∂z/∂θ = -r·sinθ（对齐 AthenaK）

  for (int row = 0; row < 4; ++row) {
    ucon_code[row] = 0.0;
    for (int col = 0; col < 4; ++col) {
      ucon_code[row] += trans[row][col] * ucon_ks[col];
    }
  }
}

void ProblemGenerator(parthenon::MeshBlock *pmb,
                      parthenon::ParameterInput *pin) {
  using namespace parthenon;

  const auto package_core = pmb->packages.Get("core");
  auto &resource = pmb->meshblock_data.Get();
  const Real kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");
  const auto enable_heating = package_core->Param<bool>("enable_heating");
  const Real kFelInit = enable_heating
      ? package_core->Param<Real>("fel_0") : Real{0.0};

  const Real kerr_a = pin->GetOrAddReal("metric", "a", 0.0);

  const Real rin = pin->GetOrAddReal("fm_torus", "rin", 6.0);
  const Real rmax = pin->GetOrAddReal("fm_torus", "rmax", 12.0);
  const Real kappa = pin->GetOrAddReal("fm_torus", "kappa", 1.0e-3);
  const Real perturbation =
      pin->GetOrAddReal("fm_torus", "perturbation", 2.0e-2);
  const std::string magnetic_field =
      pin->GetOrAddString("fm_torus", "magnetic_field", "sane");

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

  const auto& prim_tags = package_core->Param<std::vector<std::string>>("primitive_field_names");
  PackIndexMap primitiveIndexMap;
  auto primitive = resource->PackVariables(prim_tags, primitiveIndexMap);
  const auto& map = primitiveIndexMap;
  const int iRho = map["density"].first;
  const int iEny = map["energy"].first;
  const int iUx  = map["weighted_velocity"].first;
  const bool hasB = map.Has("magnetic_field");
  const int iBx  = hasB ? map["magnetic_field"].first : -1;
  const int iEnt = map["entropy"].first;
  const bool hasK = map.Has("electron_entropy");
  const int iKel = hasK ? map["electron_entropy"].first : -1;

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
        const Real x = x_code[1];
        const Real y = x_code[2];
        const Real z = x_code[3];
        
        const Real rad2 = x*x + y*y + z*z;
        const Real r = Kokkos::fmax(Kokkos::sqrt((rad2 - a2 +
                           Kokkos::sqrt(SQR(rad2 - a2) + 4.0*a2*z*z)) / 2.0), 1.0);
        const Real th = (Kokkos::fabs(z/r) < 1.0) ? Kokkos::acos(z/r) :
                         ((z > 0) ? 0.0 : M_PI);
        const Real sth = Kokkos::sin(th);
        const Real cth = Kokkos::cos(th);

        Real gcov[4][4];
        CKS::CalculateCodeMetric(x_code, gcov, kerr_a);

        Real gcov_bl[4][4];
        CalculateBoyerLindquistMetric(r, th, kerr_a, gcov_bl);

        Real gcon[4][4];
        invert(gcov, gcon);

        const Real alpha = 1.0 / Kokkos::sqrt(-gcon[0][0]);
        const Real beta1 = gcon[0][1] * alpha * alpha;
        const Real beta2 = gcon[0][2] * alpha * alpha;
        const Real beta3 = gcon[0][3] * alpha * alpha;

        // Non-torus cells start at zero; absolute floor is applied after
        // density normalisation in MeshPostInitialization.
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

        primitive(iRho, k, j, i) = rho;
        primitive(iEny, k, j, i) = eint;
        primitive(iUx, k, j, i) = wvx1;
        primitive(iUx + 1, k, j, i) = wvx2;
        primitive(iUx + 2, k, j, i) = wvx3;
        if (hasB) {
          primitive(iBx, k, j, i) = 0.0;
          primitive(iBx + 1, k, j, i) = 0.0;
          primitive(iBx + 2, k, j, i) = 0.0;
        }
        primitive(iEnt, k, j, i) = ent;
        if (hasK) {
          primitive(iKel, k, j, i) = kFelInit * ent;
        }
      });

  Real rhomax = 0.0;
  Real umax = 0.0;

  pmb->par_reduce(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_rhomax) {
        local_rhomax =
            Kokkos::max(local_rhomax, primitive(iRho, k, j, i));
      },
      Kokkos::Max<Real>(rhomax));
  
  pmb->par_reduce(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_umax) {
        local_umax = Kokkos::max(local_umax, primitive(iEny, k, j, i));
      },
      Kokkos::Max<Real>(umax));
}

void MeshPostInitialization(parthenon::Mesh *pmesh,
                            parthenon::ParameterInput *pin,
                            parthenon::MeshData<Real> *md) {
  using namespace parthenon;

  const auto package_core = pmesh->packages.Get("core");
  const Real kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");
  const auto enable_B = package_core->Param<bool>("enable_B");
  const auto enable_heating = package_core->Param<bool>("enable_heating");

  Real local_rhomax = 0.0;
  Real local_umax = 0.0;

  for (const auto &pmb : pmesh->block_list) {
    auto &resource = pmb->meshblock_data.Get();
    const auto& prim_tags = package_core->Param<std::vector<std::string>>("primitive_field_names");
    PackIndexMap primitiveIndexMap;
    auto primitive = resource->PackVariables(prim_tags, primitiveIndexMap);
    const auto& map = primitiveIndexMap;
    const int iRho = map["density"].first;
    const int iEny = map["energy"].first;
    const int iEnt = map["entropy"].first;

    auto cellbounds = pmb->cellbounds;
    const auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
    const auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
    const auto kb = cellbounds.GetBoundsK(IndexDomain::entire);

    Real block_rhomax = 0.0;
    Real block_umax = 0.0;

    pmb->par_reduce(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_max) {
          local_max = Kokkos::max(local_max, primitive(iRho, k, j, i));
        },
        Kokkos::Max<Real>(block_rhomax));

    pmb->par_reduce(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_max) {
          local_max = Kokkos::max(local_max, primitive(iEny, k, j, i));
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
    const auto& prim_tags = package_core->Param<std::vector<std::string>>("primitive_field_names");
    PackIndexMap primitiveIndexMap;
    auto primitive = resource->PackVariables(prim_tags, primitiveIndexMap);
    const auto& map = primitiveIndexMap;
    const int iRho = map["density"].first;
    const int iEny = map["energy"].first;
    const int iEnt = map["entropy"].first;

    auto cellbounds = pmb->cellbounds;
    const auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
    const auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
    const auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
    auto coords = pmb->coords;

    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          primitive(iRho, k, j, i) /= global_rhomax;
          primitive(iEny, k, j, i) /= global_rhomax;
        });

    // Re-apply absolute floor after normalization.
    const auto package_metric = pmesh->packages.Get("metric");
    const Real r_excise =
        package_metric->Param<Real>("r_excise");
    const Real dexcise =
        package_metric->Param<Real>("dexcise");
    const Real pexcise =
        package_metric->Param<Real>("pexcise");
    const Real e_excise = pexcise / (kAdiabaticIndex - 1.0);
    const Real density_floor =
        package_core->Param<Real>("density_floor");
    const Real density_floor_pow =
        package_core->Param<Real>("density_floor_pow");
    const Real energy_floor =
        package_core->Param<Real>("energy_floor");
    const Real energy_floor_pow =
        package_core->Param<Real>("energy_floor_pow");
    const Real kerr_a_floor = package_metric->Param<Real>("a");

    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real xc = coords.Xc<X1DIR>(i);
          const Real yc = coords.Xc<X2DIR>(j);
          const Real zc = coords.Xc<X3DIR>(k);
          // Use un-clamped BL radius for excision check
          // (GetBLRadius clamps to >=1.0 which defeats r < r_excise)
          const Real rad2 = xc*xc + yc*yc + zc*zc;
          const Real a_sq = kerr_a_floor * kerr_a_floor;
          const Real disc_val = (rad2-a_sq)*(rad2-a_sq) + 4.0*a_sq*zc*zc;
          const Real r_bl =
              Kokkos::sqrt(0.5*(rad2 - a_sq + Kokkos::sqrt(disc_val)));

          Real rho_floor, eng_floor;
          if (r_bl < r_excise) {
            rho_floor = dexcise;
            eng_floor = e_excise;
          } else {
            rho_floor = density_floor *
                Kokkos::pow(Kokkos::fmax(r_bl, 1.0), density_floor_pow);
            eng_floor = energy_floor *
                Kokkos::pow(Kokkos::fmax(r_bl, 1.0), energy_floor_pow);
          }

          if (primitive(iRho, k, j, i) < rho_floor)
            primitive(iRho, k, j, i) = rho_floor;
          if (primitive(iEny, k, j, i) < eng_floor)
            primitive(iEny, k, j, i) = eng_floor;
        });
  }

  // --- Magnetic field initialization ---
  if (enable_B) {
    const Real beta_target =
        pin->GetOrAddReal("fm_torus", "beta", 100.0);
    const Real aphi_rho_cut =
        pin->GetOrAddReal("fm_torus", "aphi_rho_cut", 0.2);
    const std::string magnetic_field_mag =
        pin->GetOrAddString("fm_torus", "magnetic_field", "sane");
    const bool is_mad = (magnetic_field_mag == "mad");
    const Real rin_mag = pin->GetOrAddReal("fm_torus", "rin", 6.0);

    const auto package_metric = pmesh->packages.Get("metric");
    const Real kerr_a_mag = package_metric->Param<Real>("a");
    const Real a2_mag = kerr_a_mag * kerr_a_mag;
    const int mtype_int = MetricType::CKS;

    // First pass: compute vector potential and B for each meshblock
    for (const auto &pmb : pmesh->block_list) {
      auto &resource = pmb->meshblock_data.Get();
      const auto &prim_tags =
          package_core->Param<std::vector<std::string>>("primitive_field_names");
      PackIndexMap idxMap;
      auto primitive = resource->PackVariables(prim_tags, idxMap);
      const int iRHO = idxMap["density"].first;
      const int iBX = idxMap["magnetic_field"].first;
      const int iENY = idxMap["energy"].first;
      const int iENT = idxMap["entropy"].first;

      auto cellbounds = pmb->cellbounds;
      const auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
      const auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
      const auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
      const auto ib_interior = cellbounds.GetBoundsI(IndexDomain::interior);
      const auto jb_interior = cellbounds.GetBoundsJ(IndexDomain::interior);
      const auto kb_interior = cellbounds.GetBoundsK(IndexDomain::interior);
      auto coords = pmb->coords;

      const int ni = ib.e - ib.s + 1;
      const int nj = jb.e - jb.s + 1;
      const int nk = kb.e - kb.s + 1;
      Kokkos::View<Real ***> A1("A1", nk, nj, ni);
      Kokkos::View<Real ***> A2("A2", nk, nj, ni);

      // Compute vector potential components at cell centers
      pmb->par_for(
          PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real rho = primitive(iRHO, k, j, i);
            Real ascale = (rho > aphi_rho_cut) ? (rho - aphi_rho_cut) : 0.0;
            if (is_mad && ascale > 0.0) {
              const Real x = coords.Xc<X1DIR>(i);
              const Real y = coords.Xc<X2DIR>(j);
              const Real zc = coords.Xc<X3DIR>(k);
              const Real rad2 = x * x + y * y + zc * zc;
              const Real r = Kokkos::fmax(
                  Kokkos::sqrt((rad2 - a2_mag +
                                Kokkos::sqrt(SQR(rad2 - a2_mag) +
                                             4.0 * a2_mag * zc * zc)) /
                               2.0),
                  1.0);
              const Real cos_theta = Kokkos::fmin(Kokkos::fmax(zc / r, -1.0), 1.0);
              const Real sin_theta = Kokkos::sqrt(1.0 - cos_theta * cos_theta);
              const Real mad_scale =
                  Kokkos::pow((r / rin_mag) * sin_theta, 3.0) *
                  Kokkos::exp(-r / 400.0);
              ascale *= mad_scale;
            }
            const int kk = k - kb.s;
            const int jj = j - jb.s;
            const int ii = i - ib.s;
            A1(kk, jj, ii) = -coords.Xc<X2DIR>(j) * ascale;
            A2(kk, jj, ii) = coords.Xc<X1DIR>(i) * ascale;
          });

      // Compute B = curl(A) at cell centers (interior only)
      pmb->par_for(
          PARTHENON_AUTO_LABEL, kb_interior.s, kb_interior.e,
          jb_interior.s, jb_interior.e, ib_interior.s, ib_interior.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real x_code_c[4] = {0.0, coords.Xc<X1DIR>(i),
                                      coords.Xc<X2DIR>(j),
                                      coords.Xc<X3DIR>(k)};
            Real gcov_c[4][4], gcon_c[4][4], gdet_c;
            ComputeMetricAtLocation(mtype_int, x_code_c, kerr_a_mag, 0.0,
                                    gcov_c, gcon_c, gdet_c);
            const Real sqrt_abs_gdet = Kokkos::sqrt(Kokkos::fabs(gdet_c));
            const Real dx1 = coords.Dxc<X1DIR>(i);
            const Real dx2 = coords.Dxc<X2DIR>(j);
            const Real dx3 = coords.Dxc<X3DIR>(k);

            const int kk = k - kb.s;
            const int jj = j - jb.s;
            const int ii = i - ib.s;

            const Real dA2_dx =
                (A2(kk, jj, ii + 1) - A2(kk, jj, ii - 1)) / (2.0 * dx1);
            const Real dA1_dy =
                (A1(kk, jj + 1, ii) - A1(kk, jj - 1, ii)) / (2.0 * dx2);
            const Real dA2_dz =
                (A2(kk + 1, jj, ii) - A2(kk - 1, jj, ii)) / (2.0 * dx3);
            const Real dA1_dz =
                (A1(kk + 1, jj, ii) - A1(kk - 1, jj, ii)) / (2.0 * dx3);

            primitive(iBX, k, j, i) = -dA2_dz / sqrt_abs_gdet;
            primitive(iBX + 1, k, j, i) = dA1_dz / sqrt_abs_gdet;
            primitive(iBX + 2, k, j, i) =
                (dA2_dx - dA1_dy) / sqrt_abs_gdet;
          });
    }

    // Compute maximum bsq for beta normalization
    Real local_bsq_max = 0.0;
    for (const auto &pmb : pmesh->block_list) {
      auto &resource = pmb->meshblock_data.Get();
      const auto &prim_tags =
          package_core->Param<std::vector<std::string>>("primitive_field_names");
      PackIndexMap idxMap;
      auto primitive = resource->PackVariables(prim_tags, idxMap);
      const int iRHO = idxMap["density"].first;
      const int iENY = idxMap["energy"].first;
      const int iUX = idxMap["weighted_velocity"].first;
      const int iENT = idxMap["entropy"].first;
      const int iBX = idxMap["magnetic_field"].first;
      const int iKEL = enable_heating
          ? idxMap["electron_entropy"].first : -1;

      auto cellbounds = pmb->cellbounds;
      const auto ib_interior = cellbounds.GetBoundsI(IndexDomain::interior);
      const auto jb_interior = cellbounds.GetBoundsJ(IndexDomain::interior);
      const auto kb_interior = cellbounds.GetBoundsK(IndexDomain::interior);
      auto coords = pmb->coords;

      Real block_bsq_max = 0.0;
      pmb->par_reduce(
          PARTHENON_AUTO_LABEL, kb_interior.s, kb_interior.e,
          jb_interior.s, jb_interior.e, ib_interior.s, ib_interior.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i,
                        Real &local_max) {
            const Real x_code_bsq[4] = {0.0, coords.Xc<X1DIR>(i),
                                        coords.Xc<X2DIR>(j),
                                        coords.Xc<X3DIR>(k)};
            Real gcov[4][4], gcon[4][4], gdet_bsq;
            ComputeMetricAtLocation(mtype_int, x_code_bsq, kerr_a_mag, 0.0,
                                    gcov, gcon, gdet_bsq);

            Real primitive_c_array[NPRIM] = {0};
            primitive_c_array[RHO] = primitive(iRHO, k, j, i);
            primitive_c_array[ENY] = primitive(iENY, k, j, i);
            primitive_c_array[UX1] = primitive(iUX, k, j, i);
            primitive_c_array[UX2] = primitive(iUX + 1, k, j, i);
            primitive_c_array[UX3] = primitive(iUX + 2, k, j, i);
            primitive_c_array[BX1] = primitive(iBX, k, j, i);
            primitive_c_array[BX2] = primitive(iBX + 1, k, j, i);
            primitive_c_array[BX3] = primitive(iBX + 2, k, j, i);
            primitive_c_array[ENT] = primitive(iENT, k, j, i);
            if (enable_heating) {
              primitive_c_array[KEL] = primitive(iKEL, k, j, i);
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
                                      MPI_PARTHENON_REAL, MPI_MAX,
                                      MPI_COMM_WORLD));
#else
    const Real global_bsq_max = local_bsq_max;
#endif

    // Compute normalization factor
    Real local_umax_norm = 0.0;
    Real local_rhomax_norm = 0.0;
    for (const auto &pmb : pmesh->block_list) {
      auto &resource = pmb->meshblock_data.Get();
      const auto &prim_tags =
          package_core->Param<std::vector<std::string>>("primitive_field_names");
      PackIndexMap idxMap;
      auto primitive = resource->PackVariables(prim_tags, idxMap);
      const int iENY = idxMap["energy"].first;
      const int iRHO = idxMap["density"].first;

      auto cellbounds = pmb->cellbounds;
      const auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
      const auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
      const auto kb = cellbounds.GetBoundsK(IndexDomain::entire);

      Real block_umax = 0.0;
      Real block_rhomax = 0.0;
      pmb->par_reduce(
          PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i,
                        Real &lmax) {
            lmax = Kokkos::max(lmax, primitive(iENY, k, j, i));
          },
          Kokkos::Max<Real>(block_umax));
      pmb->par_reduce(
          PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i,
                        Real &lmax) {
            lmax = Kokkos::max(lmax, primitive(iRHO, k, j, i));
          },
          Kokkos::Max<Real>(block_rhomax));
      local_umax_norm = Kokkos::max(local_umax_norm, block_umax);
      local_rhomax_norm = Kokkos::max(local_rhomax_norm, block_rhomax);
    }

#ifdef MPI_PARALLEL
    Real global_umax_norm = local_umax_norm;
    Real global_rhomax_norm = local_rhomax_norm;
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &global_umax_norm, 1,
                                      MPI_PARTHENON_REAL, MPI_MAX,
                                      MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &global_rhomax_norm, 1,
                                      MPI_PARTHENON_REAL, MPI_MAX,
                                      MPI_COMM_WORLD));
#else
    const Real global_umax_norm = local_umax_norm;
    const Real global_rhomax_norm = local_rhomax_norm;
#endif

    const Real beta_min =
        2.0 * (kAdiabaticIndex - 1.0) *
        (global_umax_norm / global_rhomax_norm) /
        (global_bsq_max + 1.0e-30);
    const Real magnetic_norm = Kokkos::sqrt(beta_min / beta_target);

    // Normalize B field to target beta
    for (const auto &pmb : pmesh->block_list) {
      auto &resource = pmb->meshblock_data.Get();
      const auto &prim_tags =
          package_core->Param<std::vector<std::string>>("primitive_field_names");
      PackIndexMap idxMap;
      auto primitive = resource->PackVariables(prim_tags, idxMap);
      const int iBX = idxMap["magnetic_field"].first;

      auto cellbounds = pmb->cellbounds;
      const auto ib_interior = cellbounds.GetBoundsI(IndexDomain::interior);
      const auto jb_interior = cellbounds.GetBoundsJ(IndexDomain::interior);
      const auto kb_interior = cellbounds.GetBoundsK(IndexDomain::interior);

      pmb->par_for(
          PARTHENON_AUTO_LABEL, kb_interior.s, kb_interior.e,
          jb_interior.s, jb_interior.e, ib_interior.s, ib_interior.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            primitive(iBX, k, j, i) *= magnetic_norm;
            primitive(iBX + 1, k, j, i) *= magnetic_norm;
            primitive(iBX + 2, k, j, i) *= magnetic_norm;
          });
    }

    if (Globals::my_rank == 0) {
      printf("Maximum initial magnetic bsq: %e\n", global_bsq_max);
      printf("Target beta: %e, beta_min: %e, magnetic normalization: %e\n",
             beta_target, beta_min, magnetic_norm);
    }
  }
}
