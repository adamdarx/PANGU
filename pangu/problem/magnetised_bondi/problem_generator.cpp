#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bvals/boundary_conditions.hpp"
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
#include "physics/state_calculation.h"
#include "prolong_restrict/prolong_restrict.hpp"
#include "task_list/task_list.h"

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

namespace {

constexpr int kBondiFieldSqrtGamma = 0;
constexpr int kBondiFieldMonopole = 1;
constexpr int kBondiSigmaNormReference = 0;
constexpr int kBondiSigmaNormMax = 1;

struct BondiSetup {
  parthenon::Real mdot;
  parthenon::Real sonic_radius;
  parthenon::Real inner_atmosphere_radius;
  parthenon::Real atmosphere_factor;
  parthenon::Real adiabatic_index;
  parthenon::Real fel_init;
  parthenon::Real mks_h;
  parthenon::Real mks_a;
  parthenon::Real magnetic_norm;
  int magnetic_field;
  bool magnetized;
};

int ParseBondiField(const std::string &name) {
  if (name == "sqrt_gamma") {
    return kBondiFieldSqrtGamma;
  }
  if (name == "monopole") {
    return kBondiFieldMonopole;
  }
  throw std::runtime_error("Unknown bondi/field: " + name);
}

int ParseBondiSigmaNorm(const std::string &name) {
  if (name == "reference") {
    return kBondiSigmaNormReference;
  }
  if (name == "max") {
    return kBondiSigmaNormMax;
  }
  throw std::runtime_error("Unknown bondi/sigma_norm: " + name);
}

KOKKOS_INLINE_FUNCTION
void CalculateBondiPrimitiveAtCodePosition(const parthenon::Real x1,
                                           const parthenon::Real x2,
                                           const parthenon::Real x3,
                                           const BondiSetup &setup,
                                           parthenon::Real primitive[NPRIM],
                                           parthenon::Real gcov[4][4],
                                           parthenon::Real gcon[4][4]) {
  using parthenon::Real;

  const Real x_code[4] = {0.0, x1, x2, x3};
  Real y[4];
  MKS::CalculatePhysicalCoordinates(x_code, y, setup.mks_h, setup.mks_a);
  MKS::CalculateCodeMetric(x_code, gcov, setup.mks_h, setup.mks_a);
  invert(gcov, gcon);

  const Real r = y[1];
  const Real alpha = 1.0 / Kokkos::sqrt(-gcon[0][0]);
  const Real beta1 = gcon[0][1] * alpha * alpha;
  const Real beta2 = gcon[0][2] * alpha * alpha;
  const Real beta3 = gcon[0][3] * alpha * alpha;

  Real rho = setup.atmosphere_factor;
  Real eint = setup.atmosphere_factor * 1.0e-3;
  Real wvx1 = 0.0;
  Real wvx2 = 0.0;
  Real wvx3 = 0.0;

  if (r >= setup.inner_atmosphere_radius) {
    Real ur = 0.0;
    SolveBondiSolution(r, setup.sonic_radius, setup.mdot,
                       setup.adiabatic_index, rho, eint, ur);

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

  primitive[RHO] = Kokkos::max(rho, setup.atmosphere_factor);
  primitive[ENY] = Kokkos::max(eint, setup.atmosphere_factor * 1.0e-6);
  primitive[UX1] = wvx1;
  primitive[UX2] = wvx2;
  primitive[UX3] = wvx3;
  primitive[BX1] = 0.0;
  primitive[BX2] = 0.0;
  primitive[BX3] = 0.0;

  if (setup.magnetized) {
    const Real sqrt_abs_g = Kokkos::sqrt(Kokkos::abs(determinant(gcov)));
    Real shape = 1.0;
    if (setup.magnetic_field == kBondiFieldMonopole) {
      // BHAC staggered Bondi 使用 A_phi=1-cos(theta)，curl 后径向场含 sin(theta)。
      shape = Kokkos::sin(y[2]);
    }
    primitive[BX1] = setup.magnetic_norm * shape / sqrt_abs_g;
  }

  primitive[ENT] = (setup.adiabatic_index - 1.0) * primitive[ENY] *
                   Kokkos::pow(primitive[RHO], -setup.adiabatic_index);
  primitive[KEL] = setup.fel_init * primitive[ENT];
}

parthenon::Real CalculateMagneticNorm(const BondiSetup &setup,
                                      const parthenon::Real sigma_inner,
                                      const parthenon::Real reference_radius) {
  if (!setup.magnetized || sigma_inner <= 0.0) {
    return 0.0;
  }

  BondiSetup unit_setup = setup;
  unit_setup.magnetic_norm = 1.0;

  parthenon::Real primitive[NPRIM];
  parthenon::Real gcov[4][4];
  parthenon::Real gcon[4][4];
  CalculateBondiPrimitiveAtCodePosition(Kokkos::log(reference_radius), 0.0, 0.0,
                                        unit_setup, primitive, gcov, gcon);

  State state;
  CalculateState(primitive, gcov, gcon, state);
  const parthenon::Real bsq_unit = state.bsq;
  if (bsq_unit <= 0.0) {
    return 0.0;
  }
  return Kokkos::sqrt(sigma_inner * primitive[RHO] / bsq_unit);
}

parthenon::Real CalculateMagneticNormByMax(parthenon::ParameterInput *pin,
                                           const BondiSetup &setup,
                                           const parthenon::Real sigma_inner) {
  if (!setup.magnetized || sigma_inner <= 0.0) {
    return 0.0;
  }

  BondiSetup unit_setup = setup;
  unit_setup.magnetic_norm = 1.0;

  const int nx1 = pin->GetInteger("parthenon/mesh", "nx1");
  const int nx2 = pin->GetOrAddInteger("parthenon/mesh", "nx2", 1);
  const int nx3 = pin->GetOrAddInteger("parthenon/mesh", "nx3", 1);
  const parthenon::Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const parthenon::Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  const parthenon::Real x2min =
      pin->GetOrAddReal("parthenon/mesh", "x2min", 0.0);
  const parthenon::Real x2max =
      pin->GetOrAddReal("parthenon/mesh", "x2max", 0.0);
  const parthenon::Real x3min =
      pin->GetOrAddReal("parthenon/mesh", "x3min", 0.0);
  const parthenon::Real x3max =
      pin->GetOrAddReal("parthenon/mesh", "x3max", 0.0);

  const parthenon::Real dx1 = (x1max - x1min) / nx1;
  const parthenon::Real dx2 = nx2 > 0 ? (x2max - x2min) / nx2 : 0.0;
  const parthenon::Real dx3 = nx3 > 0 ? (x3max - x3min) / nx3 : 0.0;
  parthenon::Real sigma_unit_max = 0.0;

  for (int k = 0; k < nx3; ++k) {
    const parthenon::Real x3 = x3min + (k + 0.5) * dx3;
    for (int j = 0; j < nx2; ++j) {
      const parthenon::Real x2 = x2min + (j + 0.5) * dx2;
      for (int i = 0; i < nx1; ++i) {
        const parthenon::Real x1 = x1min + (i + 0.5) * dx1;

        parthenon::Real primitive[NPRIM];
        parthenon::Real gcov[4][4];
        parthenon::Real gcon[4][4];
        CalculateBondiPrimitiveAtCodePosition(x1, x2, x3, unit_setup,
                                              primitive, gcov, gcon);

        State state;
        CalculateState(primitive, gcov, gcon, state);
        const parthenon::Real rho =
            Kokkos::max(primitive[RHO], static_cast<parthenon::Real>(1.0e-300));
        sigma_unit_max = Kokkos::max(sigma_unit_max, state.bsq / rho);
      }
    }
  }

  if (sigma_unit_max <= 0.0) {
    return 0.0;
  }
  return Kokkos::sqrt(sigma_inner / sigma_unit_max);
}

void ApplyBondiRadialBoundary(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource,
    const bool inner, const BondiSetup setup) {
  using namespace parthenon;

  const auto pmb = resource->GetBlockPointer();
  const auto package_core_bc = pmb->packages.Get("core");
  const auto enable_B = package_core_bc->Param<bool>("enable_B");
  const auto enable_heating = package_core_bc->Param<bool>("enable_heating");
  const auto& fnames =
      package_core_bc->Param<std::vector<std::string>>("primitive_field_names");

  PackIndexMap idxMap;
  auto primitive = resource->PackVariables(fnames, idxMap);

  const int iRHO = idxMap["density"].first;
  const int iENY = idxMap["energy"].first;
  const int iUX  = idxMap["weighted_velocity"].first;
  const int iENT = idxMap["entropy"].first;
  const int iBX  = enable_B ? idxMap["magnetic_field"].first : -1;
  const int iKEL = enable_heating ? idxMap["electron_entropy"].first : -1;

  auto coords = pmb->coords;
  auto cellbounds = pmb->cellbounds;
  const auto ib =
      cellbounds.GetBoundsI(inner ? IndexDomain::inner_x1
                                  : IndexDomain::outer_x1);
  const auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x1 = coords.Xc<X1DIR>(i);
        const Real x2 = coords.Xc<X2DIR>(j);
        const Real x3 = coords.Xc<X3DIR>(k);

        Real primitive_array[NPRIM];
        Real gcov[4][4];
        Real gcon[4][4];
        CalculateBondiPrimitiveAtCodePosition(x1, x2, x3, setup,
                                              primitive_array, gcov, gcon);

        primitive(iRHO, k, j, i) = primitive_array[RHO];
        primitive(iENY, k, j, i) = primitive_array[ENY];
        primitive(iUX,   k, j, i) = primitive_array[UX1];
        primitive(iUX+1, k, j, i) = primitive_array[UX2];
        primitive(iUX+2, k, j, i) = primitive_array[UX3];
        if (enable_B) {
          primitive(iBX,   k, j, i) = primitive_array[BX1];
          primitive(iBX+1, k, j, i) = primitive_array[BX2];
          primitive(iBX+2, k, j, i) = primitive_array[BX3];
        }
        primitive(iENT, k, j, i) = primitive_array[ENT];
        if (enable_heating) {
          primitive(iKEL, k, j, i) = primitive_array[KEL];
        }
      });
}

BondiSetup ReadBondiSetup(parthenon::ParameterInput *pin,
                          const parthenon::Real adiabatic_index,
                          const parthenon::Real fel_init) {
  BondiSetup setup;
  setup.mdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
  setup.sonic_radius = pin->GetOrAddReal("bondi", "rs", 8.0);
  setup.inner_atmosphere_radius = pin->GetOrAddReal("bondi", "rin", 10.0);
  setup.atmosphere_factor =
      pin->GetOrAddReal("bondi", "atmosphere_factor", 1.0e-7);
  setup.adiabatic_index = adiabatic_index;
  setup.fel_init = fel_init;
  setup.mks_h = pin->GetOrAddReal("metric", "h", 0.0);
  setup.mks_a = pin->GetOrAddReal("metric", "a", 0.0);
  setup.magnetized = pin->GetOrAddBoolean("bondi", "magnetized", false);
  setup.magnetic_field =
      ParseBondiField(pin->GetOrAddString("bondi", "field", "sqrt_gamma"));
  setup.magnetic_norm = 0.0;

  const auto sigma_inner = pin->GetOrAddReal("bondi", "sigma_inner", 0.0);
  const auto sigma_norm =
      ParseBondiSigmaNorm(pin->GetOrAddString("bondi", "sigma_norm", "reference"));
  const auto sigma_reference_radius =
      pin->GetOrAddReal("bondi", "sigma_reference_radius", 1.9);
  if (sigma_norm == kBondiSigmaNormMax) {
    setup.magnetic_norm = CalculateMagneticNormByMax(pin, setup, sigma_inner);
  } else {
    setup.magnetic_norm =
        CalculateMagneticNorm(setup, sigma_inner, sigma_reference_radius);
  }
  return setup;
}

}  // namespace

void ProblemGenerator(parthenon::MeshBlock *pmb,
                      parthenon::ParameterInput *pin) {
  using namespace parthenon;

  const auto package_core = pmb->packages.Get("core");
  auto &resource = pmb->meshblock_data.Get();
  const auto kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");
  const auto kFelInit = package_core->Param<Real>("fel_0");
  const BondiSetup setup = ReadBondiSetup(pin, kAdiabaticIndex, kFelInit);

  if (setup.magnetized) {
    printf("Magnetized Bondi: sigma_inner=%e, magnetic_norm=%e, field=%d, "
           "sigma_norm=%s\n",
           pin->GetOrAddReal("bondi", "sigma_inner", 0.0),
           setup.magnetic_norm, setup.magnetic_field,
           pin->GetOrAddString("bondi", "sigma_norm", "reference").c_str());
  }

  PackIndexMap idxMap;
  const auto enable_B = package_core->Param<bool>("enable_B");
  const auto enable_heating = package_core->Param<bool>("enable_heating");
  const auto& fnames =
      package_core->Param<std::vector<std::string>>("primitive_field_names");
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
        Real gcov[4][4];
        MKS::CalculateCodeMetric(x_code, gcov, setup.mks_h, setup.mks_a);

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
          MKS::CalculateCodeMetric(x_code_loc[loc], gcov_loc, setup.mks_h,
                                   setup.mks_a);
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

            MKS::CalculateCodeMetric(x_plus, gp, setup.mks_h, setup.mks_a);
            MKS::CalculateCodeMetric(x_minus, gm, setup.mks_h, setup.mks_a);
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

        Real primitive_array[NPRIM];
        Real gcov_center[4][4];
        Real gcon_center[4][4];
        CalculateBondiPrimitiveAtCodePosition(x1, x2, x3, setup,
                                              primitive_array, gcov_center,
                                              gcon_center);

        primitive(iRHO, k, j, i) = primitive_array[RHO];
        primitive(iENY, k, j, i) = primitive_array[ENY];
        primitive(iUX,   k, j, i) = primitive_array[UX1];
        primitive(iUX+1, k, j, i) = primitive_array[UX2];
        primitive(iUX+2, k, j, i) = primitive_array[UX3];
        if (enable_B) {
          primitive(iBX,   k, j, i) = primitive_array[BX1];
          primitive(iBX+1, k, j, i) = primitive_array[BX2];
          primitive(iBX+2, k, j, i) = primitive_array[BX3];
        }
        primitive(iENT, k, j, i) = primitive_array[ENT];
        if (enable_heating) {
          primitive(iKEL, k, j, i) = primitive_array[KEL];
        }
      });
}

void MeshPostInitialization(parthenon::Mesh *pm, parthenon::ParameterInput *pin,
                            parthenon::MeshData<parthenon::Real> *md) {
  (void)pm;
  (void)pin;
  (void)md;
}

void ProblemEnrollBoundaryFunctions(
    std::shared_ptr<parthenon::StateDescriptor> package_core,
    parthenon::ParameterInput *pin) {
  if (!pin->GetOrAddBoolean("bondi", "fixed_radial_bc", false)) {
    return;
  }

  const auto adiabatic_index =
      pin->GetOrAddReal("core", "adiabatic_index", 5.0 / 3.0);
  const auto fel_init = pin->GetOrAddReal("electron", "fel_0", 0.1);
  const BondiSetup setup = ReadBondiSetup(pin, adiabatic_index, fel_init);

  using BF = parthenon::BoundaryFace;
  package_core->UserBoundaryFunctions[BF::inner_x1].push_back(
      [setup](std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &rc,
              bool coarse) {
        (void)coarse;
        ApplyBondiRadialBoundary(rc, true, setup);
      });
  package_core->UserBoundaryFunctions[BF::outer_x1].push_back(
      [setup](std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &rc,
              bool coarse) {
        (void)coarse;
        ApplyBondiRadialBoundary(rc, false, setup);
      });
}
