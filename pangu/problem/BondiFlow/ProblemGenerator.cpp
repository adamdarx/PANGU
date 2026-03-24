#include <memory>
#include <string>
#include <vector>

#include "../../src/simulator/Simulator.hpp"
#include "../../src/initialize/mnemonic.hpp"
#include "../../src/metric/Kerr.hpp"
#include "../../src/metric/linear_algebra.hpp"
#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
parthenon::Real BondiTemperatureResidual(const parthenon::Real T,
                                         const parthenon::Real r,
                                         const parthenon::Real C1,
                                         const parthenon::Real C2,
                                         const parthenon::Real n) {
    const parthenon::Real T_safe = Kokkos::max(T, static_cast<parthenon::Real>(1.0e-14));
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
    const parthenon::Real Tnear = Kokkos::pow(C1 * Kokkos::sqrt(2.0 / (r * r * r)), 1.0 / n);

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

        if (Kokkos::abs(fh) <= ftol ||
            Kokkos::abs(Th - T0) <= epsT ||
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
void SolveBondiSolution(const parthenon::Real r,
                        const parthenon::Real rs,
                        const parthenon::Real mdot,
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
    const parthenon::Real K = Kokkos::pow(4.0 * M_PI * C1 / mdot, 1.0 / n);
    const parthenon::Real Kn = Kokkos::pow(K, n);

    const parthenon::Real T = SolveBondiTemperature(r, C1, C2, n, rs);
    const parthenon::Real Tn = Kokkos::pow(T, n);

    rho = Tn / Kn;
    u = rho * T * n;
    ur = -C1 / (Tn * r * r);
}

} // namespace

void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin) {
    using namespace parthenon;

    const auto PackageCORE = pmb->packages.Get("CORE");
    auto &resource = pmb->meshblock_data.Get();
    const auto AdiabaticIndex = PackageCORE->Param<Real>("AdiabaticIndex");

    // Bondi parameters follow the Sisyphus defaults when not explicitly provided.
    const Real BondiMdot = pin->GetOrAddReal("BONDI", "mdot", 1.0);
    const Real BondiSonicRadius = pin->GetOrAddReal("BONDI", "rs", 8.0);
    const Real BondiInnerAtmosphereRadius = pin->GetOrAddReal("BONDI", "rin", 10.0);
    const Real BondiAtmosphereFactor = pin->GetOrAddReal("BONDI", "atmosphere_factor", 1.0e-7);

    // Kerr metric parameters used by device-side metric functions.
    const Real kerr_h = pin->GetOrAddReal("METRIC", "h", 0.0);
    const Real kerr_a = pin->GetOrAddReal("METRIC", "a", 0.0);

    PackIndexMap primitiveIndexMap;
    const std::vector<std::string> PrimitiveTags = {"Density", "Energy", "WeightedVelocity", "MagneticField"};
    auto primitive = resource->PackVariables(PrimitiveTags, primitiveIndexMap);

    PackIndexMap covariantMetricIndexMap;
    const std::vector<std::string> CovariantMetricTags = {"CovariantMetric"};
    auto CovariantMetric = resource->PackVariables(CovariantMetricTags, covariantMetricIndexMap);

    PackIndexMap contravariantMetricIndexMap;
    const std::vector<std::string> ContravariantMetricTags = {"ContravariantMetric"};
    auto ContravariantMetric = resource->PackVariables(ContravariantMetricTags, contravariantMetricIndexMap);

    PackIndexMap metricDeterminantIndexMap;
    const std::vector<std::string> MetricDeterminantTags = {"MetricDeterminant"};
    auto MetricDeterminant = resource->PackVariables(MetricDeterminantTags, metricDeterminantIndexMap);

    PackIndexMap covariantMetricDerivativeIndexMap;
    const std::vector<std::string> CovariantMetricDerivativeTags = {"CovariantMetricDerivative"};
    auto CovariantMetricDerivative =
        resource->PackVariables(CovariantMetricDerivativeTags, covariantMetricDerivativeIndexMap);

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
            Kerr::CalculatePhysicalCoordinates(x_code, y, kerr_h, kerr_a);
            Real gcov[4][4];
            Kerr::CalculateMetricTensor(x_code, gcov, kerr_h, kerr_a);

            Real gcon[4][4];
            invert(gcov, gcon);
            const Real gdet = determinant(gcov);

            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    CovariantMetric(row * 4 + col, k, j, i) = gcov[row][col];
                    ContravariantMetric(row * 4 + col, k, j, i) = gcon[row][col];
                }
            }
            MetricDeterminant(0, k, j, i) = gdet;

            const Real dx1 = coords.Dxc<X1DIR>(i);
            const Real dx2 = coords.Dxc<X2DIR>(j);
            const Real dx3 = coords.Dxc<X3DIR>(k);
            const Real dx[4] = {0.0, dx1, dx2, dx3};

            for (int dir = 0; dir < 4; ++dir) {
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        Real deriv = 0.0;
                        if (dir > 0 && dx[dir] > 0.0) {
                            Real gp[4][4], gm[4][4];
                            Real xp1 = x1, xp2 = x2, xm1 = x1, xm2 = x2;
                            if (dir == 1) {
                                xp1 += 0.5 * dx[1];
                                xm1 -= 0.5 * dx[1];
                            } else if (dir == 2) {
                                xp2 += 0.5 * dx[2];
                                xm2 -= 0.5 * dx[2];
                            }

                            const Real x_plus[4] = {0.0, xp1, xp2, x3};
                            const Real x_minus[4] = {0.0, xm1, xm2, x3};
                            Kerr::CalculateMetricTensor(x_plus, gp, kerr_h, kerr_a);
                            Kerr::CalculateMetricTensor(x_minus, gm, kerr_h, kerr_a);
                            deriv = (gp[row][col] - gm[row][col]) / dx[dir];
                        }
                        CovariantMetricDerivative((dir * 4 + row) * 4 + col, k, j, i) = deriv;
                    }
                }
            }

            const Real r = y[1];
            const Real alpha = 1.0 / Kokkos::sqrt(-gcon[0][0]);
            const Real beta1 = gcon[0][1] * alpha * alpha;
            const Real beta2 = gcon[0][2] * alpha * alpha;
            const Real beta3 = gcon[0][3] * alpha * alpha;

            Real rho = BondiAtmosphereFactor;
            Real eint = BondiAtmosphereFactor * 1.0e-3;
            Real wvx1 = 0.0;
            Real wvx2 = 0.0;
            Real wvx3 = 0.0;

            if (r >= BondiInnerAtmosphereRadius) {
                Real ur = 0.0;
                SolveBondiSolution(r, BondiSonicRadius, BondiMdot, AdiabaticIndex, rho, eint, ur);

                // Bondi solution provides u^r in physical radius; convert to internal x1 velocity.
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

            primitive(DensityIndex, k, j, i) = Kokkos::max(rho, BondiAtmosphereFactor);
            primitive(EnergyIndex, k, j, i) = Kokkos::max(eint, BondiAtmosphereFactor * 1.0e-6);
            primitive(WeightedVelocityX1, k, j, i) = wvx1;
            primitive(WeightedVelocityX2, k, j, i) = wvx2;
            primitive(WeightedVelocityX3, k, j, i) = wvx3;
            primitive(MagneticFieldX1, k, j, i) = 0.0;
            primitive(MagneticFieldX2, k, j, i) = 0.0;
            primitive(MagneticFieldX3, k, j, i) = 0.0;
        });
}