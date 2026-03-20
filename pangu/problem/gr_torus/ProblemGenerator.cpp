#include <cmath>
#include <string>
#include <vector>

#include "../../src/initialize/mnemonic.hpp"
#include "../../src/metric/kerr_metric.hpp"
#include "../../src/metric/schwarzschild_metric.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/package.hpp"

namespace {

KOKKOS_INLINE_FUNCTION Real LFishCalc(const Real a, const Real r) {
    const Real sqtr = Kokkos::sqrt(Kokkos::max(r, 1e-12));
    const Real term1 = a * a - 2.0 * a * sqtr + r * r;
    const Real term2 = Kokkos::sqrt(Kokkos::max(2.0 * a * sqtr + (-3.0 + r) * r, 1e-12));
    const Real term3 = Kokkos::sqrt(Kokkos::max(1.0 + (2.0 * a) / Kokkos::pow(r, 1.5) - 3.0 / r, 1e-12));
    const Real numerator =
        term1 *
        ((-2.0 * a * r * term1) / term2 + ((a + (-2.0 + r) * sqtr) * (r * r * r + a * a * (2.0 + r))) / term3);
    const Real denominator =
        r * r * r * term2 * (a * a + (-2.0 + r) * r);
    return numerator / (denominator + 1e-20);
}

KOKKOS_INLINE_FUNCTION Real LnhCalc(const Real a, const Real l, const Real rin, const Real r,
                                    const Real th) {
    const Real sth = Kokkos::sin(th);
    const Real cth = Kokkos::cos(th);
    const Real thin = M_PI / 2.0;
    const Real sthin = Kokkos::sin(thin);
    const Real cthin = Kokkos::cos(thin);

    const Real r2 = r * r;
    const Real a2 = a * a;
    const Real dd = r2 - 2.0 * r + a2;
    const Real aa = Kokkos::pow(r2 + a2, 2) - dd * a2 * sth * sth;
    const Real ss = r2 + a2 * cth * cth;

    const Real rin2 = rin * rin;
    const Real ddin = rin2 - 2.0 * rin + a2;
    const Real aain = Kokkos::pow(rin2 + a2, 2) - ddin * a2 * sthin * sthin;
    const Real ssin = rin2 + a2 * cthin * cthin;

    if (r < rin) {
        return 1.0;
    }

    const Real q1 = Kokkos::sqrt(Kokkos::max(1.0 + 4.0 * (l * l * ss * ss) * dd / (aa * aa * sth * sth + 1e-20), 1e-20));
    const Real q2 = Kokkos::sqrt(Kokkos::max(1.0 + 4.0 * (l * l * ssin * ssin) * ddin /
                                                       (aain * aain * sthin * sthin + 1e-20),
                                            1e-20));

    const Real part_r = 0.5 * Kokkos::log((1.0 + q1) / (ss * dd / (aa + 1e-20) + 1e-20)) -
                        0.5 * q1 - 2.0 * a * r * l / (aa + 1e-20);
    const Real part_in = 0.5 * Kokkos::log((1.0 + q2) / (ssin * ddin / (aain + 1e-20) + 1e-20)) -
                         0.5 * q2 - 2.0 * a * rin * l / (aain + 1e-20);
    return part_r - part_in;
}

KOKKOS_INLINE_FUNCTION Real FmTorusRho(const Real a, const Real rin, const Real rmax,
                                       const Real gam, const Real kappa,
                                       const Real r, const Real th) {
    const Real l = LFishCalc(a, rmax);
    const Real lnh = LnhCalc(a, l, rin, r, th);
    if (lnh < 0.0 || r < rin) {
        return 0.0;
    }
    const Real hm1 = Kokkos::exp(lnh) - 1.0;
    return Kokkos::pow(hm1 * (gam - 1.0) / (kappa * gam + 1e-20), 1.0 / (gam - 1.0));
}

}  // namespace

void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin) {
    using namespace parthenon;
    const auto package = pmb->packages.Get("PANGU");
    auto &resource = pmb->meshblock_data.Get();
    const Real gamma = package->Param<Real>("AdiabaticIndex");
    const auto metric_name = package->Param<std::string>("MetricName");
    const Real metric_spin = package->Param<Real>("MetricSpin");
    const Real rin = pin->GetOrAddReal("gr_torus", "rin", 6.0);
    const Real rmax = pin->GetOrAddReal("gr_torus", "rmax", 12.0);
    const Real kappa = pin->GetOrAddReal("gr_torus", "kappa", 1e-3);
    const Real rho_floor = pin->GetOrAddReal("gr_torus", "rho_floor", 1e-6);
    const Real p_floor = pin->GetOrAddReal("gr_torus", "p_floor", 1e-8);
    const Real b0 = pin->GetOrAddReal("gr_torus", "b0", 0.0);
    const int metric_type =
        (metric_name == "Minkowski") ? 0 : ((metric_name == "Schwarzschild") ? 1 : 2);

    PackIndexMap primitive_index_map;
    const std::vector<std::string> primitive_tags = {
        "Density", "Energy", "WeightedVelocity", "MagneticField"};
    auto primitive = resource->PackVariables(primitive_tags, primitive_index_map);

    const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    const auto coords = pmb->coords;

    pmb->par_for(
        PARTHENON_AUTO_LABEL,
        kb.s,
        kb.e,
        jb.s,
        jb.e,
        ib.s,
        ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real x = coords.Xc<X1DIR>(i);
            const Real y = coords.Xc<X2DIR>(j);
            const Real z = coords.Xc<X3DIR>(k);

            Real rho = 1.0;
            Real p = 1.0;
            Real u1 = 0.0;
            Real u2 = 0.0;
            Real u3 = 0.0;
            Real b1 = 0.0;
            Real b2 = 0.0;
            Real b3 = 0.0;

            const Real rcyl = Kokkos::sqrt(x * x + y * y) + 1e-12;
            const Real r = Kokkos::sqrt(x * x + y * y + z * z) + 1e-12;
            const Real th = Kokkos::acos(Kokkos::min(1.0, Kokkos::max(-1.0, z / r)));

            const Real rho_torus = FmTorusRho(metric_spin, rin, rmax, gamma, kappa, r, th);
            if (rho_torus > 0.0) {
                rho = rho_torus;
                p = kappa * Kokkos::pow(Kokkos::max(rho, 1e-20), gamma);

                const Real vphi = Kokkos::min(0.5, Kokkos::sqrt(1.0 / r));
                u1 = -vphi * y / rcyl;
                u2 = vphi * x / rcyl;
                u3 = 0.0;
                b3 = b0;
            } else {
                rho = rho_floor;
                p = p_floor;
                u1 = 0.0;
                u2 = 0.0;
                u3 = 0.0;
                b1 = 0.0;
                b2 = 0.0;
                b3 = 0.0;
            }

            // Convert coordinate-basis seed velocity into the normal-observer 3-velocity.
            Real gcov[4][4];
            Real gcon[4][4];
            if (metric_type == 0) {
                for (int mu = 0; mu < 4; ++mu) {
                    for (int nu = 0; nu < 4; ++nu) {
                        gcov[mu][nu] = 0.0;
                        gcon[mu][nu] = 0.0;
                    }
                }
                gcov[0][0] = -1.0;
                gcov[1][1] = 1.0;
                gcov[2][2] = 1.0;
                gcov[3][3] = 1.0;
                gcon[0][0] = -1.0;
                gcon[1][1] = 1.0;
                gcon[2][2] = 1.0;
                gcon[3][3] = 1.0;
            } else if (metric_type == 1) {
                ComputeSchwarzschildMetric(x, y, z, gcov, gcon);
            } else {
                ComputeKerrMetric(x, y, z, metric_spin, gcov, gcon);
            }

            const Real g0u = gcov[0][1] * u1 + gcov[0][2] * u2 + gcov[0][3] * u3;
            const Real guu =
                gcov[1][1] * u1 * u1 + 2.0 * gcov[1][2] * u1 * u2 +
                2.0 * gcov[1][3] * u1 * u3 + gcov[2][2] * u2 * u2 +
                2.0 * gcov[2][3] * u2 * u3 + gcov[3][3] * u3 * u3;
            const Real qa = gcov[0][0];
            const Real qb = 2.0 * g0u;
            const Real qc = guu + 1.0;
            const Real disc = Kokkos::max(0.0, qb * qb - 4.0 * qa * qc);
            const Real sqrt_disc = Kokkos::sqrt(disc);
            const Real u0a = (-qb + sqrt_disc) / (2.0 * qa);
            const Real u0b = (-qb - sqrt_disc) / (2.0 * qa);
            const Real u0 = (u0a > 0.0) ? u0a : u0b;

            const Real weighted_u1 = u1 - (gcon[0][1] / gcon[0][0]) * u0;
            const Real weighted_u2 = u2 - (gcon[0][2] / gcon[0][0]) * u0;
            const Real weighted_u3 = u3 - (gcon[0][3] / gcon[0][0]) * u0;

            primitive(DensityIndex, k, j, i) = rho;
            primitive(EnergyIndex, k, j, i) = p / (gamma - 1.0);
            primitive(WeightedVelocityX1, k, j, i) = weighted_u1;
            primitive(WeightedVelocityX2, k, j, i) = weighted_u2;
            primitive(WeightedVelocityX3, k, j, i) = weighted_u3;
            primitive(MagneticFieldX1, k, j, i) = b1;
            primitive(MagneticFieldX2, k, j, i) = b2;
            primitive(MagneticFieldX3, k, j, i) = b3;
        });
}
