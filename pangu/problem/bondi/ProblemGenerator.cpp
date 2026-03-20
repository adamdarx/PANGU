#include <cmath>
#include <string>
#include <vector>

#include "../../src/initialize/mnemonic.hpp"
#include "../../src/metric/kerr_metric.hpp"
#include "../../src/metric/schwarzschild_metric.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/package.hpp"

namespace {

KOKKOS_INLINE_FUNCTION Real GetTFunc(const Real t, const Real r,
                                     const Real c1, const Real c2, const Real n) {
    const Real a = 1.0 + (1.0 + n) * t;
    const Real b = c1 / (r * r * Kokkos::pow(t, n) + 1e-20);
    return a * a * (1.0 - 2.0 / r + b * b) - c2;
}

KOKKOS_INLINE_FUNCTION Real GetT(const Real r, const Real c1, const Real c2,
                                 const Real n, const Real rs) {
    const Real rtol = 1e-12;
    const Real ftol = 1e-14;
    const Real tinf = (Kokkos::sqrt(c2) - 1.0) / (n + 1.0);
    const Real tnear = Kokkos::pow(c1 * Kokkos::sqrt(2.0 / (r * r * r + 1e-20)), 1.0 / n);

    Real tmin = (r < rs) ? tinf : Kokkos::max(tnear, tinf);
    Real tmax = (r < rs) ? tnear : 1.0;

    Real f0 = GetTFunc(tmin, r, c1, c2, n);
    Real f1 = GetTFunc(tmax, r, c1, c2, n);
    if (f0 * f1 > 0.0) {
        return tinf;
    }

    Real t0 = tmin;
    Real t1 = tmax;
    Real th = 0.5 * (t0 + t1);
    Real fh = GetTFunc(th, r, c1, c2, n);
    const Real eps_t = rtol * (tmin + tmax + 1e-20);

    for (int iter = 0; iter < 80; ++iter) {
        if (!(Kokkos::abs(th - t0) > eps_t && Kokkos::abs(th - t1) > eps_t && Kokkos::abs(fh) > ftol)) {
            break;
        }
        if (fh * f0 > 0.0) {
            t0 = th;
            f0 = fh;
        } else {
            t1 = th;
            f1 = fh;
        }
        th = 0.5 * (t0 + t1);
        fh = GetTFunc(th, r, c1, c2, n);
    }

    return th;
}

KOKKOS_INLINE_FUNCTION void GetBondiSolution(const Real r, const Real rs,
                                             const Real mdot, const Real gam,
                                             Real &rho, Real &u, Real &ur) {
    const Real n = 1.0 / (gam - 1.0);
    const Real uc = Kokkos::sqrt(1.0 / (2.0 * rs));
    const Real vc = Kokkos::sqrt(uc * uc / (1.0 - 3.0 * uc * uc));
    const Real tc = -n * vc * vc / ((n + 1.0) * (n * vc * vc - 1.0));
    const Real c1 = uc * rs * rs * Kokkos::pow(tc, n);
    const Real a = 1.0 + (1.0 + n) * tc;
    const Real c2 = a * a * (1.0 - 2.0 / rs + uc * uc);
    const Real kpoly = Kokkos::pow(4.0 * M_PI * c1 / (mdot + 1e-20), 1.0 / n);
    const Real kpolyn = Kokkos::pow(kpoly, n);

    const Real t = GetT(r, c1, c2, n, rs);
    const Real tn = Kokkos::pow(t, n);

    rho = tn / (kpolyn + 1e-20);
    u = rho * t * n;
    ur = -c1 / (tn * r * r + 1e-20);
}

}  // namespace

void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin) {
    using namespace parthenon;
    const auto package = pmb->packages.Get("PANGU");
    auto &resource = pmb->meshblock_data.Get();
    const Real gamma = package->Param<Real>("AdiabaticIndex");
    const auto metric_name = package->Param<std::string>("MetricName");
    const Real metric_spin = package->Param<Real>("MetricSpin");
    const Real rs = pin->GetOrAddReal("bondi", "rs", 8.0);
    const Real mdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
    const Real ur_frac = pin->GetOrAddReal("bondi", "ur_frac", 1.0);
    const Real uphi = pin->GetOrAddReal("bondi", "uphi", 0.0);
    const Real rho_floor = pin->GetOrAddReal("bondi", "rho_floor", 1e-8);
    const Real u_floor = pin->GetOrAddReal("bondi", "u_floor", 1e-10);
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

            Real ur = 0.0;
            GetBondiSolution(r, rs, mdot, gamma, rho, p, ur);
            rho = Kokkos::max(rho, rho_floor);
            p = Kokkos::max(p, u_floor);

            const Real ur_local = ur_frac * ur;
            u1 = ur_local * x / r - uphi * y / rcyl;
            u2 = ur_local * y / r + uphi * x / rcyl;
            u3 = ur_local * z / r;
            b1 = 0.0;
            b2 = 0.0;
            b3 = 0.0;

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
