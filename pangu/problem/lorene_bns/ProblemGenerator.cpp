#include <cmath>
#include <string>
#include <vector>

#include "../../src/initialize/mnemonic.hpp"
#include "../../src/metric/kerr_metric.hpp"
#include "../../src/metric/schwarzschild_metric.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/package.hpp"

void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin) {
    using namespace parthenon;

    const auto package = pmb->packages.Get("PANGU");
    auto &resource = pmb->meshblock_data.Get();
    const Real gamma = package->Param<Real>("AdiabaticIndex");
    const auto metric_name = package->Param<std::string>("MetricName");
    const Real metric_spin = package->Param<Real>("MetricSpin");
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

            const Real r1 = Kokkos::sqrt((x - 0.35) * (x - 0.35) + y * y + z * z);
            const Real r2 = Kokkos::sqrt((x + 0.35) * (x + 0.35) + y * y + z * z);
            rho = 0.8 * Kokkos::exp(-r1 * r1 / 0.06) + 0.8 * Kokkos::exp(-r2 * r2 / 0.06) + 1e-4;
            p = 0.15 * rho;
            u2 = 0.08 * x;

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
