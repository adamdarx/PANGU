#include <cmath>
#include <string>
#include <vector>

#include "../../src/initialize/mnemonic.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/package.hpp"

void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin) {
    using namespace parthenon;
    (void)pin;

    const auto package = pmb->packages.Get("PANGU");
    auto &resource = pmb->meshblock_data.Get();
    const Real gamma = package->Param<Real>("AdiabaticIndex");

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

            rho = (Kokkos::abs(y) <= 0.25) ? 2.0 : 1.0;
            p = 2.5;
            u1 = (Kokkos::abs(y) <= 0.25) ? 0.5 : -0.5;
            u2 = 0.01 * Kokkos::sin(4.0 * M_PI * x);
            b1 = 0.5;
            primitive(DensityIndex, k, j, i) = rho;
            primitive(EnergyIndex, k, j, i) = p / (gamma - 1.0);
            primitive(WeightedVelocityX1, k, j, i) = u1;
            primitive(WeightedVelocityX2, k, j, i) = u2;
            primitive(WeightedVelocityX3, k, j, i) = u3;
            primitive(MagneticFieldX1, k, j, i) = b1;
            primitive(MagneticFieldX2, k, j, i) = b2;
            primitive(MagneticFieldX3, k, j, i) = b3;
        });
}
