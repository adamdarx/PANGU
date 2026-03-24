#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Random.hpp>
#include "../../src/simulator/Simulator.hpp"
#include "../../src/initialize/mnemonic.hpp"
#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin) {
    using namespace parthenon;
    const auto PackageCORE = pmb->packages.Get("CORE");
    auto &resource = pmb->meshblock_data.Get();
    const auto AdiabaticIndex = PackageCORE->Param<Real>("AdiabaticIndex");
    PackIndexMap primitiveIndexMap;
    const std::vector<std::string> PrimitiveTags = {"Density", "Energy", "WeightedVelocity", "MagneticField"};
    auto primitive = resource->PackVariables(PrimitiveTags, primitiveIndexMap);

    auto cellbounds = pmb->cellbounds;
    const auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
    const auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
    const auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
    auto coords = pmb->coords;
    Kokkos::Random_XorShift64_Pool<> random_pool(10086);

    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            auto generator = random_pool.get_state();
            primitive(DensityIndex, k, j, i) = 1.0 * (Kokkos::abs(coords.Xc<X2DIR>(j)) > 0.25) + 2.0 * (Kokkos::abs(coords.Xc<X2DIR>(j)) <= 0.25);
            primitive(EnergyIndex, k, j, i) = 2.5 / (AdiabaticIndex - 1);
            primitive(WeightedVelocityX1, k, j, i) = -0.5 * (Kokkos::abs(coords.Xc<X2DIR>(j)) > 0.25) + 0.5 * (Kokkos::abs(coords.Xc<X2DIR>(j)) <= 0.25) + 1e-2 * generator.drand(-1., 1.);
            primitive(WeightedVelocityX2, k, j, i) = 5e-2 * generator.drand(-1., 1.);
            primitive(WeightedVelocityX3, k, j, i) = 0.;
            primitive(MagneticFieldX1, k, j, i) = 0.5 * Kokkos::sqrt(4 * M_PI);
            primitive(MagneticFieldX2, k, j, i) = 0.;
            primitive(MagneticFieldX3, k, j, i) = 0.;
            random_pool.free_state(generator);
        }
    );
}
