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
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "task_list/task_list.h"
#include <parthenon/package.hpp>

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

  PackIndexMap idxMap;
  auto primitive = resource->PackVariables(fnames, idxMap);

  const int iRHO = idxMap["density"].first;
  const int iENY = idxMap["energy"].first;
  const int iUX  = idxMap["weighted_velocity"].first;
  const int iENT = idxMap["entropy"].first;
  const int iBX  = enable_B ? idxMap["magnetic_field"].first : -1;
  const int iKEL = enable_heating ? idxMap["electron_entropy"].first : -1;

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
        primitive(iRHO, k, j, i) =
            1.0 * (Kokkos::abs(coords.Xc<X2DIR>(j)) > 0.25) +
            2.0 * (Kokkos::abs(coords.Xc<X2DIR>(j)) <= 0.25);
        primitive(iENY, k, j, i) = 2.5 / (kAdiabaticIndex - 1);
        primitive(iUX, k, j, i) =
            -0.5 * (Kokkos::abs(coords.Xc<X2DIR>(j)) > 0.25) +
            0.5 * (Kokkos::abs(coords.Xc<X2DIR>(j)) <= 0.25) +
            1e-2 * generator.drand(-1., 1.);
        primitive(iUX+1, k, j, i) =
            5e-2 * generator.drand(-1., 1.);
        primitive(iUX+2, k, j, i) = 0.;
        if (enable_B) {
          primitive(iBX, k, j, i) = 0.5 * Kokkos::sqrt(4 * M_PI);
          primitive(iBX+1, k, j, i) = 0.;
          primitive(iBX+2, k, j, i) = 0.;
        }
        primitive(iENT, k, j, i) =
            (kAdiabaticIndex - 1.0) * primitive(iENY, k, j, i) *
            Kokkos::pow(primitive(iRHO, k, j, i), -kAdiabaticIndex);
        if (enable_heating) {
          primitive(iKEL, k, j, i) = kFelInit * primitive(iENT, k, j, i);
        }
        random_pool.free_state(generator);
      });
}

void MeshPostInitialization(parthenon::Mesh *pmesh,
                            parthenon::ParameterInput *pin,
                            parthenon::MeshData<Real> *md) {
  using namespace parthenon;
}
