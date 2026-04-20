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

void ProblemGenerator(parthenon::MeshBlock *pmb,
                      parthenon::ParameterInput *pin) {
  using namespace parthenon;
  const auto package_core = pmb->packages.Get("core");
  auto &resource = pmb->meshblock_data.Get();
  const auto kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");
  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field"};
  auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);

  auto cellbounds = pmb->cellbounds;
  const auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
  auto coords = pmb->coords;

  const Real LorentzFactor = 1 / Kokkos::sqrt(1 - 0.99 * 0.99);

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        primitive(DensityIndex, k, j, i) = 1.0;
        primitive(EnergyIndex, k, j, i) = 10.0 / (kAdiabaticIndex - 1);
        primitive(WeightedVelocityX1, k, j, i) =
            -LorentzFactor * Kokkos::sin(coords.Xc<X2DIR>(j));
        primitive(WeightedVelocityX2, k, j, i) =
            LorentzFactor * Kokkos::sin(coords.Xc<X1DIR>(i));
        primitive(WeightedVelocityX3, k, j, i) = 0.;
        primitive(MagneticFieldX1, k, j, i) =
            -Kokkos::sin(coords.Xc<X2DIR>(j));
        primitive(MagneticFieldX2, k, j, i) =
            Kokkos::sin(2 * coords.Xc<X1DIR>(i));
        primitive(MagneticFieldX3, k, j, i) = 0.;
      });
}
