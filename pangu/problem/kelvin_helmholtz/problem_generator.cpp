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

void ProblemGenerator(parthenon::MeshBlock *pmb,
                      parthenon::ParameterInput *pin) {
  using namespace parthenon;
  const auto package_core = pmb->packages.Get("core");
  auto &resource = pmb->meshblock_data.Get();
  const auto kAdiabaticIndex = package_core->Param<Real>("adiabatic_index");
  const auto kFelInit = package_core->Param<Real>("fel_0");
  PackIndexMap primitiveIndexMap;
  const std::vector<std::string> primitive_tags = {
      "density", "energy", "weighted_velocity", "magnetic_field", "entropy",
      "electron_entropy"};
  auto primitive = resource->PackVariables(primitive_tags, primitiveIndexMap);

  auto covariant_metric = resource->Get("covariant_metric").data;
  auto contravariant_metric = resource->Get("contravariant_metric").data;
  auto metric_determinant = resource->Get("metric_determinant").data;
  auto connection = resource->Get("connection").data;

  auto cellbounds = pmb->cellbounds;
  const auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
  auto coords = pmb->coords;
  Kokkos::Random_XorShift64_Pool<> random_pool(10086);

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real gcov[4][4];
        Real gcon[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            gcov[row][col] = 0.0;
            gcon[row][col] = 0.0;
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

        for (int loc = 0; loc < 4; ++loc) {
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
              covariant_metric(loc, col, row, k, j, i) = gcov[row][col];
              contravariant_metric(loc, col, row, k, j, i) = gcon[row][col];
            }
          }
          metric_determinant(loc, k, j, i) = -1.0;
        }
        for (int dir = 0; dir < 4; ++dir) {
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
              connection(dir, col, row, k, j, i) = 0.0;
            }
          }
        }

        auto generator = random_pool.get_state();
        primitive(RHO, k, j, i) =
            1.0 * (Kokkos::abs(coords.Xc<X2DIR>(j)) > 0.25) +
            2.0 * (Kokkos::abs(coords.Xc<X2DIR>(j)) <= 0.25);
        primitive(ENY, k, j, i) = 2.5 / (kAdiabaticIndex - 1);
        primitive(UX1, k, j, i) =
            -0.5 * (Kokkos::abs(coords.Xc<X2DIR>(j)) > 0.25) +
            0.5 * (Kokkos::abs(coords.Xc<X2DIR>(j)) <= 0.25) +
            1e-2 * generator.drand(-1., 1.);
        primitive(UX2, k, j, i) =
            5e-2 * generator.drand(-1., 1.);
        primitive(UX3, k, j, i) = 0.;
        primitive(BX1, k, j, i) = 0.5 * Kokkos::sqrt(4 * M_PI);
        primitive(BX2, k, j, i) = 0.;
        primitive(BX3, k, j, i) = 0.;
        primitive(ENT, k, j, i) =
            (kAdiabaticIndex - 1.0) * primitive(ENY, k, j, i) *
            Kokkos::pow(primitive(RHO, k, j, i), -kAdiabaticIndex);
        primitive(KEL, k, j, i) = kFelInit * primitive(ENT, k, j, i);
        random_pool.free_state(generator);
      });
}
