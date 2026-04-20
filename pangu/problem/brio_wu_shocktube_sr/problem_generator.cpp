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
        Real gcov[4][4];
        Real gcon[4][4];
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

        primitive(DensityIndex, k, j, i) = 1.0 * (coords.Xc<X1DIR>(i) < 0) +
                                           0.125 * (coords.Xc<X1DIR>(i) >= 0);
        primitive(EnergyIndex, k, j, i) =
            1.0 / (kAdiabaticIndex - 1.0) * (coords.Xc<X1DIR>(i) < 0) +
            0.1 / (kAdiabaticIndex - 1.0) * (coords.Xc<X1DIR>(i) >= 0);
        primitive(WeightedVelocityX1, k, j, i) = 0.;
        primitive(WeightedVelocityX2, k, j, i) = 0.;
        primitive(WeightedVelocityX3, k, j, i) = 0.;
        primitive(MagneticFieldX1, k, j, i) = 0.5;
        primitive(MagneticFieldX2, k, j, i) =
            1. * (coords.Xc<X1DIR>(i) < 0) - 1. * (coords.Xc<X1DIR>(i) >= 0);
        primitive(MagneticFieldX3, k, j, i) = 0.;
      });
}