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

  auto covariant_metric = resource->Get("covariant_metric").data;
  auto contravariant_metric = resource->Get("contravariant_metric").data;
  auto metric_determinant = resource->Get("metric_determinant").data;
  auto connection = resource->Get("connection").data;

  auto cellbounds = pmb->cellbounds;
  const auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
  auto coords = pmb->coords;

  const Real LorentzFactor = 1 / Kokkos::sqrt(1 - 0.99 * 0.99);

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

        primitive(iRHO, k, j, i) = 1.0;
        primitive(iENY, k, j, i) = 10.0 / (kAdiabaticIndex - 1);
        primitive(iUX, k, j, i) =
            -LorentzFactor * Kokkos::sin(coords.Xc<X2DIR>(j));
        primitive(iUX+1, k, j, i) =
            LorentzFactor * Kokkos::sin(coords.Xc<X1DIR>(i));
        primitive(iUX+2, k, j, i) = 0.;
        if (enable_B) {
          primitive(iBX, k, j, i) =
              -Kokkos::sin(coords.Xc<X2DIR>(j));
          primitive(iBX+1, k, j, i) =
              Kokkos::sin(2 * coords.Xc<X1DIR>(i));
          primitive(iBX+2, k, j, i) = 0.;
        }
        primitive(iENT, k, j, i) =
            (kAdiabaticIndex - 1.0) * primitive(iENY, k, j, i) *
            Kokkos::pow(primitive(iRHO, k, j, i), -kAdiabaticIndex);
        if (enable_heating) {
          primitive(iKEL, k, j, i) = kFelInit * primitive(iENT, k, j, i);
        }
      });
}

void MeshPostInitialization(parthenon::Mesh *pmesh,
                            parthenon::ParameterInput *pin,
                            parthenon::MeshData<Real> *md) {
  using namespace parthenon;
}
