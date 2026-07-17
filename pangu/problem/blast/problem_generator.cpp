#include <memory>
#include <string>
#include <vector>

#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "initialization/variable_mnemonics.h"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "metric/tensor_algebra.h"
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "task_list/task_list.h"
#include <parthenon/package.hpp>

//----------------------------------------------------------------------------------------
// ProblemGenerator — spherical blast wave in Minkowski spacetime.
// Matching AthenaK gr_blast (minkowski=true): H=0 → pure flat metric, no black hole.

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

  const Real kOuterRadius = pin->GetOrAddReal("blast", "outer_radius", 1.0);
  const Real kInnerRadius = pin->GetOrAddReal("blast", "inner_radius", 0.8);
  const Real kDrat = pin->GetOrAddReal("blast", "drat", 100.0);
  const Real kPrat = pin->GetOrAddReal("blast", "prat", 33333.333);
  const Real kDamb = pin->GetOrAddReal("blast", "damb", 1.0e-4);
  const Real kPamb = pin->GetOrAddReal("blast", "pamb", 3.0e-5);
  const Real kBamb = pin->GetOrAddReal("blast", "bamb", 0.1);

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

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Blast initial conditions — spherical with smooth ramp
        const Real x = coords.Xc<X1DIR>(i);
        const Real y = coords.Xc<X2DIR>(j);
        const Real z = coords.Xc<X3DIR>(k);
        const Real rad = Kokkos::sqrt(x * x + y * y + z * z);

        Real den = kDamb;
        Real pres = kPamb;
        if (rad < kOuterRadius) {
          if (rad < kInnerRadius) {
            den *= kDrat;
            pres *= kPrat;
          } else {
            const Real f = (rad - kInnerRadius) / (kOuterRadius - kInnerRadius);
            const Real log_den =
                (1.0 - f) * Kokkos::log(kDrat * kDamb) + f * Kokkos::log(kDamb);
            den = Kokkos::exp(log_den);
            const Real log_pres =
                (1.0 - f) * Kokkos::log(kPrat * kPamb) + f * Kokkos::log(kPamb);
            pres = Kokkos::exp(log_pres);
          }
        }

        primitive(iRHO, k, j, i) = den;
        primitive(iENY, k, j, i) = pres / (kAdiabaticIndex - 1.0);
        primitive(iUX, k, j, i) = 0.0;
        primitive(iUX+1, k, j, i) = 0.0;
        primitive(iUX+2, k, j, i) = 0.0;
        if (enable_B) {
          primitive(iBX, k, j, i) = 0.0;
          primitive(iBX+1, k, j, i) = kBamb;
          primitive(iBX+2, k, j, i) = 0.0;
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
