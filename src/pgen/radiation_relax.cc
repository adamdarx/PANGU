#include "pgen/radiation_relax.h"

#include <cmath>
#include <vector>

#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "radiation/transport_package.h"
#include "relativity/relativistic_hydro.h"

namespace pangu::pgen {
using namespace parthenon;

void RadiationRelaxation(MeshBlock* block, ParameterInput* pin) {
  const Real radiation_energy = pin->GetReal("problem", "erad");
  const Real temperature = pin->GetReal("problem", "temp");
  const Real velocity = pin->GetOrAddReal("problem", "v1", 0.0);
  PARTHENON_REQUIRE(fabs(velocity) < 1.0, "radiation relaxation requires |problem/v1| < 1");
  const Real lorentz = 1.0 / sqrt(1.0 - velocity * velocity);
  const Real ux = lorentz * velocity;

  const auto hydro_package = block->packages.Get("hydro");
  const auto eos = pangu::eos::ReadRelativistic(*hydro_package);
  const auto radiation_package = block->packages.Get("radiation");
  const auto directions = radiation_package->Param<Kokkos::View<Real**>>("angle_directions");
  const int nangles = radiation_package->Param<int>("nangles");
  auto data = block->meshblock_data.Get();
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto intensity = data->PackVariables(std::vector<std::string>{"radiation.intensity"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  block->par_for(
      "PANGU radiation thermal-relaxation initial condition", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const relativity::HydroPrimitiveState state{1.0, {ux, 0.0, 0.0}, temperature};
        geometry::MetricPoint metric{};
        metric.lower[0][0] = -1.0;
        metric.upper[0][0] = -1.0;
        for (int axis = 0; axis < 3; ++axis) {
          metric.lower[axis + 1][axis + 1] = 1.0;
          metric.upper[axis + 1][axis + 1] = 1.0;
        }
        const auto converted = relativity::ConvertGRHDP2C(state, eos, metric);
        conserved(hydro::IDN, k, j, i) = converted.density;
        conserved(hydro::IM1, k, j, i) = converted.momentum[0];
        conserved(hydro::IM2, k, j, i) = converted.momentum[1];
        conserved(hydro::IM3, k, j, i) = converted.momentum[2];
        conserved(hydro::IEN, k, j, i) = converted.energy;
        primitive(hydro::IDN, k, j, i) = 1.0;
        primitive(hydro::IV1, k, j, i) = ux;
        primitive(hydro::IV2, k, j, i) = 0.0;
        primitive(hydro::IV3, k, j, i) = 0.0;
        primitive(hydro::IPR, k, j, i) = temperature;

        const Real comoving_intensity = radiation_energy / (4.0 * M_PI);
        for (int angle = 0; angle < nangles; ++angle) {
          const Real n0_comoving = lorentz - ux * directions(angle, 1);
          // AthenaK stores n^0 n_0 I.  In Minkowski space this is negative,
          // and I/nu^3 invariance supplies the fourth Doppler-factor power.
          intensity(angle, k, j, i) =
              -comoving_intensity / (n0_comoving * n0_comoving * n0_comoving * n0_comoving);
        }
      });
}

} // namespace pangu::pgen
