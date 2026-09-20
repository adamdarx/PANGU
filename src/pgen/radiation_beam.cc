#include "pgen/radiation_beam.h"

#include <vector>

#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "relativity/relativistic_hydro.h"

namespace pangu::pgen {
using namespace parthenon;

void RadiationBeam(MeshBlock* block, ParameterInput*) {
  const auto eos = pangu::eos::ReadRelativistic(*block->packages.Get("hydro"));
  auto data = block->meshblock_data.Get();
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto intensity = data->PackVariables(std::vector<std::string>{"radiation.intensity"});
  const int nangles = block->packages.Get("radiation")->Param<int>("nangles");
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  block->par_for(
      "PANGU flat radiation beam initial condition", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const relativity::HydroPrimitiveState state{1.0, {0.0, 0.0, 0.0}, 1.0};
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
        primitive(hydro::IV1, k, j, i) = 0.0;
        primitive(hydro::IV2, k, j, i) = 0.0;
        primitive(hydro::IV3, k, j, i) = 0.0;
        primitive(hydro::IPR, k, j, i) = 1.0;
        for (int angle = 0; angle < nangles; ++angle)
          intensity(angle, k, j, i) = 0.0;
      });
}

} // namespace pangu::pgen
