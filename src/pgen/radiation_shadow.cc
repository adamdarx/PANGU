#include "pgen/radiation_shadow.h"

#include <cmath>
#include <memory>
#include <vector>

#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "relativity/relativistic_hydro.h"

namespace pangu::pgen {
using namespace parthenon;

namespace {

void RadiationShadowInnerBoundary(std::shared_ptr<MeshBlockData<Real>>& data, const bool coarse) {
  BoundaryFunction::OutflowInnerX1(data, coarse);

  auto* block = data->GetBlockPointer();
  const auto& bounds = coarse ? block->c_cellbounds : block->cellbounds;
  const auto ib = bounds.GetBoundsI(IndexDomain::interior);
  const auto jb = bounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = bounds.GetBoundsK(IndexDomain::entire);
  const auto ei = bounds.GetBoundsI(IndexDomain::entire);
  auto intensity = data->PackVariables(std::vector<std::string>{"radiation.intensity"}, coarse);
  const int nangles = block->packages.Get("radiation")->Param<int>("nangles");
  block->par_for(
      "PANGU radiation shadow inflow", 0, nangles - 1, kb.s, kb.e, jb.s, jb.e, ei.s, ib.s - 1,
      KOKKOS_LAMBDA(const int angle, const int k, const int j, const int i) {
        intensity(angle, k, j, i) = (angle == 2 || angle == 5) ? -100.0 : 0.0;
      });
}

} // namespace

void RadiationShadow(MeshBlock* block, ParameterInput* pin) {
  const auto radiation = block->packages.Get("radiation");
  PARTHENON_REQUIRE(radiation->Param<int>("nlevel") == 2,
                    "radiation shadow requires radiation/nlevel=2");
  PARTHENON_REQUIRE(!pin->GetBoolean("radiation", "rotate_geo"),
                    "radiation shadow requires radiation/rotate_geo=false");
  const auto eos = pangu::eos::ReadRelativistic(*block->packages.Get("hydro"));
  auto data = block->meshblock_data.Get();
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto intensity = data->PackVariables(std::vector<std::string>{"radiation.intensity"});
  const int nangles = radiation->Param<int>("nangles");
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coordinates = block->coords;
  block->par_for(
      "PANGU radiation shadow initial condition", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coordinates.Xc<1>(i);
        const Real y = coordinates.Xc<2>(j);
        const Real exponent = 10.0 * (x * x / 0.01 + y * y / 0.0036 - 1.0);
        const Real density = 1.0 + 9.0 / (1.0 + exp(exponent));
        const relativity::HydroPrimitiveState state{density, {0.0, 0.0, 0.0}, 1.0};
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
        primitive(hydro::IDN, k, j, i) = density;
        primitive(hydro::IV1, k, j, i) = 0.0;
        primitive(hydro::IV2, k, j, i) = 0.0;
        primitive(hydro::IV3, k, j, i) = 0.0;
        primitive(hydro::IPR, k, j, i) = 1.0;
        for (int angle = 0; angle < nangles; ++angle)
          intensity(angle, k, j, i) = 0.0;
      });
}

void RegisterRadiationShadowBoundaries(ApplicationInput* app_input) {
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x1, "radiation_shadow",
                                       RadiationShadowInnerBoundary);
}

} // namespace pangu::pgen
