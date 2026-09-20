#include "pgen/radiation_hohlraum.h"

#include <cmath>
#include <memory>
#include <vector>

#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "relativity/relativistic_hydro.h"

namespace pangu::pgen {
using namespace parthenon;

namespace {

template <int Direction>
void RadiationHohlraumInnerBoundary(std::shared_ptr<MeshBlockData<Real>>& data, const bool coarse) {
  static_assert(Direction == 0 || Direction == 1);
  if constexpr (Direction == 0)
    BoundaryFunction::OutflowInnerX1(data, coarse);
  else
    BoundaryFunction::OutflowInnerX2(data, coarse);

  auto* block = data->GetBlockPointer();
  const auto& bounds = coarse ? block->c_cellbounds : block->cellbounds;
  const auto ib = bounds.GetBoundsI(IndexDomain::interior);
  const auto jb = bounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = bounds.GetBoundsK(IndexDomain::interior);
  const auto ei = bounds.GetBoundsI(IndexDomain::entire);
  const auto ej = bounds.GetBoundsJ(IndexDomain::entire);
  const auto ek = bounds.GetBoundsK(IndexDomain::entire);
  auto intensity = data->PackVariables(std::vector<std::string>{"radiation.intensity"}, coarse);
  const int nangles = block->packages.Get("radiation")->Param<int>("nangles");
  const Real inflow = -1.0 / (4.0 * M_PI);

  int il = ei.s;
  int iu = ei.e;
  int jl = ej.s;
  int ju = ej.e;
  if constexpr (Direction == 0) {
    iu = ib.s - 1;
  } else {
    ju = jb.s - 1;
  }
  block->par_for(
      "PANGU radiation hohlraum inflow", 0, nangles - 1, ek.s, ek.e, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int angle, const int k, const int j, const int i) {
        intensity(angle, k, j, i) = inflow;
      });
}

} // namespace

void RadiationHohlraum(MeshBlock* block, ParameterInput*) {
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
      "PANGU radiation hohlraum initial condition", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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

void RegisterRadiationHohlraumBoundaries(ApplicationInput* app_input) {
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x1, "radiation_hohlraum",
                                       RadiationHohlraumInnerBoundary<0>);
  app_input->RegisterBoundaryCondition(BoundaryFace::inner_x2, "radiation_hohlraum",
                                       RadiationHohlraumInnerBoundary<1>);
}

} // namespace pangu::pgen
