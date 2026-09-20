#include "pgen/radiation_diffusion.h"

#include <cmath>
#include <vector>

#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "radiation/transport_package.h"
#include "relativity/relativistic_hydro.h"

namespace pangu::pgen {
using namespace parthenon;

void RadiationDiffusion(MeshBlock* block, ParameterInput* pin) {
  const Real velocity = pin->GetReal("problem", "v1");
  const Real nu = pin->GetReal("problem", "nu");
  PARTHENON_REQUIRE(fabs(velocity) < 1.0, "radiation diffusion requires |problem/v1| < 1");
  const Real lorentz = 1.0 / sqrt(1.0 - velocity * velocity);
  const Real ux = lorentz * velocity;
  const Real scattering = block->packages.Get("radiation")->Param<Real>("kappa_s");
  PARTHENON_REQUIRE(scattering > 0.0, "radiation diffusion requires radiation/kappa_s > 0");
  const Real diffusivity = 1.0 / (3.0 * scattering);
  const Real nu_squared = nu * nu;
  const Real proper_time_offset = 6.0 * ux;
  const auto eos = pangu::eos::ReadRelativistic(*block->packages.Get("hydro"));
  const auto directions =
      block->packages.Get("radiation")->Param<Kokkos::View<Real**>>("angle_directions");
  const int nangles = block->packages.Get("radiation")->Param<int>("nangles");
  auto data = block->meshblock_data.Get();
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto intensity = data->PackVariables(std::vector<std::string>{"radiation.intensity"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  block->par_for(
      "PANGU radiation diffusion initial condition", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const relativity::HydroPrimitiveState state{1.0, {ux, 0.0, 0.0}, 1.0};
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
        primitive(hydro::IPR, k, j, i) = 1.0;

        const Real x = coordinates.Xc<1>(i);
        const Real proper_time = -lorentz * velocity * x;
        const Real proper_x = lorentz * x;
        const Real denominator =
            1.0 + 4.0 * diffusivity * nu_squared * (proper_time_offset + proper_time);
        const Real energy =
            fmax(exp(-nu_squared * proper_x * proper_x / denominator) / sqrt(denominator), 1.0e-20);
        const Real numerator =
            proper_x - velocity * diffusivity -
            2.0 * nu_squared * diffusivity *
                (2.0 * velocity * diffusivity * (proper_time_offset - velocity * proper_x) +
                 proper_x * (velocity * proper_x - 2.0 * proper_time_offset));
        const Real flux =
            2.0 * diffusivity * nu_squared * numerator / (denominator * denominator) * energy;
        const Real reduced_flux = fabs(flux) / energy;
        const Real flux_direction = flux >= 0.0 ? 1.0 : -1.0;

        for (int angle = 0; angle < nangles; ++angle) {
          const Real projection = ux * directions(angle, 1);
          const Real n0_comoving = lorentz - projection;
          const Real n1_comoving = -ux + ux / (lorentz + 1.0) * projection + directions(angle, 1);
          const Real angular_projection = flux_direction * n1_comoving;
          Real comoving_intensity = 0.0;
          if (reduced_flux <= 1.0 / 3.0) {
            comoving_intensity =
                energy / (4.0 * M_PI) * (1.0 + 3.0 * reduced_flux * angular_projection);
          } else {
            comoving_intensity = energy / (9.0 * M_PI) *
                                 (angular_projection - 3.0 * reduced_flux + 2.0) /
                                 ((1.0 - reduced_flux) * (1.0 - reduced_flux));
          }
          intensity(angle, k, j, i) =
              -comoving_intensity / (n0_comoving * n0_comoving * n0_comoving * n0_comoving);
        }
      });
}

} // namespace pangu::pgen
