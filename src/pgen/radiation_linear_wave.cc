#include "pgen/radiation_linear_wave.h"

#include <cmath>
#include <vector>

#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "radiation/transport_package.h"
#include "relativity/relativistic_hydro.h"

namespace pangu::pgen {
using namespace parthenon;

void RadiationLinearWave(MeshBlock* block, ParameterInput* pin) {
  PARTHENON_REQUIRE(pin->GetOrAddBoolean("problem", "along_x1", true) &&
                        !pin->GetOrAddBoolean("problem", "along_x2", false) &&
                        !pin->GetOrAddBoolean("problem", "along_x3", false),
                    "the first radiation linear-wave gate is aligned with x1");
  const Real density0 = pin->GetReal("problem", "rho");
  const Real pressure0 = pin->GetReal("problem", "pgas");
  const Real ux0 = pin->GetOrAddReal("problem", "ux", 0.0);
  const Real uy0 = pin->GetOrAddReal("problem", "uy", 0.0);
  const Real uz0 = pin->GetOrAddReal("problem", "uz", 0.0);
  const Real radiation_energy0 = pin->GetReal("problem", "erad");
  const Real radiation_flux10 = pin->GetOrAddReal("problem", "fxrad", 0.0);
  const Real radiation_flux20 = pin->GetOrAddReal("problem", "fyrad", 0.0);
  const Real radiation_flux30 = pin->GetOrAddReal("problem", "fzrad", 0.0);
  const Real amplitude = pin->GetReal("problem", "delta");
  const Real density_real = pin->GetReal("problem", "drho_real");
  const Real density_imag = pin->GetReal("problem", "drho_imag");
  const Real pressure_real = pin->GetReal("problem", "dpgas_real");
  const Real pressure_imag = pin->GetReal("problem", "dpgas_imag");
  const Real ux_real = pin->GetReal("problem", "dux_real");
  const Real ux_imag = pin->GetReal("problem", "dux_imag");
  const Real uy_real = pin->GetOrAddReal("problem", "duy_real", 0.0);
  const Real uy_imag = pin->GetOrAddReal("problem", "duy_imag", 0.0);
  const Real uz_real = pin->GetOrAddReal("problem", "duz_real", 0.0);
  const Real uz_imag = pin->GetOrAddReal("problem", "duz_imag", 0.0);
  const Real radiation_energy_real = pin->GetReal("problem", "derad_real");
  const Real radiation_energy_imag = pin->GetReal("problem", "derad_imag");
  const Real radiation_flux1_real = pin->GetReal("problem", "dfxrad_real");
  const Real radiation_flux1_imag = pin->GetReal("problem", "dfxrad_imag");
  const Real radiation_flux2_real = pin->GetOrAddReal("problem", "dfyrad_real", 0.0);
  const Real radiation_flux2_imag = pin->GetOrAddReal("problem", "dfyrad_imag", 0.0);
  const Real radiation_flux3_real = pin->GetOrAddReal("problem", "dfzrad_real", 0.0);
  const Real radiation_flux3_imag = pin->GetOrAddReal("problem", "dfzrad_imag", 0.0);

  const auto hydro_package = block->packages.Get("hydro");
  const auto eos = pangu::eos::ReadRelativistic(*hydro_package);
  const auto radiation_package = block->packages.Get("radiation");
  const auto directions = radiation_package->Param<Kokkos::View<Real**>>("angle_directions");
  const int nangles = radiation_package->Param<int>("nangles");
  auto data = block->meshblock_data.Get();
  auto conserved = data->PackVariables(std::vector<std::string>{"hydro.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"hydro.prim"});
  auto intensity = data->PackVariables(std::vector<std::string>{"radiation.intensity"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  const Real wavelength =
      block->pmy_mesh->mesh_size.xmax(X1DIR) - block->pmy_mesh->mesh_size.xmin(X1DIR);
  const Real wave_number = 2.0 * M_PI / wavelength;
  block->par_for(
      "PANGU radiation linear-wave initial condition", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real phase = wave_number * coordinates.Xc<1>(i);
        const Real sine = sin(phase);
        const Real cosine = cos(phase);
        const Real density = density0 + amplitude * (density_real * cosine - density_imag * sine);
        const Real pressure =
            pressure0 + amplitude * (pressure_real * cosine - pressure_imag * sine);
        const Real ux = ux0 + amplitude * (ux_real * cosine - ux_imag * sine);
        const Real uy = uy0 + amplitude * (uy_real * cosine - uy_imag * sine);
        const Real uz = uz0 + amplitude * (uz_real * cosine - uz_imag * sine);
        const relativity::HydroPrimitiveState state{density, {ux, uy, uz}, pressure};
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
        primitive(hydro::IV1, k, j, i) = ux;
        primitive(hydro::IV2, k, j, i) = uy;
        primitive(hydro::IV3, k, j, i) = uz;
        primitive(hydro::IPR, k, j, i) = pressure;

        const Real radiation_energy =
            radiation_energy0 +
            amplitude * (radiation_energy_real * cosine - radiation_energy_imag * sine);
        const Real radiation_flux[3]{radiation_flux10 + amplitude * (radiation_flux1_real * cosine -
                                                                     radiation_flux1_imag * sine),
                                     radiation_flux20 + amplitude * (radiation_flux2_real * cosine -
                                                                     radiation_flux2_imag * sine),
                                     radiation_flux30 + amplitude * (radiation_flux3_real * cosine -
                                                                     radiation_flux3_imag * sine)};
        const Real flux_magnitude =
            sqrt(radiation_flux[0] * radiation_flux[0] + radiation_flux[1] * radiation_flux[1] +
                 radiation_flux[2] * radiation_flux[2]);
        const Real reduced_flux = flux_magnitude / radiation_energy;
        const Real flux_direction[3]{radiation_flux[0] / flux_magnitude,
                                     radiation_flux[1] / flux_magnitude,
                                     radiation_flux[2] / flux_magnitude};
        const Real fluid_lorentz = sqrt(1.0 + ux * ux + uy * uy + uz * uz);
        for (int angle = 0; angle < nangles; ++angle) {
          const Real projection =
              ux * directions(angle, 1) + uy * directions(angle, 2) + uz * directions(angle, 3);
          const Real n0_fluid = fluid_lorentz - projection;
          const Real n1_fluid =
              -ux + ux / (fluid_lorentz + 1.0) * projection + directions(angle, 1);
          const Real n2_fluid =
              -uy + uy / (fluid_lorentz + 1.0) * projection + directions(angle, 2);
          const Real n3_fluid =
              -uz + uz / (fluid_lorentz + 1.0) * projection + directions(angle, 3);
          const Real angular_projection = flux_direction[0] * n1_fluid +
                                          flux_direction[1] * n2_fluid +
                                          flux_direction[2] * n3_fluid;
          Real comoving_intensity = 0.0;
          if (reduced_flux <= 1.0 / 3.0) {
            comoving_intensity =
                radiation_energy / (4.0 * M_PI) * (1.0 + 3.0 * reduced_flux * angular_projection);
          } else {
            comoving_intensity = radiation_energy / (9.0 * M_PI) *
                                 (angular_projection - 3.0 * reduced_flux + 2.0) /
                                 ((1.0 - reduced_flux) * (1.0 - reduced_flux));
          }
          intensity(angle, k, j, i) =
              -comoving_intensity / (n0_fluid * n0_fluid * n0_fluid * n0_fluid);
        }
      });
}

} // namespace pangu::pgen
