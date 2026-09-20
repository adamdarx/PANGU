#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "z4c/tov.h"

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  pangu::nr::TOVParameters parameters{};
  const auto profile = pangu::nr::BuildPolytropicTOV(parameters);
  const auto radius =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), profile.schwarzschild_radius);
  const auto isotropic =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), profile.isotropic_radius);
  const auto mass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), profile.mass);
  const auto pressure =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), profile.pressure);
  const auto lapse = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), profile.lapse);

  bool valid = profile.points > 3 && profile.surface_radius > 0.0 &&
               profile.surface_isotropic_radius > 0.0 && profile.total_mass > 0.0 &&
               profile.surface_radius > 2.0 * profile.total_mass;
  for (int index = 0; index < profile.points; ++index) {
    valid = valid && std::isfinite(radius(index)) && std::isfinite(isotropic(index)) &&
            std::isfinite(mass(index)) && std::isfinite(pressure(index)) &&
            std::isfinite(lapse(index)) && pressure(index) > 0.0 && lapse(index) > 0.0;
    if (index > 0) {
      valid = valid && radius(index) > radius(index - 1) &&
              isotropic(index) > isotropic(index - 1) && mass(index) >= mass(index - 1) &&
              pressure(index) <= pressure(index - 1) && lapse(index) >= lapse(index - 1);
    }
  }
  Kokkos::View<double[8]> samples("TOV profile samples");
  Kokkos::parallel_for(
      "SYNC-1 TOV profile samples", Kokkos::RangePolicy<>(0, 2),
      KOKKOS_LAMBDA(const int sample) {
        const auto point = profile.EvaluateIsotropic(
            sample == 0 ? 0.0 : profile.surface_isotropic_radius);
        const int offset = 4 * sample;
        samples(offset) = point.density;
        samples(offset + 1) = point.lapse;
        samples(offset + 2) = point.mass;
        samples(offset + 3) = point.inside ? 1.0 : 0.0;
      });
  const auto host_samples =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), samples);
  const double central_error =
      std::abs(host_samples(0) - parameters.central_density) / parameters.central_density;
  const double exterior_lapse =
      (1.0 - profile.total_mass / (2.0 * profile.surface_isotropic_radius)) /
      (1.0 + profile.total_mass / (2.0 * profile.surface_isotropic_radius));
  const double lapse_error = std::abs(host_samples(5) - exterior_lapse);
  // AthenaK whisky_tov.athinput reference (rhoc=1.28e-3, kappa=100,
  // Gamma=2, dr=1e-3), evaluated with the same RK4/star-edge convention.
  const double reference_error = std::max(
      {std::abs(profile.surface_radius - 9.5856924301473612),
       std::abs(profile.surface_isotropic_radius - 8.1252128284209668),
       std::abs(profile.total_mass - 1.4001597275277418)});

  // The Gamma=2, K=100 sequence reaches its maximum gravitational mass near
  // rho_c=3.18e-3.  Samples on either side of that turning point provide a
  // cheap, deterministic gate for the stable (dM/drho_c > 0) and unstable
  // (dM/drho_c < 0) branches before either profile is handed to the 3-D
  // synchronized evolution.
  auto branch_parameters = parameters;
  branch_parameters.central_density = 2.0e-3;
  const auto stable = pangu::nr::BuildPolytropicTOV(branch_parameters);
  branch_parameters.central_density = 3.18e-3;
  const auto turning = pangu::nr::BuildPolytropicTOV(branch_parameters);
  branch_parameters.central_density = 8.0e-3;
  const auto unstable = pangu::nr::BuildPolytropicTOV(branch_parameters);
  const bool stable_branch = stable.total_mass > profile.total_mass &&
                             turning.total_mass > stable.total_mass;
  const bool unstable_branch = unstable.total_mass < turning.total_mass;
  const double branch_reference_error = std::max(
      {std::abs(stable.total_mass - 1.5737692223597960),
       std::abs(turning.total_mass - 1.6372759860251351),
       std::abs(unstable.total_mass - 1.4472943338265134)});
  valid = valid && central_error < 2.0e-12 && lapse_error < 4.0e-15 &&
          reference_error < 2.0e-13 && host_samples(7) == 0.0 &&
          host_samples(6) == profile.total_mass && stable_branch && unstable_branch &&
          branch_reference_error < 3.0e-13;

  std::cout.precision(17);
  std::cout << "SYNC-1 TOV profile: points=" << profile.points
            << " R=" << profile.surface_radius << " R_iso="
            << profile.surface_isotropic_radius << " M=" << profile.total_mass
            << " central_error=" << central_error << " surface_lapse_error=" << lapse_error
            << " AthenaK_reference_error=" << reference_error
            << " stable_mass=" << stable.total_mass
            << " turning_mass=" << turning.total_mass
            << " unstable_mass=" << unstable.total_mass
            << " branch_reference_error=" << branch_reference_error
            << '\n';
  if (!valid) {
    std::cerr << "SYNC-1 TOV profile validation failed\n";
    return 1;
  }
  return 0;
}
