#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "eos/newtonian_eos.h"
#include "eos/ideal_gas.h"
#include "reconstruct/hydro_reconstruction.h"
#include "riemann/registry.h"

namespace {

using pangu::eos::NewtonianEOS;
using pangu::hydro::EosMode;
using pangu::hydro::Primitive;
using pangu::hydro::Reconstruction;
using pangu::riemann::Physics;
using pangu::riemann::Solver;
using parthenon::Real;

constexpr Reconstruction kDC = *pangu::reconstruct::Parse("dc");
constexpr Reconstruction kPLM = *pangu::reconstruct::Parse("plm");
constexpr Reconstruction kPPM = *pangu::reconstruct::Parse("ppm");
constexpr Reconstruction kPPMC = *pangu::reconstruct::Parse("ppmc");
constexpr Reconstruction kWENOZ = *pangu::reconstruct::Parse("wenoz");

int failures = 0;

struct TestReconstruction {
  KOKKOS_INLINE_FUNCTION static void Reconstruct(
      const Real, const Real, const Real qm, const Real q0, const Real, const Real,
      Real& left, Real& right) {
    left = 0.75 * qm + 0.25 * q0;
    right = 0.25 * qm + 0.75 * q0;
  }
};

void Check(const bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void CheckNear(const Real actual, const Real expected, const Real tolerance,
               const std::string& message) {
  Check(std::abs(actual - expected) <= tolerance,
        message + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}

template <Solver S>
void CheckEqualStateFlux(const Primitive& state, const NewtonianEOS& eos) {
  Real flux[pangu::hydro::kIdealComponents]{};
  pangu::riemann::Implementation<Physics::newtonian_hydro, S>::Solve(state, state, eos, 0, flux);
  const auto expected = eos.PrimitiveToConservedAndFlux(state, 0);
  const int components =
      eos.HasEnergy() ? pangu::hydro::kIdealComponents : pangu::hydro::kIsothermalComponents;
  for (int n = 0; n < components; ++n) {
    CheckNear(flux[n], expected.flux[n], 2.0e-13,
              "equal-state Riemann flux solver=" + std::string(pangu::riemann::Describe(S).name) +
                  " component=" + std::to_string(n));
  }
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  const pangu::eos::IdealGas thermodynamics{1.4};
  Check(pangu::reconstruct::Parse("ppm4") == kPPM,
        "reconstruction alias resolves through the registry");
  Check(pangu::reconstruct::Describe(kPPM).relativistic_ghost_zones == 4,
        "registry exposes relativistic stencil requirements");
  Check(!pangu::reconstruct::Describe(kWENOZ).passive_transport,
        "registry exposes optional transport capability");
  CheckNear(thermodynamics.InternalEnergyDensity(0.8), 2.0, 1.0e-14,
            "ideal-gas pressure-to-internal-energy closure");
  CheckNear(thermodynamics.PressureFromInternalEnergyDensity(2.0), 0.8, 1.0e-14,
            "ideal-gas internal-energy-to-pressure closure");
  CheckNear(thermodynamics.EnthalpyDensity(1.2, 0.8), 4.0, 1.0e-14,
            "ideal-gas relativistic enthalpy density");
  const NewtonianEOS ideal{EosMode::ideal, 1.4, 1.0, 1.0e-12, 1.0e-12};
  const NewtonianEOS isothermal{EosMode::isothermal, 1.4, 0.7, 1.0e-12, 1.0e-12};
  const Primitive state{1.2, {0.3, -0.2, 0.1}, 0.9};
  const auto converted = ideal.PrimitiveToConservedAndFlux(state, 0);
  CheckNear(converted.conserved[pangu::hydro::IDN], 1.2, 1.0e-14, "density conversion");
  CheckNear(converted.conserved[pangu::hydro::IM1], 0.36, 1.0e-14, "momentum conversion");
  CheckNear(converted.conserved[pangu::hydro::IEN], 0.9 / 0.4 + 0.5 * 1.2 * 0.14, 1.0e-14,
            "energy conversion");

  Real left = 0.0;
  Real right = 0.0;
  pangu::reconstruct::ReconstructFace<kDC>(0.0, 0.0, 2.0, 3.0, 0.0, 0.0, left, right);
  CheckNear(left, 2.0, 0.0, "DC left state");
  CheckNear(right, 3.0, 0.0, "DC right state");
  pangu::reconstruct::ReconstructFace<kPLM>(0.0, 1.0, 2.0, 3.0, 4.0, 0.0, left, right);
  CheckNear(left, 2.5, 1.0e-14, "PLM linear exactness left");
  CheckNear(right, 2.5, 1.0e-14, "PLM linear exactness right");
  pangu::reconstruct::ReconstructFace<kPPM>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, left, right);
  CheckNear(left, 1.0, 1.0e-14, "PPM constant preservation left");
  CheckNear(right, 1.0, 1.0e-14, "PPM constant preservation right");
  pangu::reconstruct::ReconstructFace<kPPMC>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, left, right);
  CheckNear(left, 1.0, 1.0e-14, "PPMC constant preservation left");
  CheckNear(right, 1.0, 1.0e-14, "PPMC constant preservation right");
  pangu::reconstruct::ReconstructFace<kWENOZ>(-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, left, right);
  CheckNear(left, 0.5, 1.0e-14, "WENO-Z linear exactness left");
  CheckNear(right, 0.5, 1.0e-14, "WENO-Z linear exactness right");
  pangu::reconstruct::ReconstructWith<TestReconstruction>(0.0, 0.0, 2.0, 4.0, 0.0, 0.0,
                                                           left, right);
  CheckNear(left, 2.5, 0.0, "external reconstruction contract left");
  CheckNear(right, 3.5, 0.0, "external reconstruction contract right");
  Kokkos::View<Real[2]> reconstructed("external_reconstruction");
  Kokkos::parallel_for(
      "external_reconstruction_contract", 1, KOKKOS_LAMBDA(const int) {
        Real device_left = 0.0;
        Real device_right = 0.0;
        pangu::reconstruct::ReconstructWith<TestReconstruction>(
            0.0, 0.0, 2.0, 4.0, 0.0, 0.0, device_left, device_right);
        reconstructed(0) = device_left;
        reconstructed(1) = device_right;
      });
  const auto reconstructed_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reconstructed);
  CheckNear(reconstructed_host(0), 2.5, 0.0, "GPU external reconstruction contract left");
  CheckNear(reconstructed_host(1), 3.5, 0.0, "GPU external reconstruction contract right");

  // Every solver registered for Newtonian Hydro is reached through the registry dispatch.
  const Primitive sod_left{1.0, {0.0, 0.0, 0.0}, 1.0};
  const Primitive sod_right{0.125, {0.0, 0.0, 0.0}, 0.1};
  int dispatched = 0;
  for (const auto& descriptor : pangu::riemann::descriptors) {
    if (descriptor.solver == Solver::none || !descriptor.newtonian_hydro)
      continue;
    pangu::riemann::Visit<Physics::newtonian_hydro>(descriptor.solver, [&]<Solver S>() {
      ++dispatched;
      CheckEqualStateFlux<S>(state, ideal);
      if (descriptor.isothermal)
        CheckEqualStateFlux<S>(state, isothermal);
      Real flux[pangu::hydro::kIdealComponents]{};
      pangu::riemann::Implementation<Physics::newtonian_hydro, S>::Solve(sod_left, sod_right,
                                                                         ideal, 0, flux);
      Check(std::all_of(std::begin(flux), std::end(flux),
                        [](const Real value) { return std::isfinite(value); }),
            "Sod flux is finite for " + std::string(descriptor.name));
      Check(flux[pangu::hydro::IDN] > 0.0,
            "Sod mass flux points right for " + std::string(descriptor.name));
    });
  }
  Check(dispatched == 4, "LLF, HLLE, HLLC, and Roe are registered for Newtonian Hydro");
  bool rejected = false;
  try {
    pangu::riemann::Visit<Physics::newtonian_hydro>(Solver::hlld, []<Solver>() {});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Check(rejected, "HLLD is rejected for Newtonian Hydro");

  if (failures == 0)
    std::cout << "PANGU Hydro kernel tests PASS\n";
  return failures == 0 ? 0 : 1;
}
