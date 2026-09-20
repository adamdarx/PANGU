#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "eos/newtonian_mhd_eos.h"
#include "riemann/registry.h"

namespace {

using pangu::eos::NewtonianMHDEOS;
using pangu::mhd::EosMode;
using pangu::mhd::InterfaceFlux;
using pangu::mhd::Primitive;
using pangu::riemann::Physics;
using pangu::riemann::Solver;
using parthenon::Real;

int failures = 0;

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
void CheckEqualState(const Primitive& state, const NewtonianMHDEOS& eos, const int direction) {
  InterfaceFlux flux{};
  pangu::riemann::Implementation<Physics::newtonian_mhd, S>::Solve(state, state, eos, direction, flux);
  const auto expected = eos.PrimitiveToConservedAndFlux(state, direction);
  const int components =
      eos.HasEnergy() ? pangu::mhd::kIdealComponents : pangu::mhd::kIsothermalComponents;
  for (int n = 0; n < components; ++n) {
    CheckNear(flux.fluid[n], expected.flux[n], 3.0e-13,
              "equal-state fluid flux component=" + std::to_string(n));
  }
  const int tangent1 = (direction + 1) % 3;
  const int tangent2 = (direction + 2) % 3;
  const Real by_flux = state.magnetic[tangent1] * state.velocity[direction] -
                       state.magnetic[direction] * state.velocity[tangent1];
  const Real bz_flux = state.magnetic[tangent2] * state.velocity[direction] -
                       state.magnetic[direction] * state.velocity[tangent2];
  CheckNear(flux.electric_t1, bz_flux, 3.0e-13, "equal-state first EMF");
  CheckNear(flux.electric_t2, -by_flux, 3.0e-13, "equal-state second EMF");
}

template <Solver S>
void CheckBrioWu(const Primitive& left, const Primitive& right, const NewtonianMHDEOS& eos) {
  InterfaceFlux flux{};
  pangu::riemann::Implementation<Physics::newtonian_mhd, S>::Solve(left, right, eos, 0, flux);
  for (const Real value : flux.fluid)
    Check(std::isfinite(value), "Brio-Wu fluid flux is finite");
  Check(std::isfinite(flux.electric_t1) && std::isfinite(flux.electric_t2),
        "Brio-Wu EMF is finite");
  Check(flux.fluid[pangu::mhd::IDN] > 0.0, "Brio-Wu mass flux points right");
}

} // namespace

int main() {
  const NewtonianMHDEOS ideal{EosMode::ideal, 5.0 / 3.0, 1.0, 1.0e-12, 1.0e-12};
  const NewtonianMHDEOS isothermal{EosMode::isothermal, 5.0 / 3.0, 0.7, 1.0e-12, 1.0e-12};
  const Primitive state{1.2, {0.3, -0.2, 0.1}, 0.9, {0.7, -0.4, 0.2}};
  const Primitive left{1.0, {0.0, 0.0, 0.0}, 1.0, {0.75, 1.0, 0.0}};
  const Primitive right{0.125, {0.0, 0.0, 0.0}, 0.1, {0.75, -1.0, 0.0}};
  // Every solver registered for Newtonian MHD is reached through the registry dispatch.
  int dispatched = 0;
  for (const auto& descriptor : pangu::riemann::descriptors) {
    if (descriptor.solver == Solver::none || !descriptor.newtonian_mhd)
      continue;
    pangu::riemann::Visit<Physics::newtonian_mhd>(descriptor.solver, [&]<Solver S>() {
      ++dispatched;
      for (int direction = 0; direction < 3; ++direction) {
        CheckEqualState<S>(state, ideal, direction);
        if (descriptor.isothermal)
          CheckEqualState<S>(state, isothermal, direction);
      }
      CheckBrioWu<S>(left, right, ideal);
    });
  }
  Check(dispatched == 3, "LLF, HLLE, and HLLD are registered for Newtonian MHD");
  bool rejected = false;
  try {
    pangu::riemann::Visit<Physics::newtonian_mhd>(Solver::hllc, []<Solver>() {});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Check(rejected, "HLLC is rejected for Newtonian MHD");
  Check(ideal.FastSpeed(left, 0) > 0.0, "ideal fast speed is positive");
  Check(isothermal.FastSpeed(left, 0) > 0.0, "isothermal fast speed is positive");

  if (failures == 0)
    std::cout << "PANGU MHD kernel tests PASS\n";
  return failures == 0 ? 0 : 1;
}
