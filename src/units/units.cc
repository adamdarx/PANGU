#include "units/units.h"

#include <string>

#include "utils/error_checking.hpp"

namespace pangu::units {
using namespace parthenon::package::prelude;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput* pin) {
  auto package = std::make_shared<StateDescriptor>("units");
  Real length = pin->GetOrAddReal("units", "length_cgs", 1.0);
  Real mass = pin->GetOrAddReal("units", "mass_cgs", 1.0);
  Real time = pin->GetOrAddReal("units", "time_cgs", 1.0);
  const Real mu = pin->GetOrAddReal("units", "mu", 1.0);
  const bool general_relativity = pin->DoesParameterExist("geometry", "background") &&
                                  pin->GetString("geometry", "background") != "minkowski" &&
                                  pin->GetString("geometry", "background") != "sr";
  if (general_relativity) {
    const Real density = pin->GetOrAddReal("units", "density_cgs", 1.0);
    const Real black_hole_mass = pin->GetOrAddReal("units", "bhmass_msun", 1.0) * Units::msun_cgs;
    length = Units::gravitational_constant_cgs * black_hole_mass /
             (Units::speed_of_light_cgs * Units::speed_of_light_cgs);
    mass = density * length * length * length;
    time = length / Units::speed_of_light_cgs;
  }
  PARTHENON_REQUIRE(length > 0.0 && mass > 0.0 && time > 0.0 && mu > 0.0,
                    "units scales and mean molecular weight must be positive");
  package->AddParam<Real>("length_cgs", length);
  package->AddParam<Real>("mass_cgs", mass);
  package->AddParam<Real>("time_cgs", time);
  package->AddParam<Real>("mu", mu);
  package->AddParam<bool>("general_relativistic_units", general_relativity);
  return package;
}

Units FromPackage(const std::shared_ptr<StateDescriptor>& package) {
  return {package->Param<Real>("length_cgs"), package->Param<Real>("mass_cgs"),
          package->Param<Real>("time_cgs"), package->Param<Real>("mu")};
}

} // namespace pangu::units
