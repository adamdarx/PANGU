#include "app/package_registry.h"

#include <string>

#include "app/problem_registry.h"
#include "geometry/geometry.h"
#include "plugin/registry.h"
#include "srcterms/source_terms.h"
#include "units/units.h"

namespace pangu::app {

parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput>& pin) {
  parthenon::Packages_t packages;
  packages.Add(units::Initialize(pin.get()));
  packages.Add(geometry::Initialize(pin.get()));
  packages.Add(srcterms::Initialize(pin.get()));

  const auto problem = pin->GetString("parthenon/job", "problem_id");
  packages.Add(GetPackageInitializer(problem)(pin.get()));
  if (const auto coupled_initializer = GetCoupledPackageInitializer(problem);
      coupled_initializer != nullptr)
    packages.Add(coupled_initializer(pin.get()));
  plugin::AddEnabledPackages(pin.get(), packages);
  return packages;
}

} // namespace pangu::app
