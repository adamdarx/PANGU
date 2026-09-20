#ifndef PANGU_APP_PACKAGE_REGISTRY_H_
#define PANGU_APP_PACKAGE_REGISTRY_H_

#include <memory>

#include <parthenon/parthenon.hpp>

namespace pangu::app {

parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput>& pin);

} // namespace pangu::app

#endif
