#ifndef PANGU_RADIATION_RADIATION_PACKAGE_H_
#define PANGU_RADIATION_RADIATION_PACKAGE_H_

#include <memory>

#include <parthenon/parthenon.hpp>

namespace pangu::radiation {

std::shared_ptr<parthenon::StateDescriptor> InitializeCooling(parthenon::ParameterInput* pin,
                                                       const parthenon::Packages_t& packages);

} // namespace pangu::radiation

#endif
