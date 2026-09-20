#ifndef PANGU_APP_PARAMETER_COMPAT_H_
#define PANGU_APP_PARAMETER_COMPAT_H_

#include <iosfwd>

#include <parthenon/parthenon.hpp>

namespace pangu::app {

void NormalizeParameters(parthenon::ParameterInput* pin);
void ValidateParameters(parthenon::ParameterInput* pin);
void PrintConfiguration(parthenon::ParameterInput* pin, std::ostream& os);

} // namespace pangu::app

#endif
