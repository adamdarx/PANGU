#ifndef PANGU_PGEN_RADIATION_HOHLRAUM_H_
#define PANGU_PGEN_RADIATION_HOHLRAUM_H_

#include <parthenon/parthenon.hpp>

namespace pangu::pgen {

void RadiationHohlraum(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);

void RegisterRadiationHohlraumBoundaries(parthenon::ApplicationInput* app_input);

} // namespace pangu::pgen

#endif
