#ifndef PANGU_PGEN_RADIATION_SHADOW_H_
#define PANGU_PGEN_RADIATION_SHADOW_H_

#include <parthenon/parthenon.hpp>

namespace pangu::pgen {

void RadiationShadow(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);

void RegisterRadiationShadowBoundaries(parthenon::ApplicationInput* app_input);

} // namespace pangu::pgen

#endif
