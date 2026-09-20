#ifndef PANGU_PGEN_RADIATION_LINEAR_WAVE_H_
#define PANGU_PGEN_RADIATION_LINEAR_WAVE_H_

#include <parthenon/parthenon.hpp>

namespace pangu::pgen {

void RadiationLinearWave(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);

} // namespace pangu::pgen

#endif
