#ifndef PANGU_PGEN_RADIATION_BEAM_H_
#define PANGU_PGEN_RADIATION_BEAM_H_

#include <parthenon/parthenon.hpp>

namespace pangu::pgen {

void RadiationBeam(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);

} // namespace pangu::pgen

#endif
