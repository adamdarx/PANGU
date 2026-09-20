#ifndef PANGU_PGEN_RADIATION_RELAX_H_
#define PANGU_PGEN_RADIATION_RELAX_H_

#include <parthenon/parthenon.hpp>

namespace pangu::pgen {

void RadiationRelaxation(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);

} // namespace pangu::pgen

#endif
