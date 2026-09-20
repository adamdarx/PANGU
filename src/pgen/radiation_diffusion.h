#ifndef PANGU_PGEN_RADIATION_DIFFUSION_H_
#define PANGU_PGEN_RADIATION_DIFFUSION_H_

#include <parthenon/parthenon.hpp>

namespace pangu::pgen {

void RadiationDiffusion(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);

} // namespace pangu::pgen

#endif
