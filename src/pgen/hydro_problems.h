#ifndef PANGU_PGEN_HYDRO_PROBLEMS_H_
#define PANGU_PGEN_HYDRO_PROBLEMS_H_

#include <parthenon/parthenon.hpp>

namespace pangu::pgen {

void RegisterTorusBoundaries(parthenon::ApplicationInput* app_input);
void RegisterBondiBoundaries(parthenon::ApplicationInput* app_input);
void HydroProblem(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);
void HydroAfterLoop(parthenon::Mesh* mesh, parthenon::ParameterInput* pin,
                    parthenon::SimTime& time);

} // namespace pangu::pgen

#endif
