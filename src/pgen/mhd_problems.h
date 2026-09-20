#ifndef PANGU_PGEN_MHD_PROBLEMS_H_
#define PANGU_PGEN_MHD_PROBLEMS_H_

#include <parthenon/parthenon.hpp>

namespace pangu::pgen {

void MHDProblem(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);
void NormalizeSaneTorus(parthenon::Mesh* mesh, parthenon::ParameterInput* pin,
                        parthenon::MeshData<parthenon::Real>* data);
void NormalizeMagnetisedBondi(parthenon::Mesh* mesh, parthenon::ParameterInput* pin,
                              parthenon::MeshData<parthenon::Real>* data);
void MHDAfterLoop(parthenon::Mesh* mesh, parthenon::ParameterInput* pin, parthenon::SimTime& time);
void RegisterMonopoleBoundaries(parthenon::ApplicationInput* app_input);
void RegisterMagnetisedBondiBoundaries(parthenon::ApplicationInput* app_input);
void RegisterChakrabartiTorusBoundaries(parthenon::ApplicationInput* app_input);

} // namespace pangu::pgen

#endif
