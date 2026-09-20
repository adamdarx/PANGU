#ifndef PANGU_PGEN_NUMERICAL_RELATIVITY_H_
#define PANGU_PGEN_NUMERICAL_RELATIVITY_H_

#include <parthenon/parthenon.hpp>

namespace pangu::pgen {

void NumericalRelativityMinkowski(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);
void SyncGRHDProblem(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);
void SyncGRMHDProblem(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);
void SyncGRHDTOV(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);
void SyncGRMHDTOV(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);
void NumericalRelativitySchwarzschild(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);
void NumericalRelativitySchwarzschildKerrSchild(parthenon::MeshBlock* block,
                                                parthenon::ParameterInput* pin);
void NumericalRelativityBoostedPuncture(parthenon::MeshBlock* block,
                                         parthenon::ParameterInput* pin);
void SyncGRMHDBoostedPuncture(parthenon::MeshBlock* block,
                              parthenon::ParameterInput* pin);
void NumericalRelativityTwoPunctures(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);
void NumericalRelativityLinearWave(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);
void NumericalRelativityGaugeWave(parthenon::MeshBlock* block, parthenon::ParameterInput* pin);
void NumericalRelativityRobustStability(parthenon::MeshBlock* block,
                                        parthenon::ParameterInput* pin);
void RegisterNumericalRelativityReflectingBoundaries(parthenon::ApplicationInput* app_input);
void RegisterNumericalRelativityExtrapolationBoundaries(parthenon::ApplicationInput* app_input);
void RegisterNumericalRelativityOutflowBoundaries(parthenon::ApplicationInput* app_input);
void NumericalRelativityAfterLoop(parthenon::Mesh* mesh, parthenon::ParameterInput* pin,
                                  parthenon::SimTime& time);
void NumericalRelativityLinearWaveAfterLoop(parthenon::Mesh* mesh, parthenon::ParameterInput* pin,
                                            parthenon::SimTime& time);
void NumericalRelativityGaugeWaveAfterLoop(parthenon::Mesh* mesh, parthenon::ParameterInput* pin,
                                           parthenon::SimTime& time);
void NumericalRelativityRobustStabilityAfterLoop(parthenon::Mesh* mesh,
                                                 parthenon::ParameterInput* pin,
                                                 parthenon::SimTime& time);

} // namespace pangu::pgen

#endif
