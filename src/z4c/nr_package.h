#ifndef PANGU_Z4C_NR_PACKAGE_H_
#define PANGU_Z4C_NR_PACKAGE_H_

#include <memory>

#include <parthenon/parthenon.hpp>

namespace pangu::nr {

std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput* pin);

parthenon::TaskStatus Z4cToADMBlockTask(parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::TaskStatus Z4cToADMMeshTask(parthenon::MeshData<parthenon::Real>* data);
// Convert only the evolved Z4c fields into the current ADM fields.  These
// variants deliberately omit constraints and Weyl diagnostics so coupled
// initial data can establish the geometry before constructing matter.
parthenon::TaskStatus Z4cToADMFieldsBlockTask(parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::TaskStatus Z4cToADMFieldsMeshTask(parthenon::MeshData<parthenon::Real>* data);
// RK stages need ADM and constraints for geometry/timestep access, but do not
// need the substantially more expensive Weyl tensor until a diagnostic or
// field output is due.
parthenon::TaskStatus Z4cToADMStageBlockTask(parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::TaskStatus Z4cToADMStageMeshTask(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus ComputeConstraintsBlockTask(parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::TaskStatus ComputeConstraintsMeshTask(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus ComputeWeylBlockTask(parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::TaskStatus ComputeWeylMeshTask(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus BuildStressEnergyBlockTask(
    parthenon::MeshBlockData<parthenon::Real>* data, parthenon::Real time);
parthenon::TaskStatus BuildStressEnergyMeshTask(parthenon::MeshData<parthenon::Real>* data,
                                                parthenon::Real time);

parthenon::TaskStatus StageUpdateMeshTask(parthenon::MeshData<parthenon::Real>* current,
                                          parthenon::MeshData<parthenon::Real>* base,
                                          parthenon::Real gamma_current, parthenon::Real gamma_base,
                                          parthenon::Real beta_dt, parthenon::Real time,
                                          parthenon::MeshData<parthenon::Real>* next);

parthenon::TaskStatus AccumulateRKStateMeshTask(parthenon::MeshData<parthenon::Real>* current,
                                                parthenon::MeshData<parthenon::Real>* accumulator,
                                                parthenon::Real delta, bool initialize);

parthenon::TaskStatus FloorChiMeshTask(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus
EnforceAlgebraicConstraintsBlockTask(parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::TaskStatus
EnforceAlgebraicConstraintsMeshTask(parthenon::MeshData<parthenon::Real>* data);

parthenon::Real EstimateTimestepBlock(parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::Real EstimateTimestepMesh(parthenon::MeshData<parthenon::Real>* data);

// Apply the uniform-grid SYNC nr_outflow operation to all local blocks through
// packed MeshData kernels. AMR and mixed/user boundaries keep the general
// MeshBlock callback path.
parthenon::TaskStatus ApplyPackedSyncOutflowBoundariesMeshTask(
    std::shared_ptr<parthenon::MeshData<parthenon::Real>>& data, bool coarse);

// Advance the compact-object ODE tracker once after a complete RK step.
void AdvancePunctureTracker(parthenon::Mesh* mesh, parthenon::Real dt);

// Wave extraction and apparent-horizon diagnostics are evaluated only after a
// complete RK step, when ADM and Weyl derived fields represent one time level.
void RunPostStepDiagnostics(parthenon::Mesh* mesh, parthenon::Real time, int cycle);

void CheckRefinementMesh(parthenon::MeshData<parthenon::Real>* data,
                         parthenon::ParArray1D<parthenon::AmrTag>& amr_tags);

} // namespace pangu::nr

#endif
