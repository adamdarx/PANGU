#pragma once

#include <memory>

#include <parthenon/parthenon.hpp>

namespace pangu::electron {

bool Enabled(parthenon::ParameterInput* pin);
std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput* pin,
                                                       const parthenon::Packages_t& packages);

void ConservedToPrimitiveBlock(parthenon::MeshBlockData<parthenon::Real>* data);
void ConservedToPrimitiveMesh(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus ConservedToPrimitiveMeshTask(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus ApplyPhysicalBoundariesMeshTask(parthenon::MeshData<parthenon::Real>* data);
void PrimitiveToConservedBlock(parthenon::MeshBlockData<parthenon::Real>* data,
                               parthenon::IndexDomain domain = parthenon::IndexDomain::entire);
void PrimitiveToConservedMesh(parthenon::MeshData<parthenon::Real>* data,
                              parthenon::IndexDomain domain = parthenon::IndexDomain::entire);
parthenon::TaskStatus
PrimitiveToConservedBlockTask(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data);
parthenon::TaskStatus PrimitiveToConservedMeshTask(parthenon::MeshData<parthenon::Real>* data);

parthenon::TaskStatus ApplyHeatingMeshTask(parthenon::MeshData<parthenon::Real>* old_data,
                                           parthenon::MeshData<parthenon::Real>* new_data);

parthenon::TaskStatus
CalculateFluxesBlockTask(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data);
parthenon::TaskStatus CalculateFluxesMeshTask(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus UpdateConservedMeshTask(parthenon::MeshData<parthenon::Real>* current,
                                              parthenon::MeshData<parthenon::Real>* base,
                                              parthenon::Real gam0, parthenon::Real gam1,
                                              parthenon::Real beta_dt,
                                              parthenon::MeshData<parthenon::Real>* next);

} // namespace pangu::electron
