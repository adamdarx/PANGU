#ifndef PANGU_HYDRO_HYDRO_PACKAGE_H_
#define PANGU_HYDRO_HYDRO_PACKAGE_H_

#include <memory>

#include <parthenon/parthenon.hpp>

namespace pangu::hydro {

std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput* pin);

parthenon::TaskStatus
CalculateFluxesBlockTask(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data);

// Pack-wide fast path used by the driver to amortize GPU launch overhead
// across all blocks in a MeshData partition. Relativistic dispatch is handled
// by the fixed-background Hydro package.
parthenon::TaskStatus CalculateFluxesMeshTask(parthenon::MeshData<parthenon::Real>* data);

parthenon::TaskStatus ApplyFirstOrderFluxCorrectionBlockTask(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data,
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& base, parthenon::Real gam0,
    parthenon::Real gam1, parthenon::Real beta_dt);
parthenon::TaskStatus ApplyFirstOrderFluxCorrectionMeshTask(
    parthenon::MeshData<parthenon::Real>* data, parthenon::MeshData<parthenon::Real>* base,
    parthenon::Real gam0, parthenon::Real gam1, parthenon::Real beta_dt);

parthenon::TaskStatus
ApplySourcesBlockTask(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data,
                      parthenon::Real dt);
parthenon::TaskStatus ApplySourcesMeshTask(parthenon::MeshData<parthenon::Real>* data,
                                           parthenon::Real dt);

void ConservedToPrimitiveBlock(parthenon::MeshBlockData<parthenon::Real>* data);
void ConservedToPrimitiveMesh(parthenon::MeshData<parthenon::Real>* data);
parthenon::Real EstimateTimestepBlock(parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::Real EstimateTimestepMesh(parthenon::MeshData<parthenon::Real>* data);
parthenon::AmrTag CheckRefinementBlock(parthenon::MeshBlockData<parthenon::Real>* data);

// Carries the hydro state unchanged into the next stage buffer.
parthenon::TaskStatus CarryStateMeshTask(parthenon::MeshData<parthenon::Real>* current,
                                         parthenon::MeshData<parthenon::Real>* next);

} // namespace pangu::hydro

#endif
