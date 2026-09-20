#ifndef PANGU_RADIATION_TRANSPORT_PACKAGE_H_
#define PANGU_RADIATION_TRANSPORT_PACKAGE_H_

#include <parthenon/parthenon.hpp>

namespace pangu::radiation {

std::shared_ptr<parthenon::StateDescriptor>
InitializeTransport(parthenon::ParameterInput* pin, const parthenon::Packages_t& packages);

parthenon::TaskStatus CalculateTransportFluxesMeshTask(parthenon::MeshData<parthenon::Real>* data);

parthenon::TaskStatus UpdateTransportMeshTask(parthenon::MeshData<parthenon::Real>* current,
                                              parthenon::MeshData<parthenon::Real>* base,
                                              parthenon::Real gam0, parthenon::Real gam1,
                                              parthenon::Real beta_dt,
                                              parthenon::MeshData<parthenon::Real>* next);

parthenon::TaskStatus CoupleTransportToFluidMeshTask(parthenon::MeshData<parthenon::Real>* data,
                                                     parthenon::Real beta_dt);

parthenon::TaskStatus ApplyBeamSourceMeshTask(parthenon::MeshData<parthenon::Real>* data,
                                              parthenon::Real beta_dt);

} // namespace pangu::radiation

#endif
