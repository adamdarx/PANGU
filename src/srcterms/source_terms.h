#ifndef PANGU_SRCTERMS_SOURCE_TERMS_H_
#define PANGU_SRCTERMS_SOURCE_TERMS_H_

#include <memory>

#include <parthenon/parthenon.hpp>

namespace pangu::srcterms {

std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput* pin);
parthenon::TaskStatus
ApplyHydroBlockTask(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data,
                    parthenon::Real dt);
parthenon::TaskStatus
ApplyMHDBlockTask(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data,
                  parthenon::Real dt);
parthenon::Real EstimateTimestepBlock(parthenon::MeshBlockData<parthenon::Real>* data);
// True when any configured source term changes the fluid during a stage.
bool HasActiveSources(const parthenon::Packages_t& packages);

} // namespace pangu::srcterms

#endif
