#pragma once

#include <string>
#include <vector>

#include <parthenon/parthenon.hpp>

namespace pangu::plugin {

// A non-fluid matter system advanced with the native synchronized Z4c graph.
// prepare reconstructs its derived fields and nr.tmunu from the CURRENT state.
// rhs must finish before either the geometric or matter update changes state.
struct SynchronizedMatter {
  std::vector<std::string> evolved;
  parthenon::TaskStatus (*prepare)(parthenon::MeshData<parthenon::Real>*, parthenon::Real);
  parthenon::TaskStatus (*rhs)(parthenon::MeshData<parthenon::Real>*);
  parthenon::TaskStatus (*classical_update)(parthenon::MeshData<parthenon::Real>*, int,
                                           parthenon::Real);
};

inline constexpr const char* synchronized_matter_key = "plugin/synchronized_matter";

inline const SynchronizedMatter* GetSynchronizedMatter(const parthenon::Packages_t& packages) {
  const SynchronizedMatter* found = nullptr;
  for (const auto& [name, package] : packages.AllPackages()) {
    if (!package->AllParams().hasKey(synchronized_matter_key)) continue;
    PARTHENON_REQUIRE(found == nullptr, "only one synchronized non-fluid matter system is allowed");
    found = &package->Param<SynchronizedMatter>(synchronized_matter_key);
  }
  return found;
}

} // namespace pangu::plugin
