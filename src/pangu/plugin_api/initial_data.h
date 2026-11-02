#ifndef PANGU_PLUGIN_API_INITIAL_DATA_H_
#define PANGU_PLUGIN_API_INITIAL_DATA_H_

#include <string>
#include <parthenon/parthenon.hpp>

namespace pangu::plugin_api {

// Optional problem initialization, independent of the evolution stage graph.
// Exactly one enabled package supplies this callback for a plugin-owned problem.
using BlockInitialData = void (*)(parthenon::MeshBlock *, parthenon::ParameterInput *);
inline constexpr const char *block_initial_data_key = "plugin/block_initial_data";

inline void InitializeBlockFromPlugin(parthenon::MeshBlock *block,
                                     parthenon::ParameterInput *input) {
  BlockInitialData selected = nullptr;
  for (const auto &[name, package] : block->packages.AllPackages()) {
    if (!package->AllParams().hasKey(block_initial_data_key)) continue;
    PARTHENON_REQUIRE(selected == nullptr,
                      "a plugin-owned problem requires exactly one initial-data provider");
    selected = package->Param<BlockInitialData>(block_initial_data_key);
    PARTHENON_REQUIRE(selected != nullptr, "plugin initial-data callback is null");
  }
  PARTHENON_REQUIRE(selected != nullptr,
                    "a plugin-owned problem requires an enabled initial-data plugin");
  selected(block, input);
}

} // namespace pangu::plugin_api

#endif
