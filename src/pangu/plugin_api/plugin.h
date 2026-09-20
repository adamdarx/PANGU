#ifndef PANGU_PLUGIN_API_PLUGIN_H_
#define PANGU_PLUGIN_API_PLUGIN_H_

#include <concepts>
#include <memory>

#include <parthenon/parthenon.hpp>

namespace pangu::plugin {

inline constexpr int kAPIVersion = 1;

struct InitContext {
  parthenon::ParameterInput& input;
  const parthenon::Packages_t& dependencies;
};

// A plugin type passed as TYPE in plugin.cmake must provide this operation.
// Device-side algorithms remain statically compiled inside that type.
template <typename Plugin>
concept PluginType = requires(InitContext& context) {
  { Plugin::Initialize(context) } ->
      std::same_as<std::shared_ptr<parthenon::StateDescriptor>>;
};

} // namespace pangu::plugin

#endif
