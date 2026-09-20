#ifndef PANGU_PLUGIN_REGISTRY_H_
#define PANGU_PLUGIN_REGISTRY_H_

#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "pangu/plugin_api/plugin.h"

namespace pangu::plugin {

struct PluginManifest {
  std::string name;
  std::string version;
  std::string description;
  std::vector<std::string> dependencies;
  std::vector<std::string> dependency_requirements;
  std::vector<std::string> required_capabilities;
  std::vector<std::string> provided_capabilities;
  std::vector<std::string> conflicts;
};

using Initialize = std::shared_ptr<parthenon::StateDescriptor> (*)(InitContext&);

struct PluginEntry {
  PluginManifest manifest;
  Initialize initialize;
};

template <PluginType Plugin>
PluginEntry MakeEntry(PluginManifest manifest) {
  return {std::move(manifest), &Plugin::Initialize};
}

struct Activation {
  const PluginEntry* entry;
  bool requested;
};

std::vector<std::string> ParseEnabled(std::string_view value);
std::vector<Activation> Resolve(const std::vector<PluginEntry>& entries,
                                const std::vector<std::string>& requested,
                                const std::vector<std::string>& core_capabilities = {});
const std::vector<PluginEntry>& CompiledPlugins();
bool IsCompiled(std::string_view name);
void AddEnabledPackages(parthenon::ParameterInput* input, parthenon::Packages_t& packages);
void ListCompiled(std::ostream& output);

} // namespace pangu::plugin

#endif
