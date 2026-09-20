#include "plugin/registry.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifndef PANGU_PLUGIN_REGISTRY_TESTING
#include "pangu/plugin_registry.h"
#endif
#include "pangu_config.h"

namespace pangu::plugin {

std::vector<std::string> ParseEnabled(const std::string_view value) {
  std::vector<std::string> names;
  std::stringstream stream{std::string(value)};
  for (std::string name; std::getline(stream, name, ',');) {
    const auto first = name.find_first_not_of(" \t");
    if (first == std::string::npos)
      continue;
    const auto last = name.find_last_not_of(" \t");
    name = name.substr(first, last - first + 1);
    if (std::find(names.begin(), names.end(), name) == names.end())
      names.push_back(std::move(name));
  }
  return names;
}

std::vector<Activation> Resolve(const std::vector<PluginEntry>& entries,
                                const std::vector<std::string>& requested,
                                const std::vector<std::string>& core_capabilities) {
  std::map<std::string, const PluginEntry*> available;
  for (const auto& entry : entries) {
    if (entry.manifest.name.empty() || !available.emplace(entry.manifest.name, &entry).second)
      throw std::runtime_error("Invalid or duplicate plugin name '" + entry.manifest.name + "'");
  }

  std::set<std::string> direct(requested.begin(), requested.end());
  std::map<std::string, int> state;
  std::vector<const PluginEntry*> ordered;
  std::function<void(const std::string&, const std::string&)> visit =
      [&](const std::string& name, const std::string& chain) {
        const auto found = available.find(name);
        if (found == available.end())
          throw std::runtime_error("Plugin is not installed: " + chain + name);
        if (state[name] == 1)
          throw std::runtime_error("Plugin dependency cycle: " + chain + name);
        if (state[name] == 2)
          return;
        state[name] = 1;
        auto dependencies = found->second->manifest.dependencies;
        std::sort(dependencies.begin(), dependencies.end());
        for (const auto& dependency : dependencies)
          visit(dependency, chain + name + " -> ");
        state[name] = 2;
        ordered.push_back(found->second);
      };
  auto requested_order = requested;
  std::sort(requested_order.begin(), requested_order.end());
  for (const auto& name : requested_order)
    visit(name, "");

  std::set<std::string> selected_names;
  for (const auto* entry : ordered)
    selected_names.insert(entry->manifest.name);
  for (const auto* entry : ordered) {
    for (const auto& conflict : entry->manifest.conflicts) {
      if (selected_names.contains(conflict))
        throw std::runtime_error("Plugin conflict: '" + entry->manifest.name + "' and '" +
                                 conflict + "'");
    }
  }

  std::set<std::string> capabilities(core_capabilities.begin(), core_capabilities.end());
  std::vector<Activation> result;
  for (const auto* entry : ordered) {
    for (const auto& required : entry->manifest.required_capabilities) {
      if (!capabilities.contains(required))
        throw std::runtime_error("Plugin '" + entry->manifest.name +
                                 "' requires unavailable capability '" + required + "'");
    }
    capabilities.insert(entry->manifest.provided_capabilities.begin(),
                        entry->manifest.provided_capabilities.end());
    result.push_back({entry, direct.contains(entry->manifest.name)});
  }
  return result;
}

const std::vector<PluginEntry>& CompiledPlugins() {
#ifdef PANGU_PLUGIN_REGISTRY_TESTING
  static const std::vector<PluginEntry> entries;
#else
  static const auto entries = generated::Entries();
#endif
  return entries;
}

bool IsCompiled(const std::string_view name) {
  const auto& entries = CompiledPlugins();
  return std::any_of(entries.begin(), entries.end(),
                     [name](const auto& entry) { return entry.manifest.name == name; });
}

void AddEnabledPackages(parthenon::ParameterInput* input, parthenon::Packages_t& packages) {
  const auto requested = ParseEnabled(input->GetOrAddString("plugins", "enabled", ""));
  std::vector<std::string> capabilities{"pangu.core"};
  const auto& core_packages = packages.AllPackages();
  if (core_packages.contains("hydro"))
    capabilities.emplace_back("fluid.hydro");
  if (core_packages.contains("mhd"))
    capabilities.emplace_back("fluid.mhd");
  if (core_packages.contains("hydro") || core_packages.contains("mhd"))
    capabilities.emplace_back("fluid");
  if (core_packages.contains("geometry"))
    capabilities.emplace_back("geometry");
  if (core_packages.contains("radiation"))
    capabilities.emplace_back("radiation");
  if (core_packages.contains("electrons"))
    capabilities.emplace_back("electron.thermodynamics");
#if PANGU_ENABLE_CUDA
  capabilities.emplace_back("backend.cuda");
#endif
#if PANGU_ENABLE_MPI
  capabilities.emplace_back("backend.mpi");
#endif
  const auto activations = Resolve(CompiledPlugins(), requested, capabilities);
  if (!activations.empty() && parthenon::Globals::my_rank == 0) {
    std::cout << "Activated plugins:\n";
    for (const auto& activation : activations)
      std::cout << "  " << activation.entry->manifest.name << ' '
                << activation.entry->manifest.version << " ("
                << (activation.requested ? "requested" : "dependency") << ")\n";
  }
  for (const auto& activation : activations) {
    InitContext context{*input, packages};
    auto package = activation.entry->initialize(context);
    if (!package)
      throw std::runtime_error("Plugin '" + activation.entry->manifest.name +
                               "' returned an empty package");
    packages.Add(std::move(package));
  }
}

void ListCompiled(std::ostream& output) {
  const auto& entries = CompiledPlugins();
  if (entries.empty()) {
    output << "Compiled plugins: none\n";
    return;
  }
  output << "Compiled plugins:\n";
  for (const auto& entry : entries) {
    output << "  " << entry.manifest.name << ' ' << entry.manifest.version;
    if (!entry.manifest.dependency_requirements.empty()) {
      output << " (requires ";
      for (std::size_t i = 0; i < entry.manifest.dependency_requirements.size(); ++i) {
        if (i != 0)
          output << ", ";
        output << entry.manifest.dependency_requirements[i];
      }
      output << ')';
    }
    if (!entry.manifest.description.empty())
      output << " - " << entry.manifest.description;
    output << '\n';
  }
}

} // namespace pangu::plugin
