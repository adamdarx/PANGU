#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "plugin/registry.h"

namespace {

using pangu::plugin::PluginEntry;
using pangu::plugin::PluginManifest;

PluginEntry Entry(std::string name, std::vector<std::string> dependencies = {},
                  std::vector<std::string> required = {},
                  std::vector<std::string> provided = {},
                  std::vector<std::string> conflicts = {}) {
  return {PluginManifest{std::move(name), "1.0.0", "", std::move(dependencies), {},
                         std::move(required), std::move(provided), std::move(conflicts)},
          nullptr};
}

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Function>
void RequireFailure(Function function, const std::string& needle) {
  try {
    function();
  } catch (const std::runtime_error& error) {
    Require(std::string(error.what()).find(needle) != std::string::npos,
            "unexpected error: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected failure containing: " + needle);
}

} // namespace

int main() {
  try {
    const std::vector<PluginEntry> diffusion_graph{
        Entry("diffusion", {}, {}, {"physics.diffusion"}),
        Entry("ohmic", {"diffusion"}, {"physics.diffusion"},
              {"diffusion.operator.ohmic"}),
        Entry("viscosity", {"diffusion"}, {"physics.diffusion"},
              {"diffusion.operator.viscosity"})};
    const auto enabled = pangu::plugin::Resolve(diffusion_graph, {"viscosity", "ohmic"});
    Require(enabled.size() == 3, "dependency closure has the wrong size");
    Require(enabled[0].entry->manifest.name == "diffusion" && !enabled[0].requested,
            "diffusion must be activated first as a dependency");
    Require(enabled[1].entry->manifest.name == "ohmic" && enabled[1].requested,
            "independent plugins must use deterministic name order");
    Require(enabled[2].entry->manifest.name == "viscosity" && enabled[2].requested,
            "second requested plugin is missing");

    const auto parsed = pangu::plugin::ParseEnabled(" ohmic, viscosity,ohmic ");
    Require(parsed == std::vector<std::string>({"ohmic", "viscosity"}),
            "enabled plugin parsing must trim and deduplicate names");

    RequireFailure([&] { pangu::plugin::Resolve(diffusion_graph, {"missing"}); },
                   "not installed");
    RequireFailure(
        [&] { pangu::plugin::Resolve({Entry("a", {"b"}), Entry("b", {"a"})}, {"a"}); },
        "cycle");
    RequireFailure(
        [&] { pangu::plugin::Resolve({Entry("ohmic", {}, {"fluid.mhd"})}, {"ohmic"}); },
        "fluid.mhd");
    RequireFailure(
        [&] {
          pangu::plugin::Resolve({Entry("a", {}, {}, {}, {"b"}), Entry("b")},
                                 {"a", "b"});
        },
        "conflict");

    std::cout << "plugin registry tests: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "plugin registry tests: FAIL: " << error.what() << '\n';
    return 1;
  }
}
