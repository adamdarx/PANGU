#ifndef PANGU_PLUGINS_RADIATION_COOLING_PLUGIN_H_
#define PANGU_PLUGINS_RADIATION_COOLING_PLUGIN_H_
#include "pangu/plugin_api/plugin.h"
namespace pangu::plugins::radiation_cooling {
struct Plugin {
  static std::shared_ptr<parthenon::StateDescriptor> Initialize(plugin::InitContext& context);
};
} // namespace pangu::plugins::radiation_cooling
#endif
