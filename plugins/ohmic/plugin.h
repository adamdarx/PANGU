#ifndef PANGU_PLUGINS_OHMIC_PLUGIN_H_
#define PANGU_PLUGINS_OHMIC_PLUGIN_H_
#include "pangu/plugin_api/plugin.h"
namespace pangu::plugins::ohmic {
struct Plugin {
  static std::shared_ptr<parthenon::StateDescriptor> Initialize(plugin::InitContext& context);
};
} // namespace pangu::plugins::ohmic
#endif
