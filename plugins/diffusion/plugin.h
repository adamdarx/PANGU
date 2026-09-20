#ifndef PANGU_PLUGINS_DIFFUSION_PLUGIN_H_
#define PANGU_PLUGINS_DIFFUSION_PLUGIN_H_

#include "pangu/plugin_api/plugin.h"

namespace pangu::plugins::diffusion {

struct Plugin {
  static std::shared_ptr<parthenon::StateDescriptor> Initialize(plugin::InitContext& context);
};

} // namespace pangu::plugins::diffusion

#endif
