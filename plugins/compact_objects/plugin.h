#ifndef PANGU_PLUGINS_COMPACT_OBJECTS_H_
#define PANGU_PLUGINS_COMPACT_OBJECTS_H_
#include "pangu/plugin_api/plugin.h"
namespace pangu::plugins::compact_objects {
struct Plugin {
  static std::shared_ptr<parthenon::StateDescriptor> Initialize(plugin::InitContext &);
};
} // namespace pangu::plugins::compact_objects
#endif
