#pragma once
#include "pangu/plugin_api/plugin.h"
namespace pangu::plugins::scalar_collapse {
struct Plugin {
  static std::shared_ptr<parthenon::StateDescriptor> Initialize(plugin::InitContext &);
};
}
