#pragma once
#include "pangu/plugin_api/plugin.h"
namespace pangu::plugins::compact_objects {
std::shared_ptr<parthenon::StateDescriptor> InitializeBNS(plugin::InitContext &);
}
