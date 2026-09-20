#ifndef PANGU_TEST_PLUGINS_OHMIC_PLUGIN_H_
#define PANGU_TEST_PLUGINS_OHMIC_PLUGIN_H_

#include "pangu/plugin_api/plugin.h"

namespace pangu_test_plugins::ohmic {

struct Plugin {
  static std::shared_ptr<parthenon::StateDescriptor> Initialize(pangu::plugin::InitContext&);
};

} // namespace pangu_test_plugins::ohmic

#endif
