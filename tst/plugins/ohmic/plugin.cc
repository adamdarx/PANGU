#include "plugin.h"

namespace pangu_test_plugins::ohmic {

std::shared_ptr<parthenon::StateDescriptor> Plugin::Initialize(pangu::plugin::InitContext&) {
  return std::make_shared<parthenon::StateDescriptor>("plugin.ohmic");
}

} // namespace pangu_test_plugins::ohmic
