#include "plugin.h"

#include "electron/electron_package.h"

namespace pangu::plugins::electron {

std::shared_ptr<parthenon::StateDescriptor> Plugin::Initialize(plugin::InitContext& context) {
  return pangu::electron::Initialize(&context.input, context.dependencies);
}

} // namespace pangu::plugins::electron
