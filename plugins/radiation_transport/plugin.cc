#include "plugin.h"
#include "radiation/transport_package.h"
namespace pangu::plugins::radiation_transport {
std::shared_ptr<parthenon::StateDescriptor> Plugin::Initialize(plugin::InitContext& context) {
  return radiation::InitializeTransport(&context.input, context.dependencies);
}
} // namespace pangu::plugins::radiation_transport
