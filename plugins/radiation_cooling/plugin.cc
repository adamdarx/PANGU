#include "plugin.h"
#include "radiation/radiation_package.h"
namespace pangu::plugins::radiation_cooling {
std::shared_ptr<parthenon::StateDescriptor> Plugin::Initialize(plugin::InitContext& context) {
  return radiation::InitializeCooling(&context.input, context.dependencies);
}
} // namespace pangu::plugins::radiation_cooling
