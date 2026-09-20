#ifndef PANGU_PLUGIN_API_DIFFUSION_H_
#define PANGU_PLUGIN_API_DIFFUSION_H_

#include <memory>
#include <string_view>

#include <parthenon/parthenon.hpp>

namespace pangu::plugin_api {

struct DiffusionOperator {
  using BlockData = std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>;
  using AddFluxesBlockTask = parthenon::TaskStatus (*)(BlockData&);

  std::string_view name;
  AddFluxesBlockTask add_fluxes;
};

inline constexpr std::string_view diffusion_operator_key =
    "plugin_api/diffusion_operator";

} // namespace pangu::plugin_api

#endif
