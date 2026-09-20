#ifndef PANGU_DRIVER_STAGE_EXTENSION_H_
#define PANGU_DRIVER_STAGE_EXTENSION_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <parthenon/parthenon.hpp>

namespace pangu::driver {

using StageData = parthenon::MeshData<parthenon::Real>;
using StageBlockData = std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>;
using ConservedUpdateMeshTask = parthenon::TaskStatus (*)(StageData*, StageData*, parthenon::Real,
                                                          parthenon::Real, parthenon::Real,
                                                          StageData*);

// A fluid package's stage fields and the updates that belong to it.
struct FluidStageExtension {
  std::vector<std::string> conserved;     // advanced by the cell RK update
  std::vector<std::string> stage_derived; // stage-start state read by sources
  bool (*has_stage_sources)(const parthenon::Packages_t&);
  ConservedUpdateMeshTask update_face_fields; // nullptr without face fields
};

// A passive field advected with the fluid and optionally heated after recovery.
struct PassiveStageExtension {
  using CalculateFluxesMeshTask = parthenon::TaskStatus (*)(StageData*);
  using ApplyFluxCorrectionMeshTask = parthenon::TaskStatus (*)(StageData*, StageData*,
                                                                parthenon::Real, parthenon::Real,
                                                                parthenon::Real,
                                                                const std::string&);
  using ConvertMeshTask = parthenon::TaskStatus (*)(StageData*);
  using ApplyHeatingMeshTask = parthenon::TaskStatus (*)(StageData*, StageData*);
  using FluidFluxesBlockTask = parthenon::TaskStatus (*)(StageBlockData&, const std::string&);
  using FluidFluxCorrectionBlockTask = parthenon::TaskStatus (*)(StageBlockData&, StageBlockData&,
                                                                 parthenon::Real, parthenon::Real,
                                                                 parthenon::Real,
                                                                 const std::string&);
  using CalculateFluxesBlockTask = parthenon::TaskStatus (*)(StageBlockData&);

  std::string_view conserved_field;
  bool applies_heating;
  CalculateFluxesMeshTask calculate_fluxes;
  ApplyFluxCorrectionMeshTask apply_flux_correction;
  ConservedUpdateMeshTask update_conserved;
  ConvertMeshTask conserved_to_primitive;
  ApplyHeatingMeshTask apply_heating;
  ConvertMeshTask primitive_to_conserved;
  ConvertMeshTask apply_physical_boundaries;
  // Block-level forms used by the block-oriented stage graph.
  FluidFluxesBlockTask calculate_fluid_fluxes_block;
  FluidFluxCorrectionBlockTask apply_fluid_flux_correction_block;
  CalculateFluxesBlockTask calculate_fluxes_block;
};

// Replaces the cell RK update of the fluid conserved fields, for example to advance
// additional ledgers in the same kernel.
struct ConservedUpdateExtension {
  std::string_view name;
  ConservedUpdateMeshTask update;
};

// A stage source applied to the updated state before boundary exchange.
struct StageSourceExtension {
  using AddSourceTask = parthenon::TaskID (*)(parthenon::TaskID, parthenon::TaskID,
                                              parthenon::TaskList&, StageData*, parthenon::Real,
                                              parthenon::Real, bool);

  std::string_view name;
  AddSourceTask add_source_task;
};

inline constexpr std::string_view fluid_stage_extension_key = "driver/fluid_stage_extension";
inline constexpr std::string_view passive_stage_extension_key = "driver/passive_stage_extension";
inline constexpr std::string_view stage_source_extension_key = "driver/stage_source_extension";
inline constexpr std::string_view conserved_update_extension_key =
    "driver/conserved_update_extension";

template <typename Extension>
std::vector<const Extension*> FindStageExtensions(const parthenon::Packages_t& packages,
                                                  const std::string_view key) {
  std::vector<const Extension*> extensions;
  for (const auto& [name, package] : packages.AllPackages()) {
    if (package->AllParams().hasKey(std::string(key)))
      extensions.push_back(&package->template Param<Extension>(std::string(key)));
  }
  return extensions;
}

// Returns the single registered extension of a kind, or nullptr.
template <typename Extension>
const Extension* FindStageExtension(const parthenon::Packages_t& packages,
                                    const std::string_view key) {
  const auto extensions = FindStageExtensions<Extension>(packages, key);
  PARTHENON_REQUIRE(extensions.size() <= 1, "at most one package may register " + std::string(key));
  return extensions.empty() ? nullptr : extensions.front();
}

} // namespace pangu::driver

#endif
