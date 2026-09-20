#ifndef PANGU_MHD_STAGE_EXTENSION_H_
#define PANGU_MHD_STAGE_EXTENSION_H_

#include <string>
#include <string_view>

#include <parthenon/parthenon.hpp>

namespace pangu::mhd {

struct PassiveStageExtension {
  using Data = parthenon::MeshData<parthenon::Real>;
  using CalculateFluxesMeshTask = parthenon::TaskStatus (*)(Data*);
  using ApplyFluxCorrectionMeshTask = parthenon::TaskStatus (*)(
      Data*, Data*, parthenon::Real, parthenon::Real, parthenon::Real,
      const std::string&);
  using UpdateConservedMeshTask = parthenon::TaskStatus (*)(
      Data*, Data*, parthenon::Real, parthenon::Real, parthenon::Real, Data*);
  using ConvertMeshTask = parthenon::TaskStatus (*)(Data*);
  using ApplyHeatingMeshTask = parthenon::TaskStatus (*)(Data*, Data*);

  std::string_view conserved_field;
  bool applies_heating;
  CalculateFluxesMeshTask calculate_fluxes;
  ApplyFluxCorrectionMeshTask apply_flux_correction;
  UpdateConservedMeshTask update_conserved;
  ConvertMeshTask conserved_to_primitive;
  ApplyHeatingMeshTask apply_heating;
  ConvertMeshTask primitive_to_conserved;
  ConvertMeshTask apply_physical_boundaries;
};

struct StageSourceExtension {
  using Data = parthenon::MeshData<parthenon::Real>;
  using AddSourceTask = parthenon::TaskID (*)(
      parthenon::TaskID, parthenon::TaskID, parthenon::TaskList&, Data*,
      parthenon::Real, parthenon::Real, bool);

  std::string_view name;
  AddSourceTask add_source_task;
};

inline constexpr std::string_view passive_stage_extension_key =
    "mhd/passive_stage_extension";
inline constexpr std::string_view stage_source_extension_key =
    "mhd/stage_source_extension";

} // namespace pangu::mhd

#endif
