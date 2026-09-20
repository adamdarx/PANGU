#ifndef PANGU_RELATIVITY_PHYSICS_ASSEMBLY_H_
#define PANGU_RELATIVITY_PHYSICS_ASSEMBLY_H_

#include <string>

#include "estimator/estimator.h"
#include "geometry_assembly.h"
#include "pangu_config.h"
#include "relativity/relativistic_hydro.h"
#include "utils/error_checking.hpp"

namespace pangu::relativity {

#if PANGU_PHYSICS_SR
inline constexpr HydroMode compiled_physics = HydroMode::sr;
#else
inline constexpr HydroMode compiled_physics = HydroMode::gr;
#endif

// Without characteristic GR speeds the estimator assumes a unit coordinate light
// speed, which only some fixed metrics bound; synchronized spacetimes estimate
// their own timestep.
static_assert(estimator::Selected::characteristic_gr_speeds ||
                  geometry::ConfiguredMetricHasUnitLightBound() ||
                  geometry::ConfiguredGeometryIsSynchronized(),
              "the configured METRIC requires an ESTIMATOR with characteristic GR speeds");

inline void ValidateCompiledPhysics(const HydroMode requested, const char* parameter) {
  if (requested != HydroMode::newtonian) {
    PARTHENON_REQUIRE(requested == compiled_physics,
                      std::string(parameter) + " does not match compile-time PHYSICS=" +
                          PANGU_PHYSICS);
  }
}

} // namespace pangu::relativity

#endif
