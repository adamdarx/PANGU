#include "geometry_assembly.h"

#include <type_traits>

namespace pangu::geometry {

static_assert(std::is_copy_constructible_v<
              decltype(ConstructGeometry(Background::kerr_schild, parthenon::Real{0.0}, nullptr))>);
static_assert(std::is_same_v<decltype(ConfiguredModeSupportsRefinement()), bool>);
static_assert(std::is_same_v<decltype(ConfiguredMetricSupportsExcision()), bool>);
static_assert(std::is_same_v<decltype(ConfiguredMetricSymmetry()), Symmetry>);

} // namespace pangu::geometry
