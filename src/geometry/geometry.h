#ifndef PANGU_GEOMETRY_GEOMETRY_H_
#define PANGU_GEOMETRY_GEOMETRY_H_

#include <memory>
#include <type_traits>

#include <parthenon/parthenon.hpp>

namespace pangu::geometry {

enum class Background { minkowski = 0, kerr_schild = 1 };
enum class Location { cell_center = 0, face1 = 1, face2 = 2, face3 = 3 };
enum class Symmetry { constant = 0, radial_1d = 1, axisymmetric_2d = 2, general_3d = 3 };

KOKKOS_INLINE_FUNCTION constexpr Location FaceLocation(const int direction) {
  return direction == 0 ? Location::face1 : (direction == 1 ? Location::face2 : Location::face3);
}

struct MetricPoint {
  parthenon::Real lower[4][4]{};
  parthenon::Real upper[4][4]{};
  parthenon::Real lapse = 1.0;
  parthenon::Real shift[3]{};
  parthenon::Real spatial_det = 1.0;
  parthenon::Real gdet = 1.0;
};

struct MetricDerivatives {
  parthenon::Real lower[3][4][4]{};
};

struct SphericalKerrSchildPoint {
  parthenon::Real radius = 0.0;
  parthenon::Real theta = 0.0;
  parthenon::Real phi = 0.0;
};

struct ExcisionParameters {
  bool enabled = false;
  parthenon::Real radius = 1.0;
  parthenon::Real density = -1.0;
  parthenon::Real pressure = -1.0;
};

template <class MetricType> ExcisionParameters ConfigureExcision(parthenon::ParameterInput* pin) {
  if constexpr (MetricType::supports_excision)
    return MetricType::ConfigureExcision(pin);
  return {};
}

KOKKOS_INLINE_FUNCTION constexpr int SymmetricComponent(int first, int second) {
  if (second < first) {
    const int temporary = first;
    first = second;
    second = temporary;
  }
  return first * 4 - first * (first - 1) / 2 + second - first;
}

static_assert(SymmetricComponent(0, 0) == 0);
static_assert(SymmetricComponent(0, 3) == 3);
static_assert(SymmetricComponent(1, 1) == 4);
static_assert(SymmetricComponent(3, 3) == 9);

template <class MetricType, class Mode> class Geometry;

std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput* pin);

} // namespace pangu::geometry

#endif
