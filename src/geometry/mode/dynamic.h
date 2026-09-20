#ifndef PANGU_GEOMETRY_MODE_DYNAMIC_H_
#define PANGU_GEOMETRY_MODE_DYNAMIC_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "geometry/geometry.h"

namespace pangu::geometry {

struct DynamicMode {
  static constexpr bool supports_refinement = true;
  static constexpr bool synchronized = false;
};

template <class MetricType> class Geometry<MetricType, DynamicMode> {
public:
  using Parameters = typename MetricType::Parameters;
  static constexpr bool supports_excision = MetricType::supports_excision;
  static constexpr bool unit_determinant = MetricType::unit_determinant;
  static constexpr bool unit_coordinate_light_bound = MetricType::unit_coordinate_light_bound;

  KOKKOS_INLINE_FUNCTION explicit Geometry(const Parameters& parameters)
      : parameters_(parameters) {}

  KOKKOS_INLINE_FUNCTION MetricPoint MetricAt(const Location location, const parthenon::Real x1,
                                              const parthenon::Real x2,
                                              const parthenon::Real x3) const {
    return MetricType::Calculate(x1, x2, x3, location, parameters_);
  }

  KOKKOS_INLINE_FUNCTION MetricDerivatives DerivativesAt(const parthenon::Real x1,
                                                         const parthenon::Real x2,
                                                         const parthenon::Real x3) const {
    return MetricType::CalculateDerivatives(x1, x2, x3, parameters_);
  }

  KOKKOS_INLINE_FUNCTION const Parameters& GetParameters() const { return parameters_; }

  KOKKOS_INLINE_FUNCTION SphericalKerrSchildPoint SphericalCoordinates(
      const parthenon::Real x1, const parthenon::Real x2, const parthenon::Real x3) const {
    return MetricType::SphericalCoordinates(x1, x2, x3, parameters_);
  }

  KOKKOS_INLINE_FUNCTION void
  BoyerLindquistSpatialToNative(const parthenon::Real x1, const parthenon::Real x2,
                                const parthenon::Real x3, const parthenon::Real boyer_lindquist[3],
                                parthenon::Real native[3]) const {
    MetricType::BoyerLindquistSpatialToNative(x1, x2, x3, boyer_lindquist, native, parameters_);
  }

  KOKKOS_INLINE_FUNCTION void AzimuthalCovectorToNative(const parthenon::Real x1,
                                                        const parthenon::Real x2,
                                                        const parthenon::Real x3,
                                                        const parthenon::Real azimuthal,
                                                        parthenon::Real native[3]) const {
    MetricType::AzimuthalCovectorToNative(x1, x2, x3, azimuthal, native, parameters_);
  }

  KOKKOS_INLINE_FUNCTION bool IsExcised(const parthenon::Real x1, const parthenon::Real x2,
                                        const parthenon::Real x3,
                                        const parthenon::Real radius) const {
    if constexpr (MetricType::supports_excision)
      return MetricType::ExcisionRadius(x1, x2, x3, parameters_) <= radius;
    return false;
  }

  template <class Coordinates>
  KOKKOS_INLINE_FUNCTION bool NeedsFluxExcision(const Coordinates& coordinates, const int k,
                                                const int j, const int i,
                                                const parthenon::Real radius) const {
    if constexpr (MetricType::supports_excision)
      return MetricType::NeedsFluxExcision(coordinates, k, j, i, parameters_, radius);
    return false;
  }

  template <class Coordinates>
  KOKKOS_INLINE_FUNCTION MetricPoint MetricAt(const Location location, const int, const int k,
                                              const int j, const int i,
                                              const Coordinates& coordinates) const {
    const parthenon::Real x1 =
        location == Location::face1 ? coordinates.template Xf<1>(i) : coordinates.template Xc<1>(i);
    const parthenon::Real x2 =
        location == Location::face2 ? coordinates.template Xf<2>(j) : coordinates.template Xc<2>(j);
    const parthenon::Real x3 =
        location == Location::face3 ? coordinates.template Xf<3>(k) : coordinates.template Xc<3>(k);
    return MetricAt(location, x1, x2, x3);
  }

  // Keep the Geometry access signature uniform across compile-time modes.
  // Stage packs are meaningful only for SyncMode; analytic dynamic geometry
  // evaluates directly from coordinates and therefore ignores them.
  template <class Coordinates, class Z4cPack, class ADMPack>
  KOKKOS_INLINE_FUNCTION MetricPoint MetricAt(const Location location, const int block, const int k,
                                              const int j, const int i,
                                              const Coordinates& coordinates, const Z4cPack&,
                                              const ADMPack&) const {
    return MetricAt(location, block, k, j, i, coordinates);
  }

  template <class Coordinates>
  KOKKOS_INLINE_FUNCTION MetricDerivatives DerivativesAt(const int, const int k, const int j,
                                                         const int i,
                                                         const Coordinates& coordinates) const {
    return DerivativesAt(coordinates.template Xc<1>(i), coordinates.template Xc<2>(j),
                         coordinates.template Xc<3>(k));
  }

  KOKKOS_INLINE_FUNCTION bool IsInitialized() const { return true; }
  constexpr std::size_t StorageBytes() const { return 0; }
  constexpr std::uint64_t Fingerprint() const { return 0; }

private:
  Parameters parameters_;
};

static_assert(std::is_trivially_copyable_v<DynamicMode>);

template <class MetricType>
inline void InitializeGeometryMode(parthenon::MeshBlock*, parthenon::ParameterInput*,
                                   Geometry<MetricType, DynamicMode>&) {}

template <class MetricType>
inline void ValidateGeometryModeInput(parthenon::ParameterInput*, DynamicMode) {}

} // namespace pangu::geometry

#endif
