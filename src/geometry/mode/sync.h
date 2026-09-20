#ifndef PANGU_GEOMETRY_MODE_SYNC_H_
#define PANGU_GEOMETRY_MODE_SYNC_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "geometry/geometry.h"
#include "z4c/component_indices.h"

namespace pangu::geometry {

struct SyncMode {
  static constexpr bool supports_refinement = true;
  static constexpr bool synchronized = true;
};

template <class MetricType> class Geometry<MetricType, SyncMode> {
public:
  using Parameters = typename MetricType::Parameters;
  static constexpr bool supports_excision = false;
  static constexpr bool unit_determinant = false;
  static constexpr bool unit_coordinate_light_bound = false;

  KOKKOS_INLINE_FUNCTION explicit Geometry(const Parameters& parameters)
      : parameters_(parameters) {}

  // A synchronized Geometry must be bound explicitly to the current RK-stage
  // Z4c and ADM packs. This overload accepts both VariablePack and
  // MeshBlockVarPack because both expose operator(block,component,k,j,i).
  template <class Z4cPack, class ADMPack, class Coordinates>
  KOKKOS_INLINE_FUNCTION MetricPoint MetricAt(const Location location, const int block, const int k,
                                              const int j, const int i, const Coordinates&,
                                              const Z4cPack& z4c, const ADMPack&) const {
    const int neighbor_k = location == Location::face3 ? k - 1 : k;
    const int neighbor_j = location == Location::face2 ? j - 1 : j;
    const int neighbor_i = location == Location::face1 ? i - 1 : i;
    const parthenon::Real weight = location == Location::cell_center ? 0.0 : 0.5;

    const auto z4c_value = [&](const int component) {
      const parthenon::Real center = z4c(block, component, k, j, i);
      const parthenon::Real neighbor = z4c(block, component, neighbor_k, neighbor_j, neighbor_i);
      return center + weight * (neighbor - center);
    };
    const auto spatial_value = [&](const int component) {
      // Match AthenaK's Z4cToADM conversion exactly, but form the spatial
      // metric from the current independent Z4c stage rather than relying on
      // a derived ADM field whose lifetime is controlled by Parthenon.  Face
      // values remain the arithmetic average of the two cell-centred ADM
      // metrics used by AthenaK's Face*Metric helpers.
      const parthenon::Real center_chi =
          z4c(block, nr::Index(nr::Z4cComponent::chi), k, j, i);
      const parthenon::Real neighbor_chi = z4c(
          block, nr::Index(nr::Z4cComponent::chi), neighbor_k, neighbor_j, neighbor_i);
      const parthenon::Real center =
          pow(center_chi, 4.0 / parameters_.chi_psi_power) *
          z4c(block, nr::Index(nr::Z4cComponent::gxx) + component, k, j, i);
      const parthenon::Real neighbor =
          pow(neighbor_chi, 4.0 / parameters_.chi_psi_power) *
          z4c(block, nr::Index(nr::Z4cComponent::gxx) + component, neighbor_k,
              neighbor_j, neighbor_i);
      return center + weight * (neighbor - center);
    };

    parthenon::Real spatial[3][3]{};
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        const auto value = spatial_value(nr::SpatialSymmetricComponent(first, second));
        spatial[first][second] = value;
        spatial[second][first] = value;
      }
    }

    const parthenon::Real det =
        spatial[0][0] * (spatial[1][1] * spatial[2][2] - spatial[1][2] * spatial[1][2]) -
        spatial[0][1] * (spatial[0][1] * spatial[2][2] - spatial[0][2] * spatial[1][2]) +
        spatial[0][2] * (spatial[0][1] * spatial[1][2] - spatial[0][2] * spatial[1][1]);
    const parthenon::Real lapse = z4c_value(nr::Index(nr::Z4cComponent::alpha));
    if (!(det > 0.0) || !(lapse > 0.0))
      Kokkos::abort("sync Geometry requires positive spatial determinant and lapse");
    const parthenon::Real inverse_det = 1.0 / det;
    parthenon::Real inverse[3][3]{};
    inverse[0][0] = (spatial[1][1] * spatial[2][2] - spatial[1][2] * spatial[1][2]) * inverse_det;
    inverse[0][1] = (spatial[0][2] * spatial[1][2] - spatial[0][1] * spatial[2][2]) * inverse_det;
    inverse[0][2] = (spatial[0][1] * spatial[1][2] - spatial[0][2] * spatial[1][1]) * inverse_det;
    inverse[1][1] = (spatial[0][0] * spatial[2][2] - spatial[0][2] * spatial[0][2]) * inverse_det;
    inverse[1][2] = (spatial[0][1] * spatial[0][2] - spatial[0][0] * spatial[1][2]) * inverse_det;
    inverse[2][2] = (spatial[0][0] * spatial[1][1] - spatial[0][1] * spatial[0][1]) * inverse_det;
    inverse[1][0] = inverse[0][1];
    inverse[2][0] = inverse[0][2];
    inverse[2][1] = inverse[1][2];

    MetricPoint result{};
    result.lapse = lapse;
    result.shift[0] = z4c_value(nr::Index(nr::Z4cComponent::betax));
    result.shift[1] = z4c_value(nr::Index(nr::Z4cComponent::betay));
    result.shift[2] = z4c_value(nr::Index(nr::Z4cComponent::betaz));
    result.spatial_det = det;
    result.gdet = result.lapse * sqrt(det);

    parthenon::Real lowered_shift[3]{};
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second)
        lowered_shift[first] += spatial[first][second] * result.shift[second];
    }
    result.lower[0][0] = -result.lapse * result.lapse;
    for (int axis = 0; axis < 3; ++axis)
      result.lower[0][0] += lowered_shift[axis] * result.shift[axis];
    for (int axis = 0; axis < 3; ++axis) {
      result.lower[0][axis + 1] = lowered_shift[axis];
      result.lower[axis + 1][0] = lowered_shift[axis];
      for (int second = 0; second < 3; ++second)
        result.lower[axis + 1][second + 1] = spatial[axis][second];
    }

    const parthenon::Real inverse_lapse2 = 1.0 / (result.lapse * result.lapse);
    result.upper[0][0] = -inverse_lapse2;
    for (int axis = 0; axis < 3; ++axis) {
      result.upper[0][axis + 1] = result.shift[axis] * inverse_lapse2;
      result.upper[axis + 1][0] = result.upper[0][axis + 1];
      for (int second = 0; second < 3; ++second) {
        result.upper[axis + 1][second + 1] =
            inverse[axis][second] - result.shift[axis] * result.shift[second] * inverse_lapse2;
      }
    }
    return result;
  }

  // Fixed-background call sites do not carry a stage pack. They remain
  // compilable in a sync-only binary, but reaching one is a hard error rather
  // than silently returning Minkowski geometry.
  KOKKOS_INLINE_FUNCTION MetricPoint MetricAt(Location, parthenon::Real, parthenon::Real,
                                              parthenon::Real) const {
    Kokkos::abort("sync Geometry requires current Z4c and ADM field packs");
    return {};
  }

  template <class Coordinates>
  KOKKOS_INLINE_FUNCTION MetricPoint MetricAt(Location, int, int, int, int,
                                              const Coordinates&) const {
    Kokkos::abort("sync Geometry requires current Z4c and ADM field packs");
    return {};
  }

  KOKKOS_INLINE_FUNCTION MetricDerivatives DerivativesAt(parthenon::Real, parthenon::Real,
                                                         parthenon::Real) const {
    Kokkos::abort("sync Geometry derivatives require evolved ADM fields");
    return {};
  }

  template <class Coordinates>
  KOKKOS_INLINE_FUNCTION MetricDerivatives DerivativesAt(int, int, int, int,
                                                         const Coordinates&) const {
    Kokkos::abort("sync Geometry derivatives require evolved ADM fields");
    return {};
  }

  KOKKOS_INLINE_FUNCTION const Parameters& GetParameters() const { return parameters_; }

  KOKKOS_INLINE_FUNCTION SphericalKerrSchildPoint SphericalCoordinates(
      const parthenon::Real x1, const parthenon::Real x2, const parthenon::Real x3) const {
    return MetricType::SphericalCoordinates(x1, x2, x3, parameters_);
  }

  KOKKOS_INLINE_FUNCTION void BoyerLindquistSpatialToNative(const parthenon::Real x1,
                                                            const parthenon::Real x2,
                                                            const parthenon::Real x3,
                                                            const parthenon::Real source[3],
                                                            parthenon::Real native[3]) const {
    MetricType::BoyerLindquistSpatialToNative(x1, x2, x3, source, native, parameters_);
  }

  KOKKOS_INLINE_FUNCTION void AzimuthalCovectorToNative(const parthenon::Real x1,
                                                        const parthenon::Real x2,
                                                        const parthenon::Real x3,
                                                        const parthenon::Real azimuthal,
                                                        parthenon::Real native[3]) const {
    MetricType::AzimuthalCovectorToNative(x1, x2, x3, azimuthal, native, parameters_);
  }

  KOKKOS_INLINE_FUNCTION bool IsExcised(parthenon::Real, parthenon::Real, parthenon::Real,
                                        parthenon::Real) const {
    return false;
  }

  template <class Coordinates>
  KOKKOS_INLINE_FUNCTION bool NeedsFluxExcision(const Coordinates&, int, int, int,
                                                parthenon::Real) const {
    return false;
  }

  KOKKOS_INLINE_FUNCTION bool IsInitialized() const { return true; }
  constexpr std::size_t StorageBytes() const { return 0; }
  constexpr std::uint64_t Fingerprint() const { return 0; }

private:
  Parameters parameters_;
};

static_assert(std::is_trivially_copyable_v<SyncMode>);

template <class MetricType>
inline void InitializeGeometryMode(parthenon::MeshBlock*, parthenon::ParameterInput*,
                                   Geometry<MetricType, SyncMode>&) {}

template <class MetricType>
inline void ValidateGeometryModeInput(parthenon::ParameterInput*, SyncMode) {}

} // namespace pangu::geometry

#endif
