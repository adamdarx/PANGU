#ifndef PANGU_GEOMETRY_MODE_STATIC_H_
#define PANGU_GEOMETRY_MODE_STATIC_H_

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "geometry/geometry.h"
#include "utils/error_checking.hpp"

namespace pangu::geometry {

struct StaticMode {
  static constexpr bool supports_refinement = false;
  static constexpr bool synchronized = false;
};

template <class MetricType> class Geometry<MetricType, StaticMode> {
public:
  using Parameters = typename MetricType::Parameters;
  static constexpr bool supports_excision = MetricType::supports_excision;
  static constexpr bool unit_determinant = MetricType::unit_determinant;
  static constexpr bool unit_coordinate_light_bound = MetricType::unit_coordinate_light_bound;

  explicit Geometry(const Parameters& parameters) : parameters_(parameters) {}

  void Allocate(const int blocks, const parthenon::IndexRange bounds_k,
                const parthenon::IndexRange bounds_j, const parthenon::IndexRange bounds_i) {
    blocks_ = blocks;
    start_k_ = bounds_k.s;
    start_j_ = bounds_j.s;
    start_i_ = bounds_i.s;
    cells_k_ = bounds_k.e - bounds_k.s + 1;
    cells_j_ = bounds_j.e - bounds_j.s + 1;
    cells_i_ = bounds_i.e - bounds_i.s + 1;

    metric_k_ = MetricType::symmetry == Symmetry::general_3d ? cells_k_ + 1 : 1;
    metric_j_ = (MetricType::symmetry == Symmetry::axisymmetric_2d ||
                 MetricType::symmetry == Symmetry::general_3d)
                    ? cells_j_ + 1
                    : 1;
    metric_i_ = MetricType::symmetry == Symmetry::constant ? 1 : cells_i_ + 1;
    derivative_k_ = MetricType::symmetry == Symmetry::general_3d ? cells_k_ : 1;
    derivative_j_ = (MetricType::symmetry == Symmetry::axisymmetric_2d ||
                     MetricType::symmetry == Symmetry::general_3d)
                        ? cells_j_
                        : 1;
    derivative_i_ = MetricType::symmetry == Symmetry::constant ? 1 : cells_i_;
    derivative_axes_ = static_cast<int>(MetricType::symmetry);

    gcov_ = parthenon::ParArray6DRaw<parthenon::Real>("PANGU static geometry gcov", 4, blocks_,
                                                      metric_k_, metric_j_, metric_i_, 10);
    gcon_ = parthenon::ParArray6DRaw<parthenon::Real>("PANGU static geometry gcon", 4, blocks_,
                                                      metric_k_, metric_j_, metric_i_, 10);
    gdet_ = parthenon::ParArray5DRaw<parthenon::Real>("PANGU static geometry gdet", 4, blocks_,
                                                      metric_k_, metric_j_, metric_i_);
    dgcov_ = parthenon::ParArray6DRaw<parthenon::Real>("PANGU static geometry dgcov", blocks_,
                                                       derivative_k_, derivative_j_, derivative_i_,
                                                       derivative_axes_, 10);
  }

  template <class Coordinates>
  void InitializeBlock(const int block, const Coordinates& coordinates) {
    for (int location_index = 0; location_index < 4; ++location_index) {
      const auto location = static_cast<Location>(location_index);
      const auto geometry = *this;
      Kokkos::parallel_for(
          "PANGU initialize static metric",
          Kokkos::MDRangePolicy<parthenon::DevExecSpace, Kokkos::Rank<3>>(
              {0, 0, 0}, {metric_k_, metric_j_, metric_i_}),
          KOKKOS_LAMBDA(const int stored_k, const int stored_j, const int stored_i) {
            const int k = geometry.start_k_ + stored_k;
            const int j = geometry.start_j_ + stored_j;
            const int i = geometry.start_i_ + stored_i;
            const parthenon::Real x1 = location == Location::face1 ? coordinates.template Xf<1>(i)
                                                                   : coordinates.template Xc<1>(i);
            const parthenon::Real x2 = location == Location::face2 ? coordinates.template Xf<2>(j)
                                                                   : coordinates.template Xc<2>(j);
            const parthenon::Real x3 = location == Location::face3 ? coordinates.template Xf<3>(k)
                                                                   : coordinates.template Xc<3>(k);
            geometry.StoreMetric(location_index, block, stored_k, stored_j, stored_i,
                                 MetricType::Calculate(x1, x2, x3, location, geometry.parameters_));
          });
    }

    const auto geometry = *this;
    Kokkos::parallel_for(
        "PANGU initialize static metric derivatives",
        Kokkos::MDRangePolicy<parthenon::DevExecSpace, Kokkos::Rank<3>>(
            {0, 0, 0}, {derivative_k_, derivative_j_, derivative_i_}),
        KOKKOS_LAMBDA(const int stored_k, const int stored_j, const int stored_i) {
          const int k = geometry.start_k_ + stored_k;
          const int j = geometry.start_j_ + stored_j;
          const int i = geometry.start_i_ + stored_i;
          geometry.StoreDerivatives(block, stored_k, stored_j, stored_i,
                                    MetricType::CalculateDerivatives(coordinates.template Xc<1>(i),
                                                                     coordinates.template Xc<2>(j),
                                                                     coordinates.template Xc<3>(k),
                                                                     geometry.parameters_));
        });
  }

  template <class Coordinates>
  KOKKOS_INLINE_FUNCTION MetricPoint MetricAt(const Location location, const int block, const int k,
                                              const int j, const int i, const Coordinates&) const {
    MetricPoint metric{};
    const int stored_k = StoredK(k);
    const int stored_j = StoredJ(j);
    const int stored_i = StoredI(i);
    const int location_index = static_cast<int>(location);
    for (int first = 0; first < 4; ++first) {
      for (int second = first; second < 4; ++second) {
        const int component = SymmetricComponent(first, second);
        const parthenon::Real lower =
            gcov_(location_index, block, stored_k, stored_j, stored_i, component);
        const parthenon::Real upper =
            gcon_(location_index, block, stored_k, stored_j, stored_i, component);
        metric.lower[first][second] = lower;
        metric.lower[second][first] = lower;
        metric.upper[first][second] = upper;
        metric.upper[second][first] = upper;
      }
    }
    metric.lapse = sqrt(-1.0 / metric.upper[0][0]);
    for (int axis = 0; axis < 3; ++axis)
      metric.shift[axis] = metric.lapse * metric.lapse * metric.upper[0][axis + 1];
    metric.gdet = gdet_(location_index, block, stored_k, stored_j, stored_i);
    const parthenon::Real determinant_ratio = metric.gdet / metric.lapse;
    metric.spatial_det = determinant_ratio * determinant_ratio;
    return metric;
  }

  // Keep the Geometry access signature uniform across compile-time modes.
  // Stage packs are meaningful only for SyncMode; fixed static geometry has
  // already cached the requested metric and therefore ignores them.
  template <class Coordinates, class Z4cPack, class ADMPack>
  KOKKOS_INLINE_FUNCTION MetricPoint MetricAt(const Location location, const int block, const int k,
                                              const int j, const int i,
                                              const Coordinates& coordinates, const Z4cPack&,
                                              const ADMPack&) const {
    return MetricAt(location, block, k, j, i, coordinates);
  }

  template <class Coordinates>
  KOKKOS_INLINE_FUNCTION MetricDerivatives DerivativesAt(const int block, const int k, const int j,
                                                         const int i, const Coordinates&) const {
    MetricDerivatives derivatives{};
    const int stored_k = StoredDerivativeK(k);
    const int stored_j = StoredDerivativeJ(j);
    const int stored_i = StoredDerivativeI(i);
    for (int axis = 0; axis < derivative_axes_; ++axis) {
      for (int first = 0; first < 4; ++first) {
        for (int second = first; second < 4; ++second) {
          const parthenon::Real value =
              dgcov_(block, stored_k, stored_j, stored_i, axis, SymmetricComponent(first, second));
          derivatives.lower[axis][first][second] = value;
          derivatives.lower[axis][second][first] = value;
        }
      }
    }
    return derivatives;
  }

  void Validate() const {
    const auto gcov_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gcov_);
    const auto gcon_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gcon_);
    const auto gdet_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gdet_);
    const auto dgcov_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dgcov_);
    for (std::size_t index = 0; index < gcov_host.size(); ++index)
      PARTHENON_REQUIRE(std::isfinite(gcov_host.data()[index]),
                        "static Geometry contains a non-finite covariant component");
    for (std::size_t index = 0; index < gcon_host.size(); ++index)
      PARTHENON_REQUIRE(std::isfinite(gcon_host.data()[index]),
                        "static Geometry contains a non-finite contravariant component");
    for (std::size_t index = 0; index < gdet_host.size(); ++index)
      PARTHENON_REQUIRE(std::isfinite(gdet_host.data()[index]) && gdet_host.data()[index] > 0.0,
                        "static Geometry contains an invalid metric determinant");
    for (std::size_t index = 0; index < dgcov_host.size(); ++index)
      PARTHENON_REQUIRE(std::isfinite(dgcov_host.data()[index]),
                        "static Geometry contains a non-finite metric derivative");
  }

  std::uint64_t Fingerprint() const {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto add = [&hash](const parthenon::Real value) {
      hash ^= std::bit_cast<std::uint64_t>(value);
      hash *= 1099511628211ULL;
    };
    const auto gcov_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gcov_);
    const auto gcon_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gcon_);
    const auto gdet_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gdet_);
    const auto dgcov_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dgcov_);
    for (std::size_t index = 0; index < gcov_host.size(); ++index)
      add(gcov_host.data()[index]);
    for (std::size_t index = 0; index < gcon_host.size(); ++index)
      add(gcon_host.data()[index]);
    for (std::size_t index = 0; index < gdet_host.size(); ++index)
      add(gdet_host.data()[index]);
    for (std::size_t index = 0; index < dgcov_host.size(); ++index)
      add(dgcov_host.data()[index]);
    return hash;
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
  KOKKOS_INLINE_FUNCTION bool IsInitialized() const { return gcov_.data() != nullptr; }

  std::size_t StorageBytes() const {
    return sizeof(parthenon::Real) * (gcov_.size() + gcon_.size() + gdet_.size() + dgcov_.size());
  }

private:
  KOKKOS_INLINE_FUNCTION int StoredK(const int k) const {
    if constexpr (MetricType::symmetry == Symmetry::general_3d)
      return k - start_k_;
    return 0;
  }

  KOKKOS_INLINE_FUNCTION int StoredJ(const int j) const {
    if constexpr (MetricType::symmetry == Symmetry::axisymmetric_2d ||
                  MetricType::symmetry == Symmetry::general_3d)
      return j - start_j_;
    return 0;
  }

  KOKKOS_INLINE_FUNCTION int StoredI(const int i) const {
    if constexpr (MetricType::symmetry != Symmetry::constant)
      return i - start_i_;
    return 0;
  }

  KOKKOS_INLINE_FUNCTION int StoredDerivativeK(const int k) const { return StoredK(k); }
  KOKKOS_INLINE_FUNCTION int StoredDerivativeJ(const int j) const { return StoredJ(j); }
  KOKKOS_INLINE_FUNCTION int StoredDerivativeI(const int i) const { return StoredI(i); }

  KOKKOS_INLINE_FUNCTION void StoreMetric(const int location, const int block, const int k,
                                          const int j, const int i,
                                          const MetricPoint& metric) const {
    for (int first = 0; first < 4; ++first) {
      for (int second = first; second < 4; ++second) {
        const int component = SymmetricComponent(first, second);
        gcov_(location, block, k, j, i, component) = metric.lower[first][second];
        gcon_(location, block, k, j, i, component) = metric.upper[first][second];
      }
    }
    gdet_(location, block, k, j, i) = metric.gdet;
  }

  KOKKOS_INLINE_FUNCTION void StoreDerivatives(const int block, const int k, const int j,
                                               const int i,
                                               const MetricDerivatives& derivatives) const {
    for (int axis = 0; axis < derivative_axes_; ++axis)
      for (int first = 0; first < 4; ++first)
        for (int second = first; second < 4; ++second)
          dgcov_(block, k, j, i, axis, SymmetricComponent(first, second)) =
              derivatives.lower[axis][first][second];
  }

  Parameters parameters_;
  parthenon::ParArray6DRaw<parthenon::Real> gcov_;
  parthenon::ParArray6DRaw<parthenon::Real> gcon_;
  parthenon::ParArray5DRaw<parthenon::Real> gdet_;
  parthenon::ParArray6DRaw<parthenon::Real> dgcov_;
  int blocks_ = 0;
  int start_k_ = 0;
  int start_j_ = 0;
  int start_i_ = 0;
  int cells_k_ = 0;
  int cells_j_ = 0;
  int cells_i_ = 0;
  int metric_k_ = 0;
  int metric_j_ = 0;
  int metric_i_ = 0;
  int derivative_k_ = 0;
  int derivative_j_ = 0;
  int derivative_i_ = 0;
  int derivative_axes_ = 0;
};

static_assert(std::is_trivially_copyable_v<StaticMode>);

template <class MetricType>
inline void InitializeGeometryMode(parthenon::MeshBlock* block, parthenon::ParameterInput*,
                                   Geometry<MetricType, StaticMode>& geometry) {
  if (geometry.IsInitialized())
    return;
  const auto bounds_i = block->cellbounds.GetBoundsI(parthenon::IndexDomain::entire);
  const auto bounds_j = block->cellbounds.GetBoundsJ(parthenon::IndexDomain::entire);
  const auto bounds_k = block->cellbounds.GetBoundsK(parthenon::IndexDomain::entire);
  geometry.Allocate(static_cast<int>(block->pmy_mesh->block_list.size()), bounds_k, bounds_j,
                    bounds_i);
  for (const auto& mesh_block : block->pmy_mesh->block_list)
    geometry.InitializeBlock(mesh_block->lid, mesh_block->coords);
  Kokkos::fence("PANGU static Geometry initialization");
  geometry.Validate();
}

template <class MetricType>
inline void ValidateGeometryModeInput(parthenon::ParameterInput* pin, StaticMode) {
  const std::string refinement = pin->GetOrAddString("parthenon/mesh", "refinement", "none");
  PARTHENON_REQUIRE(refinement == "none", "MODE=static requires <parthenon/mesh>/refinement=none");
  PARTHENON_REQUIRE(pin->GetOrAddInteger("parthenon/mesh", "numlevel", 1) == 1,
                    "MODE=static requires <parthenon/mesh>/numlevel=1");
  PARTHENON_REQUIRE(pin->GetBlockNamesWithPrefix("parthenon/static_refinement").empty(),
                    "MODE=static rejects every <parthenon/static_refinement*> block");
  PARTHENON_REQUIRE(!pin->GetOrAddBoolean("parthenon/mesh", "multigrid", false),
                    "MODE=static does not support a multigrid hierarchy");
}

} // namespace pangu::geometry

#endif
