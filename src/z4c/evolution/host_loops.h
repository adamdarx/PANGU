#ifndef PANGU_Z4C_EVOLUTION_HOST_LOOPS_H_
#define PANGU_Z4C_EVOLUTION_HOST_LOOPS_H_

#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "kokkos_abstraction.hpp"

#ifndef PANGU_ENABLE_HOST_SIMD_THREADS
#define PANGU_ENABLE_HOST_SIMD_THREADS 0
#endif

#if PANGU_ENABLE_HOST_SIMD_THREADS
#if !defined(KOKKOS_ENABLE_OPENMP) || defined(PARTHENON_ENABLE_GPU)
#error "Threaded host SIMD requires a CPU Kokkos OpenMP build"
#endif
#include <omp.h>
#endif

namespace pangu::nr::host_loops {

#if PANGU_ENABLE_HOST_SIMD_THREADS
namespace detail {
template <class Tuple, std::size_t... Axes>
inline void ThreadedSimd(const Tuple& arguments, std::index_sequence<Axes...>) {
  constexpr std::size_t rank = sizeof...(Axes) + 1;
  // Bounds remain inclusive, including nonzero/negative starts. Empty ranges
  // must return before MakeIndexer can form an invalid unsigned extent.
  const bool empty = ((std::get<2 * Axes + 1>(arguments) <
                      std::get<2 * Axes>(arguments)) || ...) ||
                     std::get<2 * rank - 1>(arguments) <
                         std::get<2 * rank - 2>(arguments);
  if (empty) return;
  const auto rows = parthenon::MakeIndexer(
      std::pair<int, int>{std::get<2 * Axes>(arguments),
                          std::get<2 * Axes + 1>(arguments)}...);
  PARTHENON_REQUIRE(rows.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                    "NR threaded SIMD row count exceeds indexer capacity");
  const int count = static_cast<int>(rows.size());
  const int first = std::get<2 * rank - 2>(arguments);
  const int last = std::get<2 * rank - 1>(arguments);
  const auto& function = std::get<2 * rank>(arguments);
  // One region per complete MeshData kernel, not per block or per SIMD row.
  // A shared const closure avoids managed View reference-count traffic per row.
  // The implicit barrier preserves the original synchronous host-loop contract.
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1 && !omp_in_parallel())
  for (int row = 0; row < count; ++row) {
    const auto indices = rows.GetIdxArray(row);
#pragma omp simd
    for (int i = first; i <= last; ++i) {
      function(indices[Axes]..., i);
    }
  }
}
} // namespace detail
#endif

// Only opt-in CPU SIMD loops change scheduling. GPU and other Kokkos patterns
// forward unchanged to Parthenon's native dispatcher. No MPI call is threaded.
template <class Pattern, class ExecSpace, class... Args>
inline void ParallelFor(Pattern pattern, const std::string& name, ExecSpace space,
                        Args&&... arguments) {
#if PANGU_ENABLE_HOST_SIMD_THREADS
  if constexpr (std::is_same_v<Pattern, parthenon::LoopPatternSimdFor> &&
                std::is_same_v<ExecSpace, parthenon::HostExecSpace>) {
    static_assert(sizeof...(Args) >= 5 && sizeof...(Args) % 2 == 1,
                  "NR host loops require inclusive bound pairs and a cell functor");
    constexpr std::size_t rank = (sizeof...(Args) - 1) / 2;
    detail::ThreadedSimd(std::forward_as_tuple(arguments...),
                         std::make_index_sequence<rank - 1>{});
  } else
#endif
  {
    parthenon::par_for(pattern, name, space, std::forward<Args>(arguments)...);
  }
}

} // namespace pangu::nr::host_loops
#endif
