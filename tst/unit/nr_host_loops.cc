#include <atomic>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "z4c/evolution/host_loops.h"

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  using parthenon::DevExecSpace;
  using pangu::nr::host_loops::ParallelFor;
  constexpr int size = 3 * 4 * 5 * 7;
  Kokkos::View<int*> values("coverage", size);
  ParallelFor(DEFAULT_LOOP_PATTERN, "nonzero bounds coverage", DevExecSpace(),
              -1, 1, 2, 5, -2, 2, 3, 9,
              KOKKOS_LAMBDA(int b, int k, int j, int i) {
                const int index = (((b + 1) * 4 + k - 2) * 5 + j + 2) * 7 + i - 3;
                Kokkos::atomic_fetch_add(&values(index), 1);
              });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  for (int i = 0; i < size; ++i) {
    if (host(i) != 1) return 1;
  }
#if PANGU_ENABLE_HOST_SIMD_THREADS
  std::atomic<int> calls{0};
  std::atomic<unsigned long long> workers{0};
  ParallelFor(parthenon::loop_pattern_simdfor_tag, "five-dimensional CPU coverage",
              DevExecSpace(), 0, 1, 0, 2, 0, 3, 0, 4, 0, 6,
              [&](int, int, int, int, int) {
                calls.fetch_add(1, std::memory_order_relaxed);
                workers.fetch_or(1ULL << (omp_get_thread_num() % 64),
                                 std::memory_order_relaxed);
              });
  if (calls != 2 * 3 * 4 * 5 * 7) return 2;
  if (omp_get_max_threads() > 1 && (workers.load() & (workers.load() - 1)) == 0) return 3;
  ParallelFor(parthenon::loop_pattern_simdfor_tag, "empty CPU bounds", DevExecSpace(),
              0, -1, 0, 3, [&](int, int) { ++calls; });
  if (calls != 2 * 3 * 4 * 5 * 7) return 4;
#pragma omp parallel num_threads(2)
  {
    ParallelFor(parthenon::loop_pattern_simdfor_tag, "nested caller", DevExecSpace(),
                0, 2, 0, 4, [&](int, int) { ++calls; });
  }
  if (calls != 2 * 3 * 4 * 5 * 7 + 2 * 3 * 5) return 5;
  std::cout << "Threaded SIMD coverage passed, worker mask=" << workers.load() << '\n';
#else
  std::cout << "Native loop fallback coverage passed\n";
#endif
  return 0;
}
