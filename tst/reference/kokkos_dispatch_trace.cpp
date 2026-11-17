// A Kokkos Tools observer for checking the execution backend of native kernels.
// Build with -shared -fPIC and the matching Kokkos core/src/impl include path.
// Set KOKKOS_TOOLS_LIBS to this library and PANGU_KOKKOS_TRACE to a TSV path.
// This records dispatches; it is not a kernel timing or performance tool.
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <type_traits>
#include "Kokkos_Profiling_C_Interface.h"

namespace {
std::FILE *output = nullptr;
std::mutex lock;
std::uint64_t sequence = 0;
void begin(const char *kind, const char *name, const std::uint32_t device,
           std::uint64_t *id) {
  std::lock_guard<std::mutex> guard(lock);
  *id = ++sequence;
  if (output) std::fprintf(output, "%s\t%" PRIu32 "\t%s\n", kind, device, name);
}
}

extern "C" void kokkosp_init_library(const int, const std::uint64_t version,
    const std::uint32_t count, Kokkos_Profiling_KokkosPDeviceInfo *) {
  const char *path = std::getenv("PANGU_KOKKOS_TRACE");
  output = path ? std::fopen(path, "w") : nullptr;
  if (output) std::fprintf(output, "# interface=%" PRIu64 " devices=%" PRIu32 "\nkind\tdevice_id\tkernel\n", version, count);
}
extern "C" void kokkosp_finalize_library() {
  if (output) { std::fclose(output); output = nullptr; }
}
extern "C" void kokkosp_begin_parallel_for(const char *name,
    const std::uint32_t device, std::uint64_t *id) { begin("for", name, device, id); }
extern "C" void kokkosp_begin_parallel_reduce(const char *name,
    const std::uint32_t device, std::uint64_t *id) { begin("reduce", name, device, id); }
extern "C" void kokkosp_begin_parallel_scan(const char *name,
    const std::uint32_t device, std::uint64_t *id) { begin("scan", name, device, id); }
static_assert(std::is_same_v<decltype(&kokkosp_init_library), Kokkos_Profiling_initFunction>);
static_assert(std::is_same_v<decltype(&kokkosp_begin_parallel_for), Kokkos_Profiling_beginFunction>);
