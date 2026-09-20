#ifndef PANGU_H_
#define PANGU_H_

#include <string>
#include <utility>

#include <parthenon/parthenon.hpp>

namespace pangu {

inline constexpr char kFrameworkTitle[] =
    "PANGU (Parthenon-based Astrophysics: Numerical Relativity and GRMHD, Unified)";

// Every PANGU reduction runs over a flat range.  Parthenon tiles MDRange reductions
// along the whole innermost index range, and Kokkos CUDA then uses that many threads
// per block regardless of the kernel: blocks wider than 512 cells are rejected, and
// heavy kernels at exactly 512 fail to launch and return the reducer identity.
template <class... Args>
inline void ParReduce(const std::string& name, Args&&... args) {
  parthenon::par_reduce(parthenon::loop_pattern_flatrange_tag, name, parthenon::DevExecSpace(),
                        std::forward<Args>(args)...);
}

} // namespace pangu

#endif
