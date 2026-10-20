#pragma once

// Minimal mesh-only adapter for compiling the reference's unmodified tree.
// No evolution, MPI, Kokkos kernels or performance measurements use this shim.
#include <cstdint>

struct LogicalLocation {
  std::int32_t lx1, lx2, lx3, level;
};
enum BoundaryFace {
  inner_x1,
  outer_x1,
  inner_x2,
  outer_x2,
  inner_x3,
  outer_x3
};
enum class BoundaryFlag { outflow, periodic, shear_periodic };
class Mesh {
public:
  bool one_d = false, two_d = false, three_d = true, multi_d = true;
  int root_level = 1, nmb_rootx1 = 2, nmb_rootx2 = 2, nmb_rootx3 = 2;
  BoundaryFlag mesh_bcs[6] = {BoundaryFlag::outflow, BoundaryFlag::outflow,
                              BoundaryFlag::outflow, BoundaryFlag::outflow,
                              BoundaryFlag::outflow, BoundaryFlag::outflow};
};
