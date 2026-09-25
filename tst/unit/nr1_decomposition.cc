#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <parthenon/parthenon.hpp>

#include "z4c/core/finite_difference.h"

namespace {

using Real = parthenon::Real;

struct Accessor {
  Kokkos::View<Real****> values;
  int block;
  int k;
  int j;
  int i;

  KOKKOS_INLINE_FUNCTION Real operator()(const int dk, const int dj, const int di) const {
    return values(block, k + dk, j + dj, i + di);
  }
};

} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int return_code = 0;
  {
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 1;
    constexpr int kGlobalCells = 64;
    constexpr int kGhost = pangu::nr::fd::Stencil<6>::transport_radius;
    const int decomposition = ranks * blocks;
    if (blocks <= 0 || kGlobalCells % decomposition != 0) {
      if (rank == 0)
        std::cerr << "NR-1 decomposition must divide 64 cells exactly\n";
      return_code = 1;
    } else {
      const int local_cells = kGlobalCells / decomposition;
      const int extent_x = local_cells + 2 * kGhost;
      const int extent_yz = kGlobalCells + 2 * kGhost;
      const Real spacing = 2.0 * std::numbers::pi_v<Real> / kGlobalCells;
      const Real inverse_spacing = 1.0 / spacing;
      Kokkos::View<Real****> field("NR-1 decomposed field", blocks, extent_yz, extent_yz, extent_x);
      Kokkos::parallel_for(
          "NR-1 initialize decomposed field", blocks * extent_yz * extent_yz * extent_x,
          KOKKOS_LAMBDA(const int index) {
            int remainder = index;
            const int i = remainder % extent_x;
            remainder /= extent_x;
            const int j = remainder % extent_yz;
            remainder /= extent_yz;
            const int k = remainder % extent_yz;
            const int block = remainder / extent_yz;
            const int global_i = (rank * blocks + block) * local_cells + i - kGhost;
            const int global_j = j - kGhost;
            const int global_k = k - kGhost;
            field(block, k, j, i) = sin((global_i + 2.0 * global_j + 3.0 * global_k) * spacing);
          });

      Real local_maximum = 0.0;
      Real local_boundary_maximum = 0.0;
      const int interior_cells = blocks * kGlobalCells * kGlobalCells * local_cells;
      Kokkos::parallel_reduce(
          "NR-1 decomposed operator maximum", interior_cells,
          KOKKOS_LAMBDA(const int index, Real& maximum) {
            int remainder = index;
            const int local_i = remainder % local_cells;
            remainder /= local_cells;
            const int local_j = remainder % kGlobalCells;
            remainder /= kGlobalCells;
            const int local_k = remainder % kGlobalCells;
            const int block = remainder / kGlobalCells;
            const int global_i = (rank * blocks + block) * local_cells + local_i;
            const Real phase = (global_i + 2.0 * local_j + 3.0 * local_k) * spacing;
            const Accessor value{field, block, local_k + kGhost, local_j + kGhost,
                                 local_i + kGhost};
            const Real wave_number[3]{1.0, 2.0, 3.0};
            for (int direction = 0; direction < 3; ++direction) {
              const Real error = fabs(pangu::nr::fd::First<6>(direction, inverse_spacing, value) -
                                      wave_number[direction] * cos(phase));
              maximum = fmax(maximum, error);
            }
          },
          Kokkos::Max<Real>(local_maximum));
      Kokkos::parallel_reduce(
          "NR-1 block-boundary operator maximum", interior_cells,
          KOKKOS_LAMBDA(const int index, Real& maximum) {
            int remainder = index;
            const int local_i = remainder % local_cells;
            remainder /= local_cells;
            const int local_j = remainder % kGlobalCells;
            remainder /= kGlobalCells;
            const int local_k = remainder % kGlobalCells;
            const int block = remainder / kGlobalCells;
            if (local_i >= kGhost && local_i < local_cells - kGhost)
              return;
            const int global_i = (rank * blocks + block) * local_cells + local_i;
            const Real phase = (global_i + 2.0 * local_j + 3.0 * local_k) * spacing;
            const Accessor value{field, block, local_k + kGhost, local_j + kGhost,
                                 local_i + kGhost};
            const Real error =
                fabs(pangu::nr::fd::First<6>(0, inverse_spacing, value) - cos(phase));
            maximum = fmax(maximum, error);
          },
          Kokkos::Max<Real>(local_boundary_maximum));
      Kokkos::fence();

      Real global_maximum = 0.0;
      Real global_boundary_maximum = 0.0;
      MPI_Allreduce(&local_maximum, &global_maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&local_boundary_maximum, &global_boundary_maximum, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      constexpr Real kTolerance = 2.0e-5;
      if (rank == 0) {
        std::cout << "NR-1 decomposition: ranks=" << ranks << " blocks_per_rank=" << blocks
                  << " maximum=" << global_maximum
                  << " boundary_maximum=" << global_boundary_maximum << '\n';
      }
      if (!std::isfinite(global_maximum) || global_maximum > kTolerance ||
          global_boundary_maximum > kTolerance)
        return_code = 1;
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return return_code;
}
