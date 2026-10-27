#ifndef PANGU_Z4C_AMR_REFINEMENT_H_
#define PANGU_Z4C_AMR_REFINEMENT_H_

#include <parthenon/parthenon.hpp>

namespace pangu::nr::amr {

// AthenaK's nghost=4 Z4c interpolation weights.  Unlike hydrodynamic
// conserved variables, the differentiable geometric fields must not be
// prolonged with a slope limiter: doing so inserts first-derivative kinks
// exactly where the Z4c RHS subsequently takes high-order derivatives.
struct ProlongateZ4cHighOrder {
  static constexpr bool
  OperationRequired(parthenon::TopologicalElement fine,
                    parthenon::TopologicalElement coarse) {
    return fine == parthenon::TopologicalElement::CC &&
           coarse == parthenon::TopologicalElement::CC;
  }

  template <int DIM,
            parthenon::TopologicalElement fine_element =
                parthenon::TopologicalElement::CC,
            parthenon::TopologicalElement coarse_element =
                parthenon::TopologicalElement::CC,
            int StencilGhost = 0>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j,
     const int i, const parthenon::IndexRange &ckb,
     const parthenon::IndexRange &cjb, const parthenon::IndexRange &cib,
     const parthenon::IndexRange &kb, const parthenon::IndexRange &jb,
     const parthenon::IndexRange &ib, const parthenon::Coordinates_t &coords,
     const parthenon::Coordinates_t &coarse_coords,
     const parthenon::ParArrayND<parthenon::Real, parthenon::VariableState>
         *coarse_ptr,
     const parthenon::ParArrayND<parthenon::Real, parthenon::VariableState>
         *fine_ptr) {
    static_assert(fine_element == parthenon::TopologicalElement::CC);
    if constexpr (StencilGhost == 0) {
      // Dispatch once per point to constant-size stencils. Preserve the
      // original arithmetic order and retain the legacy fallback.
      if (ib.s == 2)
        Do<DIM, fine_element, coarse_element, 2>(
            l, m, n, k, j, i, ckb, cjb, cib, kb, jb, ib, coords, coarse_coords,
            coarse_ptr, fine_ptr);
      else if (ib.s == 4)
        Do<DIM, fine_element, coarse_element, 4>(
            l, m, n, k, j, i, ckb, cjb, cib, kb, jb, ib, coords, coarse_coords,
            coarse_ptr, fine_ptr);
      else
        Do<DIM, fine_element, coarse_element, -1>(
            l, m, n, k, j, i, ckb, cjb, cib, kb, jb, ib, coords, coarse_coords,
            coarse_ptr, fine_ptr);
    } else {
      const auto &coarse = *coarse_ptr;
      const auto &fine = *fine_ptr;
      constexpr int element =
          static_cast<int>(parthenon::TopologicalElement::CC) % 3;
      const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
      const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
      const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;
      const int ghost = StencilGhost > 0 ? StencilGhost : ib.s;
      const int points = ghost == 2 ? 3 : 5;
      const int radius = ghost / 2;
      parthenon::Real weights[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
      if (ghost == 2) {
        weights[0] = 0.15625;
        weights[1] = 0.9375;
        weights[2] = -0.09375;
      } else {
        weights[0] = -0.02197265625;
        weights[1] = 0.205078125;
        weights[2] = 0.9228515625;
        weights[3] = -0.123046875;
        weights[4] = 0.01708984375;
      }

      // Parthenon restricts same-level fine ghosts only where a corresponding
      // fine cell exists. The allocation has additional coarse padding, but
      // its edge/corner combinations need not have been communicated. A
      // centered tensor stencil there reads unfilled values, even for a
      // constant field. Keep the stencil inside the coarse image of the fine
      // allocation, using one-sided Lagrange weights near its outer faces.
      // Interior interpolation retains the original centered weights.
      const parthenon::IndexRange coarse_bounds[3] = {cib, cjb, ckb};
      const int parent[3] = {i, j, k};
      int first[3] = {i - radius, j - radius, k - radius};
      parthenon::Real axis_weights[3][2][5]{};
      for (int axis = 0; axis < DIM; ++axis) {
        if (ghost == 2 || ghost == 4) {
          const int lower = coarse_bounds[axis].s - ghost / 2;
          const int upper_start = coarse_bounds[axis].e + ghost / 2 - points + 1;
          first[axis] = first[axis] < lower ? lower : first[axis];
          first[axis] = first[axis] > upper_start ? upper_start : first[axis];
        }
        for (int child = 0; child < 2; ++child) {
          for (int node = 0; node < points; ++node) {
            if (first[axis] == parent[axis] - radius) {
              axis_weights[axis][child][node] = weights[child == 0 ? node : ghost - node];
            } else {
              const parthenon::Real target = parent[axis] + (child == 0 ? -0.25 : 0.25);
              parthenon::Real weight = 1.0;
              for (int other = 0; other < points; ++other) {
                if (other != node)
                  weight *= (target - (first[axis] + other)) / (node - other);
              }
              axis_weights[axis][child][node] = weight;
            }
          }
        }
      }

      for (int child_k = 0; child_k < (DIM > 2 ? 2 : 1); ++child_k) {
        for (int child_j = 0; child_j < (DIM > 1 ? 2 : 1); ++child_j) {
          for (int child_i = 0; child_i < (DIM > 0 ? 2 : 1); ++child_i) {
            parthenon::Real value = 0.0;
            for (int pk = 0; pk < (DIM > 2 ? points : 1); ++pk) {
              const parthenon::Real wk =
                  DIM > 2 ? axis_weights[2][child_k][pk] : 1.0;
              for (int pj = 0; pj < (DIM > 1 ? points : 1); ++pj) {
                const parthenon::Real wj =
                    DIM > 1 ? axis_weights[1][child_j][pj] : 1.0;
                for (int pi = 0; pi < (DIM > 0 ? points : 1); ++pi) {
                  const parthenon::Real wi =
                      DIM > 0 ? axis_weights[0][child_i][pi] : 1.0;
                  value +=
                      wk * wj * wi *
                      coarse(element, l, m, n, DIM > 2 ? first[2] + pk : k,
                             DIM > 1 ? first[1] + pj : j,
                             DIM > 0 ? first[0] + pi : i);
                }
              }
            }
            fine(element, l, m, n, fk + child_k, fj + child_j, fi + child_i) =
                value;
          }
        }
      }
    }
  }
};

struct RestrictZ4cHighOrder {
  static constexpr bool
  OperationRequired(parthenon::TopologicalElement fine,
                    parthenon::TopologicalElement coarse) {
    return fine == parthenon::TopologicalElement::CC &&
           coarse == parthenon::TopologicalElement::CC;
  }

  template <int DIM,
            parthenon::TopologicalElement fine_element =
                parthenon::TopologicalElement::CC,
            parthenon::TopologicalElement coarse_element =
                parthenon::TopologicalElement::CC,
            int StencilGhost = 0>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int ck, const int cj,
     const int ci, const parthenon::IndexRange &ckb,
     const parthenon::IndexRange &cjb, const parthenon::IndexRange &cib,
     const parthenon::IndexRange &kb, const parthenon::IndexRange &jb,
     const parthenon::IndexRange &ib, const parthenon::Coordinates_t &coords,
     const parthenon::Coordinates_t &coarse_coords,
     const parthenon::ParArrayND<parthenon::Real, parthenon::VariableState>
         *coarse_ptr,
     const parthenon::ParArrayND<parthenon::Real, parthenon::VariableState>
         *fine_ptr) {
    static_assert(fine_element == parthenon::TopologicalElement::CC);
    if constexpr (StencilGhost == 0) {
      if (ib.s == 2)
        Do<DIM, fine_element, coarse_element, 2>(
            l, m, n, ck, cj, ci, ckb, cjb, cib, kb, jb, ib, coords,
            coarse_coords, coarse_ptr, fine_ptr);
      else if (ib.s == 4)
        Do<DIM, fine_element, coarse_element, 4>(
            l, m, n, ck, cj, ci, ckb, cjb, cib, kb, jb, ib, coords,
            coarse_coords, coarse_ptr, fine_ptr);
      else
        Do<DIM, fine_element, coarse_element, -1>(
            l, m, n, ck, cj, ci, ckb, cjb, cib, kb, jb, ib, coords,
            coarse_coords, coarse_ptr, fine_ptr);
    } else {
      auto &coarse = *coarse_ptr;
      const auto &fine = *fine_ptr;
      constexpr int element =
          static_cast<int>(parthenon::TopologicalElement::CC) % 3;
      const int fi = (DIM > 0) ? (ci - cib.s) * 2 + ib.s : ib.s;
      const int fj = (DIM > 1) ? (cj - cjb.s) * 2 + jb.s : jb.s;
      const int fk = (DIM > 2) ? (ck - ckb.s) * 2 + kb.s : kb.s;
      const int ghost = StencilGhost > 0 ? StencilGhost : ib.s;
      const int points = ghost == 2 ? 3 : 5;
      parthenon::Real weights[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
      parthenon::Real edge_weights[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
      if (ghost == 2) {
        weights[0] = 0.375;
        weights[1] = 0.75;
        weights[2] = -0.125;
      } else {
        weights[0] = -0.0390625;
        weights[1] = 0.46875;
        weights[2] = 0.703125;
        weights[3] = -0.15625;
        weights[4] = 0.0234375;
        edge_weights[0] = 0.2734375;
        edge_weights[1] = 1.09375;
        edge_weights[2] = -0.546875;
        edge_weights[3] = 0.21875;
        edge_weights[4] = -0.0390625;
      }
      const bool lower_i = fi < ib.s + (ib.e - ib.s + 1) / 2;
      const bool lower_j = fj < jb.s + (jb.e - jb.s + 1) / 2;
      const bool lower_k = fk < kb.s + (kb.e - kb.s + 1) / 2;
      const bool edge_i = DIM > 0 && (fi == ib.s || fi == ib.e - 1);
      const bool edge_j = DIM > 1 && (fj == jb.s || fj == jb.e - 1);
      const bool edge_k = DIM > 2 && (fk == kb.s || fk == kb.e - 1);
      int ref_i =
          fi -
          (DIM > 0 ? (ghost == 2 ? (lower_i ? 0 : 1) : (lower_i ? 1 : 2)) : 0);
      int ref_j =
          fj -
          (DIM > 1 ? (ghost == 2 ? (lower_j ? 0 : 1) : (lower_j ? 1 : 2)) : 0);
      int ref_k =
          fk -
          (DIM > 2 ? (ghost == 2 ? (lower_k ? 0 : 1) : (lower_k ? 1 : 2)) : 0);
      if constexpr (DIM > 0) {
        if (ghost == 4) {
          if (fi == ib.s)
            ++ref_i;
          if (fi == ib.e - 1)
            --ref_i;
        }
      }
      if constexpr (DIM > 1) {
        if (ghost == 4) {
          if (fj == jb.s)
            ++ref_j;
          if (fj == jb.e - 1)
            --ref_j;
        }
      }
      if constexpr (DIM > 2) {
        if (ghost == 4) {
          if (fk == kb.s)
            ++ref_k;
          if (fk == kb.e - 1)
            --ref_k;
        }
      }

      // Parthenon's restriction index space includes the outer half of the
      // coarse ghost region. At its outermost points a centered five-point
      // stencil leaves the fine allocation. Shift that stencil one-sided,
      // instead of averaging two cells: geometric fields are point values,
      // and averaging introduces a second-order error into a high-order halo.
      const bool high_i =
          DIM == 0 || (ref_i >= 0 && ref_i + points <= fine.GetDim(1));
      const bool high_j =
          DIM <= 1 || (ref_j >= 0 && ref_j + points <= fine.GetDim(2));
      const bool high_k =
          DIM <= 2 || (ref_k >= 0 && ref_k + points <= fine.GetDim(3));
      if (!(high_i && high_j && high_k) && ghost != 4) {
        parthenon::refinement_ops::RestrictAverage::template Do<
            DIM, fine_element, parthenon::TopologicalElement::CC>(
            l, m, n, ck, cj, ci, ckb, cjb, cib, kb, jb, ib, coords,
            coarse_coords, coarse_ptr, fine_ptr);
        return;
      }

      const bool complete[3] = {high_i, high_j, high_k};
      const int target[3] = {fi, fj, fk};
      int first[3] = {ref_i, ref_j, ref_k};
      parthenon::Real boundary_weights[3][5]{};
      for (int axis = 0; axis < DIM; ++axis) {
        if (complete[axis])
          continue;
        first[axis] = first[axis] < 0 ? 0 : first[axis];
        const int upper_start = fine.GetDim(axis + 1) - points;
        first[axis] = first[axis] > upper_start ? upper_start : first[axis];
        for (int node = 0; node < points; ++node) {
          parthenon::Real weight = 1.0;
          for (int other = 0; other < points; ++other) {
            if (other != node)
              weight *= (target[axis] + 0.5 - (first[axis] + other)) / (node - other);
          }
          boundary_weights[axis][node] = weight;
        }
      }

      parthenon::Real value = 0.0;
      for (int pk = 0; pk < (DIM > 2 ? points : 1); ++pk) {
        const int weight_k = lower_k ? pk : ghost - pk;
        const parthenon::Real wk =
            DIM > 2 ? (!high_k ? boundary_weights[2][pk]
                              : edge_k && ghost == 4 ? edge_weights[weight_k]
                                                    : weights[weight_k])
                    : 1.0;
        for (int pj = 0; pj < (DIM > 1 ? points : 1); ++pj) {
          const int weight_j = lower_j ? pj : ghost - pj;
          const parthenon::Real wj =
              DIM > 1 ? (!high_j ? boundary_weights[1][pj]
                                : edge_j && ghost == 4 ? edge_weights[weight_j]
                                                      : weights[weight_j])
                      : 1.0;
          for (int pi = 0; pi < (DIM > 0 ? points : 1); ++pi) {
            const int weight_i = lower_i ? pi : ghost - pi;
            const parthenon::Real wi =
                DIM > 0 ? (!high_i ? boundary_weights[0][pi]
                                  : edge_i && ghost == 4 ? edge_weights[weight_i]
                                                        : weights[weight_i])
                        : 1.0;
            value += wk * wj * wi *
                     fine(element, l, m, n, first[2] + pk, first[1] + pj, first[0] + pi);
          }
        }
      }
      coarse(element, l, m, n, ck, cj, ci) = value;
    }
  }
};

} // namespace pangu::nr::amr

#endif
