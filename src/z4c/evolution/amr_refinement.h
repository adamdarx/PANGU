#ifndef PANGU_Z4C_AMR_REFINEMENT_H_
#define PANGU_Z4C_AMR_REFINEMENT_H_

#include <parthenon/parthenon.hpp>

namespace pangu::nr::amr {

// AthenaK's nghost=4 Z4c interpolation weights.  Unlike hydrodynamic
// conserved variables, the differentiable geometric fields must not be
// prolonged with a slope limiter: doing so inserts first-derivative kinks
// exactly where the Z4c RHS subsequently takes high-order derivatives.
// Return scalar literals rather than indexing namespace-scope constexpr arrays.
// nvcc otherwise emits host-only symbols when these functions are instantiated
// inside Parthenon's device refinement kernels.
KOKKOS_INLINE_FUNCTION constexpr parthenon::Real ProlongBaseWeight(const int ghost,
                                                                  const int point) {
  if (ghost == 2)
    return point == 0 ? 0.15625 : (point == 1 ? 0.9375 : -0.09375);
  return point == 0   ? -0.02197265625
         : point == 1 ? 0.205078125
         : point == 2 ? 0.9228515625
         : point == 3 ? -0.123046875
                      : 0.01708984375;
}

KOKKOS_INLINE_FUNCTION constexpr parthenon::Real RestrictBaseWeight(const int ghost,
                                                                   const int point) {
  if (ghost == 2)
    return point == 0 ? 0.375 : (point == 1 ? 0.75 : -0.125);
  return point == 0   ? -0.0390625
         : point == 1 ? 0.46875
         : point == 2 ? 0.703125
         : point == 3 ? -0.15625
                      : 0.0234375;
}

KOKKOS_INLINE_FUNCTION constexpr parthenon::Real RestrictEdgeWeight(const int point) {
  return point == 0   ? 0.2734375
         : point == 1 ? 1.09375
         : point == 2 ? -0.546875
         : point == 3 ? 0.21875
                      : -0.0390625;
}

static_assert(ProlongBaseWeight(2, 0) + ProlongBaseWeight(2, 1) +
                  ProlongBaseWeight(2, 2) ==
              1.0);
static_assert(RestrictBaseWeight(2, 0) + RestrictBaseWeight(2, 1) +
                  RestrictBaseWeight(2, 2) ==
              1.0);
static_assert(ProlongBaseWeight(4, 0) + ProlongBaseWeight(4, 1) +
                  ProlongBaseWeight(4, 2) + ProlongBaseWeight(4, 3) +
                  ProlongBaseWeight(4, 4) ==
              1.0);
static_assert(RestrictBaseWeight(4, 0) + RestrictBaseWeight(4, 1) +
                  RestrictBaseWeight(4, 2) + RestrictBaseWeight(4, 3) +
                  RestrictBaseWeight(4, 4) ==
              1.0);
static_assert(RestrictEdgeWeight(0) + RestrictEdgeWeight(1) + RestrictEdgeWeight(2) +
                  RestrictEdgeWeight(3) + RestrictEdgeWeight(4) ==
              1.0);

KOKKOS_INLINE_FUNCTION constexpr parthenon::Real
ProlongWeight(const int ghost, const int point, const int child) {
  const int reflected = child == 0 ? point : ghost - point;
  return ProlongBaseWeight(ghost, reflected);
}

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
            parthenon::TopologicalElement /*coarse_element*/ =
                parthenon::TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j,
     const int i, const parthenon::IndexRange &ckb,
     const parthenon::IndexRange &cjb, const parthenon::IndexRange &cib,
     const parthenon::IndexRange &kb, const parthenon::IndexRange &jb,
     const parthenon::IndexRange &ib, const parthenon::Coordinates_t &,
     const parthenon::Coordinates_t &,
     const parthenon::ParArrayND<parthenon::Real, parthenon::VariableState>
         *coarse_ptr,
     const parthenon::ParArrayND<parthenon::Real, parthenon::VariableState>
         *fine_ptr) {
    static_assert(fine_element == parthenon::TopologicalElement::CC);
    const auto &coarse = *coarse_ptr;
    const auto &fine = *fine_ptr;
    constexpr int element =
        static_cast<int>(parthenon::TopologicalElement::CC) % 3;
    const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
    const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
    const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;
    const int ghost = ib.s;
    const int points = ghost == 2 ? 3 : 5;
    const int radius = ghost / 2;

    for (int child_k = 0; child_k < (DIM > 2 ? 2 : 1); ++child_k) {
      for (int child_j = 0; child_j < (DIM > 1 ? 2 : 1); ++child_j) {
        for (int child_i = 0; child_i < (DIM > 0 ? 2 : 1); ++child_i) {
          parthenon::Real value = 0.0;
          for (int pk = 0; pk < (DIM > 2 ? points : 1); ++pk) {
            const parthenon::Real wk =
                DIM > 2 ? ProlongWeight(ghost, pk, child_k) : 1.0;
            for (int pj = 0; pj < (DIM > 1 ? points : 1); ++pj) {
              const parthenon::Real wj =
                  DIM > 1 ? ProlongWeight(ghost, pj, child_j) : 1.0;
              for (int pi = 0; pi < (DIM > 0 ? points : 1); ++pi) {
                const parthenon::Real wi =
                    DIM > 0 ? ProlongWeight(ghost, pi, child_i) : 1.0;
                value +=
                    wk * wj * wi *
                    coarse(element, l, m, n, k - (DIM > 2 ? radius : 0) + pk,
                           j - (DIM > 1 ? radius : 0) + pj,
                           i - (DIM > 0 ? radius : 0) + pi);
              }
            }
          }
          fine(element, l, m, n, fk + child_k, fj + child_j, fi + child_i) =
              value;
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

  KOKKOS_INLINE_FUNCTION static parthenon::Real Weight(const int ghost,
                                                       const int point,
                                                       const bool lower_half,
                                                       const bool edge) {
    const int index = lower_half ? point : ghost - point;
    return edge ? RestrictEdgeWeight(index) : RestrictBaseWeight(ghost, index);
  }

  template <int DIM,
            parthenon::TopologicalElement fine_element =
                parthenon::TopologicalElement::CC,
            parthenon::TopologicalElement /*coarse_element*/ =
                parthenon::TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int ck, const int cj,
     const int ci, const parthenon::IndexRange &ckb,
     const parthenon::IndexRange &cjb, const parthenon::IndexRange &cib,
     const parthenon::IndexRange &kb, const parthenon::IndexRange &jb,
     const parthenon::IndexRange &ib, const parthenon::Coordinates_t &,
     const parthenon::Coordinates_t &,
     const parthenon::ParArrayND<parthenon::Real, parthenon::VariableState>
         *coarse_ptr,
     const parthenon::ParArrayND<parthenon::Real, parthenon::VariableState>
         *fine_ptr) {
    static_assert(fine_element == parthenon::TopologicalElement::CC);
    auto &coarse = *coarse_ptr;
    const auto &fine = *fine_ptr;
    constexpr int element =
        static_cast<int>(parthenon::TopologicalElement::CC) % 3;
    const int fi = (DIM > 0) ? (ci - cib.s) * 2 + ib.s : ib.s;
    const int fj = (DIM > 1) ? (cj - cjb.s) * 2 + jb.s : jb.s;
    const int fk = (DIM > 2) ? (ck - ckb.s) * 2 + kb.s : kb.s;
    const int ghost = ib.s;
    const int points = ghost == 2 ? 3 : 5;
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

    parthenon::Real value = 0.0;
    for (int pk = 0; pk < (DIM > 2 ? points : 1); ++pk) {
      const parthenon::Real wk =
          DIM > 2 ? Weight(ghost, pk, lower_k, edge_k) : 1.0;
      for (int pj = 0; pj < (DIM > 1 ? points : 1); ++pj) {
        const parthenon::Real wj =
            DIM > 1 ? Weight(ghost, pj, lower_j, edge_j) : 1.0;
        for (int pi = 0; pi < (DIM > 0 ? points : 1); ++pi) {
          const parthenon::Real wi =
              DIM > 0 ? Weight(ghost, pi, lower_i, edge_i) : 1.0;
          value += wk * wj * wi *
                   fine(element, l, m, n, ref_k + pk, ref_j + pj, ref_i + pi);
        }
      }
    }
    coarse(element, l, m, n, ck, cj, ci) = value;
  }
};

} // namespace pangu::nr::amr

#endif
