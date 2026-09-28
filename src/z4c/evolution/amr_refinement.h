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

    for (int child_k = 0; child_k < (DIM > 2 ? 2 : 1); ++child_k) {
      for (int child_j = 0; child_j < (DIM > 1 ? 2 : 1); ++child_j) {
        for (int child_i = 0; child_i < (DIM > 0 ? 2 : 1); ++child_i) {
          parthenon::Real value = 0.0;
          for (int pk = 0; pk < (DIM > 2 ? points : 1); ++pk) {
            const parthenon::Real wk =
                DIM > 2 ? weights[child_k == 0 ? pk : ghost - pk] : 1.0;
            for (int pj = 0; pj < (DIM > 1 ? points : 1); ++pj) {
              const parthenon::Real wj =
                  DIM > 1 ? weights[child_j == 0 ? pj : ghost - pj] : 1.0;
              for (int pi = 0; pi < (DIM > 0 ? points : 1); ++pi) {
                const parthenon::Real wi =
                    DIM > 0 ? weights[child_i == 0 ? pi : ghost - pi] : 1.0;
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
     const parthenon::IndexRange &ib, const parthenon::Coordinates_t &coords,
     const parthenon::Coordinates_t &coarse_coords,
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

    // Parthenon's restriction index space includes the outer half of the coarse
    // ghost region.  For nghost=4 those points map to fi=0 (or the symmetric
    // upper point), where AthenaK's five-point stencil would start at -1 (or
    // finish one past the allocation).  Use the ordinary two-cell restriction
    // only for such points; all locations with a complete stencil retain the
    // AthenaK high-order operator.
    const bool high_i = DIM == 0 || (ref_i >= 0 && ref_i + points <= fine.GetDim(1));
    const bool high_j = DIM <= 1 || (ref_j >= 0 && ref_j + points <= fine.GetDim(2));
    const bool high_k = DIM <= 2 || (ref_k >= 0 && ref_k + points <= fine.GetDim(3));
    if (!(high_i && high_j && high_k)) {
      parthenon::refinement_ops::RestrictAverage::template Do<
          DIM, fine_element, parthenon::TopologicalElement::CC>(
          l, m, n, ck, cj, ci, ckb, cjb, cib, kb, jb, ib, coords,
          coarse_coords, coarse_ptr, fine_ptr);
      return;
    }

    parthenon::Real value = 0.0;
    for (int pk = 0; pk < (DIM > 2 ? points : 1); ++pk) {
      const int weight_k = lower_k ? pk : ghost - pk;
      const parthenon::Real wk = DIM > 2
                                      ? (edge_k && ghost == 4 ? edge_weights[weight_k]
                                                               : weights[weight_k])
                                      : 1.0;
      for (int pj = 0; pj < (DIM > 1 ? points : 1); ++pj) {
        const int weight_j = lower_j ? pj : ghost - pj;
        const parthenon::Real wj = DIM > 1
                                        ? (edge_j && ghost == 4 ? edge_weights[weight_j]
                                                                 : weights[weight_j])
                                        : 1.0;
        for (int pi = 0; pi < (DIM > 0 ? points : 1); ++pi) {
          const int weight_i = lower_i ? pi : ghost - pi;
          const parthenon::Real wi = DIM > 0
                                          ? (edge_i && ghost == 4 ? edge_weights[weight_i]
                                                                   : weights[weight_i])
                                          : 1.0;
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
