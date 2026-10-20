#ifndef PANGU_Z4C_CLASSICAL_RK4_H_
#define PANGU_Z4C_CLASSICAL_RK4_H_

#include <parthenon/parthenon.hpp>

namespace pangu::nr {

namespace classical_rk4 {

// The original step state is immutable. Only RHS increments are accumulated;
// in particular, a zero RHS must not repeatedly reweight a unit background.
KOKKOS_INLINE_FUNCTION parthenon::Real Accumulate(const int stage,
                                                 const parthenon::Real rhs,
                                                 const parthenon::Real previous) {
  return stage == 1 ? rhs : previous + (stage == 4 ? rhs : 2.0 * rhs);
}

KOKKOS_INLINE_FUNCTION parthenon::Real Update(const int stage,
                                             const parthenon::Real initial,
                                             const parthenon::Real rhs,
                                             const parthenon::Real sum,
                                             const parthenon::Real dt) {
  if (stage == 4) return initial + (dt / 6.0) * sum;
  return initial + (stage == 3 ? dt : 0.5 * dt) * rhs;
}

} // namespace classical_rk4

parthenon::TaskStatus ClassicalRK4UpdateMeshTask(
    parthenon::MeshData<parthenon::Real>* current,
    parthenon::MeshData<parthenon::Real>* initial,
    parthenon::MeshData<parthenon::Real>* rhs_sum, int stage, parthenon::Real dt,
    parthenon::MeshData<parthenon::Real>* next);

} // namespace pangu::nr

#endif
