#ifndef PANGU_Z4C_PUNCTURE_BOWEN_YORK_H_
#define PANGU_Z4C_PUNCTURE_BOWEN_YORK_H_

#include <cmath>

#include <parthenon/parthenon.hpp>

#include "z4c/core/component_indices.h"
#include "z4c/puncture/puncture_data.h"

namespace pangu::nr::puncture {

KOKKOS_INLINE_FUNCTION int LeviCivita(const int i, const int j, const int k) {
  if (i == j || j == k || i == k) return 0;
  return ((i == 0 && j == 1 && k == 2) || (i == 1 && j == 2 && k == 0) ||
          (i == 2 && j == 0 && k == 1))
             ? 1
             : -1;
}

// Evaluate the conformally flat Bowen--York free data.  The returned tensor is
// \tilde A_ij; indices can be raised with delta_ij.  The analytic expression
// satisfies the vacuum momentum constraint away from the punctures.
KOKKOS_INLINE_FUNCTION bool EvaluateFreeData(const parthenon::Real x,
                                             const parthenon::Real y,
                                             const parthenon::Real z,
                                             const Puncture* punctures,
                                             const int count, FreeData& output) {
  using Real = parthenon::Real;
  if (count < 0 || (count > 0 && punctures == nullptr)) return false;
  output = FreeData{};
  const Real point[3] = {x, y, z};
  for (int puncture = 0; puncture < count; ++puncture) {
    const Puncture& body = punctures[puncture];
    if (!(body.mass >= 0.0)) return false;
    Real displacement[3]{};
    Real radius_squared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      displacement[axis] = point[axis] - body.center[axis];
      radius_squared += displacement[axis] * displacement[axis];
    }
    if (!(radius_squared > 0.0)) return false;
    const Real radius = sqrt(radius_squared);
    const Real inverse_radius = 1.0 / radius;
    const Real normal[3] = {displacement[0] * inverse_radius,
                            displacement[1] * inverse_radius,
                            displacement[2] * inverse_radius};
    output.singular_psi += 0.5 * body.mass * inverse_radius;

    Real momentum_normal = 0.0;
    for (int axis = 0; axis < 3; ++axis)
      momentum_normal += body.momentum[axis] * normal[axis];
    Real spin_cross_normal[3]{};
    for (int component = 0; component < 3; ++component)
      for (int spin_axis = 0; spin_axis < 3; ++spin_axis)
        for (int normal_axis = 0; normal_axis < 3; ++normal_axis)
          spin_cross_normal[component] +=
              LeviCivita(spin_axis, normal_axis, component) * body.spin[spin_axis] *
              normal[normal_axis];

    const Real momentum_factor = 1.5 * inverse_radius * inverse_radius;
    const Real spin_factor = 3.0 * inverse_radius * inverse_radius * inverse_radius;
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        const Real delta = first == second ? 1.0 : 0.0;
        const Real momentum_term =
            body.momentum[first] * normal[second] + body.momentum[second] * normal[first] -
            (delta - normal[first] * normal[second]) * momentum_normal;
        const Real spin_term = spin_cross_normal[first] * normal[second] +
                               spin_cross_normal[second] * normal[first];
        output.conformal_extrinsic[SpatialSymmetricComponent(first, second)] +=
            momentum_factor * momentum_term + spin_factor * spin_term;
      }
    }
  }

  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second) {
      const Real value = output.conformal_extrinsic[SpatialSymmetricComponent(first, second)];
      output.conformal_extrinsic_squared += value * value;
    }
  return Kokkos::isfinite(output.singular_psi) &&
         Kokkos::isfinite(output.conformal_extrinsic_squared);
}

// For psi = psi_singular + u and maximal slicing K=0, the Hamiltonian
// constraint is Laplacian(u) = source.
KOKKOS_INLINE_FUNCTION bool EvaluateHamiltonianPoint(const parthenon::Real x,
                                                     const parthenon::Real y,
                                                     const parthenon::Real z,
                                                     const Puncture* punctures,
                                                     const int count,
                                                     const parthenon::Real regular_correction,
                                                     HamiltonianPoint& output) {
  output = HamiltonianPoint{};
  if (!EvaluateFreeData(x, y, z, punctures, count, output.free)) return false;
  output.regular_correction = regular_correction;
  output.psi = output.free.singular_psi + regular_correction;
  if (!(output.psi > 0.0) || !Kokkos::isfinite(output.psi)) return false;
  output.source = -0.125 * pow(output.psi, -7.0) * output.free.conformal_extrinsic_squared;
  return Kokkos::isfinite(output.source);
}

} // namespace pangu::nr::puncture

#endif
