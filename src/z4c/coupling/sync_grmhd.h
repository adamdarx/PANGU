#ifndef PANGU_Z4C_SYNC_GRMHD_H_
#define PANGU_Z4C_SYNC_GRMHD_H_

#include <limits>

#include <parthenon/parthenon.hpp>

#include "geometry/geometry.h"
#include "z4c/coupling/stress_energy.h"
#include "relativity/relativistic_mhd.h"

namespace pangu::nr {

parthenon::TaskStatus
BuildSyncGRMHDStressEnergyBlockTask(parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::TaskStatus
BuildSyncGRMHDStressEnergyMeshTask(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus CalculateSyncGRMHDFluxesMeshTask(
    parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus ResetSyncGRMHDDiagnosticsMeshTask(
    parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus ApplySyncGRMHDFluxCorrectionMeshTask(
    parthenon::MeshData<parthenon::Real>* data,
    parthenon::MeshData<parthenon::Real>* base, parthenon::Real gam0,
    parthenon::Real gam1, parthenon::Real beta_dt);
parthenon::TaskStatus ApplySyncGRMHDPunctureProtectionMeshTask(
    parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus ApplySyncGRMHDSourcesMeshTask(
    parthenon::MeshData<parthenon::Real>* current,
    parthenon::MeshData<parthenon::Real>* next, parthenon::Real dt);
void SyncGRMHDPrimitiveToConservedBlock(parthenon::MeshBlockData<parthenon::Real>* data);
void SyncGRMHDPrimitiveToConservedMesh(parthenon::MeshData<parthenon::Real>* data);
void SyncGRMHDConservedToPrimitiveBlock(parthenon::MeshBlockData<parthenon::Real>* data);
void SyncGRMHDConservedToPrimitiveMesh(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus SyncGRMHDConservedToPrimitiveMeshTask(
    parthenon::MeshData<parthenon::Real>* data);
parthenon::Real EstimateSyncGRMHDTimestepBlock(
    parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::Real EstimateSyncGRMHDTimestepMesh(parthenon::MeshData<parthenon::Real>* data);

struct SyncGRMHDFluxState {
  // AthenaK's dynamic-GRMHD face solver first works with undensitized
  // Eulerian conserved variables and magnetic field.  The Riemann result is
  // multiplied by sqrt(gamma) (and lapse where appropriate) only once.
  parthenon::Real conserved[5]{};
  parthenon::Real magnetic[3]{};
  parthenon::Real flux[5]{};
  parthenon::Real induction[3]{};
  parthenon::Real lambda_plus = 0.0;
  parthenon::Real lambda_minus = 0.0;
};

KOKKOS_INLINE_FUNCTION void InvertSyncSpatialMetric(const geometry::MetricPoint& metric,
                                                     parthenon::Real inverse[3][3]) {
  const parthenon::Real inverse_det = 1.0 / metric.spatial_det;
  inverse[0][0] = (metric.lower[2][2] * metric.lower[3][3] -
                   metric.lower[2][3] * metric.lower[3][2]) *
                  inverse_det;
  inverse[0][1] = (metric.lower[1][3] * metric.lower[3][2] -
                   metric.lower[1][2] * metric.lower[3][3]) *
                  inverse_det;
  inverse[0][2] = (metric.lower[1][2] * metric.lower[2][3] -
                   metric.lower[1][3] * metric.lower[2][2]) *
                  inverse_det;
  inverse[1][0] = inverse[0][1];
  inverse[1][1] = (metric.lower[1][1] * metric.lower[3][3] -
                   metric.lower[1][3] * metric.lower[3][1]) *
                  inverse_det;
  inverse[1][2] = (metric.lower[1][3] * metric.lower[2][1] -
                   metric.lower[1][1] * metric.lower[2][3]) *
                  inverse_det;
  inverse[2][0] = inverse[0][2];
  inverse[2][1] = inverse[1][2];
  inverse[2][2] = (metric.lower[1][1] * metric.lower[2][2] -
                   metric.lower[1][2] * metric.lower[2][1]) *
                  inverse_det;
}

// Dynamic-spacetime GRMHD evolves Valencia variables and a densitized CT
// magnetic field:
//   (D, S_i, tau, Btilde^i) = sqrt(gamma) (Dhat, Shat_i, tauhat, B^i).
// This differs deliberately from the alpha*sqrt(gamma) mixed-tensor state in
// PANGU's analytic fixed-background GRMHD path.
KOKKOS_INLINE_FUNCTION relativity::HydroConservedState ConvertSyncGRMHDP2C(
    const relativity::MHDPrimitiveState& primitive, const eos::RelativisticEOS& eos,
    const geometry::MetricPoint& metric) {
  using parthenon::Real;
  relativity::HydroConservedState conserved{};
  Real projected_lower[3]{};
  Real magnetic_lower[3]{};
  Real projected_squared = 0.0;
  Real magnetic_squared = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      const Real spatial = metric.lower[first + 1][second + 1];
      projected_lower[first] += spatial * primitive.fluid.u[second];
      magnetic_lower[first] += spatial * primitive.magnetic[second];
    }
    projected_squared += primitive.fluid.u[first] * projected_lower[first];
    magnetic_squared += primitive.magnetic[first] * magnetic_lower[first];
  }
  const Real lorentz = sqrt(1.0 + projected_squared);
  const Real inverse_lorentz = 1.0 / lorentz;
  Real magnetic_velocity = 0.0;
  for (int axis = 0; axis < 3; ++axis)
    magnetic_velocity += primitive.magnetic[axis] * projected_lower[axis] * inverse_lorentz;
  const Real gas_enthalpy =
      eos.EnthalpyDensity(primitive.fluid.density, primitive.fluid.pressure);
  const Real enthalpy_lorentz_squared = gas_enthalpy * lorentz * lorentz;
  const Real common = enthalpy_lorentz_squared + magnetic_squared;
  conserved.density = primitive.fluid.density * lorentz;
  for (int axis = 0; axis < 3; ++axis) {
    const Real velocity_lower = projected_lower[axis] * inverse_lorentz;
    conserved.momentum[axis] =
        common * velocity_lower - magnetic_velocity * magnetic_lower[axis];
  }
  conserved.energy = common - primitive.fluid.pressure -
                     0.5 * (magnetic_velocity * magnetic_velocity +
                            magnetic_squared * inverse_lorentz * inverse_lorentz) -
                     conserved.density;
  const Real spatial_volume = sqrt(metric.spatial_det);
  conserved.density *= spatial_volume;
  conserved.energy *= spatial_volume;
  for (int axis = 0; axis < 3; ++axis)
    conserved.momentum[axis] *= spatial_volume;
  return conserved;
}

KOKKOS_INLINE_FUNCTION relativity::MHDC2PResult SolveSyncGRMHDC2P(
    const relativity::HydroConservedState& conserved,
    const parthenon::Real densitized_magnetic[3], const eos::RelativisticEOS& eos,
    const parthenon::Real gamma_max, const parthenon::Real sigma_max,
    const geometry::MetricPoint& metric) {
  using parthenon::Real;
  const Real inverse_volume = 1.0 / sqrt(metric.spatial_det);
  Real spatial_inverse[3][3]{};
  InvertSyncSpatialMetric(metric, spatial_inverse);

  relativity::HydroConservedState local{};
  local.density = conserved.density * inverse_volume;
  local.energy = conserved.energy * inverse_volume;
  Real momentum_lower[3]{};
  Real physical_magnetic[3]{};
  for (int axis = 0; axis < 3; ++axis) {
    momentum_lower[axis] = conserved.momentum[axis] * inverse_volume;
    physical_magnetic[axis] = densitized_magnetic[axis] * inverse_volume;
  }
  for (int axis = 0; axis < 3; ++axis) {
    for (int second = 0; second < 3; ++second)
      local.momentum[axis] += spatial_inverse[axis][second] * momentum_lower[second];
  }
  Real momentum_squared = 0.0;
  Real magnetic_squared = 0.0;
  Real magnetic_momentum = 0.0;
  for (int first = 0; first < 3; ++first) {
    momentum_squared += momentum_lower[first] * local.momentum[first];
    magnetic_momentum += physical_magnetic[first] * momentum_lower[first];
    for (int second = 0; second < 3; ++second)
      magnetic_squared += metric.lower[first + 1][second + 1] *
                          physical_magnetic[first] * physical_magnetic[second];
  }
  const Real parallel = magnetic_momentum / local.density;
  auto result = relativity::SolveSRMHDC2PFromInvariants(
      local, physical_magnetic, momentum_squared, magnetic_squared, parallel, eos,
      std::numeric_limits<Real>::max(), sigma_max);
  Real projected_squared = 0.0;
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      projected_squared += metric.lower[first + 1][second + 1] *
                           result.primitive.fluid.u[first] *
                           result.primitive.fluid.u[second];
  const Real lorentz = sqrt(1.0 + projected_squared);
  if (lorentz > gamma_max) {
    const Real scale = sqrt((gamma_max * gamma_max - 1.0) / projected_squared);
    for (int axis = 0; axis < 3; ++axis)
      result.primitive.fluid.u[axis] *= scale;
    result.lorentz_ceiling = true;
  }
  for (int axis = 0; axis < 3; ++axis)
    result.primitive.magnetic[axis] = physical_magnetic[axis];
  return result;
}

KOKKOS_INLINE_FUNCTION StressEnergy BuildSyncGRMHDStressEnergy(
    const relativity::MHDPrimitiveState& primitive,
    const relativity::HydroConservedState& conserved,
    const geometry::MetricPoint& metric) {
  using parthenon::Real;
  StressEnergy matter{};
  const Real inverse_volume = 1.0 / sqrt(metric.spatial_det);
  Real projected_lower[3]{};
  Real magnetic_lower[3]{};
  Real projected_squared = 0.0;
  Real magnetic_squared = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      const Real spatial = metric.lower[first + 1][second + 1];
      projected_lower[first] += spatial * primitive.fluid.u[second];
      magnetic_lower[first] += spatial * primitive.magnetic[second];
    }
    projected_squared += primitive.fluid.u[first] * projected_lower[first];
    magnetic_squared += primitive.magnetic[first] * magnetic_lower[first];
  }
  const Real inverse_lorentz = 1.0 / sqrt(1.0 + projected_squared);
  Real magnetic_projected_velocity = 0.0;
  for (int axis = 0; axis < 3; ++axis)
    magnetic_projected_velocity += primitive.magnetic[axis] * projected_lower[axis];
  const Real comoving_magnetic_squared =
      (magnetic_squared + magnetic_projected_velocity * magnetic_projected_velocity) *
      inverse_lorentz * inverse_lorentz;
  matter.energy = (conserved.energy + conserved.density) * inverse_volume;
  for (int first = 0; first < 3; ++first) {
    matter.momentum[first] = conserved.momentum[first] * inverse_volume;
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      matter.stress[pair] =
          conserved.momentum[first] * inverse_volume * projected_lower[second] *
              inverse_lorentz -
          (magnetic_lower[first] +
           magnetic_projected_velocity * projected_lower[first]) *
              inverse_lorentz * inverse_lorentz * magnetic_lower[second] +
          (primitive.fluid.pressure + 0.5 * comoving_magnetic_squared) *
              metric.lower[first + 1][second + 1];
    }
  }
  return matter;
}

KOKKOS_INLINE_FUNCTION SyncGRMHDFluxState BuildSyncGRMHDFluxState(
    const relativity::MHDPrimitiveState& primitive, const eos::RelativisticEOS& eos,
    const int direction, const geometry::MetricPoint& metric) {
  using parthenon::Real;
  SyncGRMHDFluxState state{};
  const Real spatial_volume = sqrt(metric.spatial_det);
  const auto densitized = ConvertSyncGRMHDP2C(primitive, eos, metric);
  state.conserved[0] = densitized.density / spatial_volume;
  for (int axis = 0; axis < 3; ++axis) {
    state.conserved[1 + axis] = densitized.momentum[axis] / spatial_volume;
    state.magnetic[axis] = primitive.magnetic[axis];
  }
  state.conserved[4] = densitized.energy / spatial_volume;

  Real projected_lower[3]{};
  Real magnetic_lower[3]{};
  Real projected_squared = 0.0;
  Real magnetic_squared = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      const Real spatial = metric.lower[first + 1][second + 1];
      projected_lower[first] += spatial * primitive.fluid.u[second];
      magnetic_lower[first] += spatial * primitive.magnetic[second];
    }
    projected_squared += primitive.fluid.u[first] * projected_lower[first];
    magnetic_squared += primitive.magnetic[first] * magnetic_lower[first];
  }
  const Real lorentz = sqrt(1.0 + projected_squared);
  const Real inverse_lorentz = 1.0 / lorentz;
  const Real inverse_lapse = 1.0 / metric.lapse;
  Real magnetic_projected_velocity = 0.0;
  for (int axis = 0; axis < 3; ++axis)
    magnetic_projected_velocity += primitive.magnetic[axis] * projected_lower[axis];
  Real comoving_lower[3]{};
  for (int axis = 0; axis < 3; ++axis) {
    comoving_lower[axis] =
        (magnetic_projected_velocity * projected_lower[axis] + magnetic_lower[axis]) *
        inverse_lorentz;
  }
  const Real comoving_magnetic_squared =
      (magnetic_squared + magnetic_projected_velocity * magnetic_projected_velocity) *
      inverse_lorentz * inverse_lorentz;
  const Real transport_velocity =
      primitive.fluid.u[direction] * inverse_lorentz -
      metric.shift[direction] * inverse_lapse;
  state.flux[0] = state.conserved[0] * transport_velocity;
  for (int axis = 0; axis < 3; ++axis) {
    state.flux[1 + axis] = state.conserved[1 + axis] * transport_velocity -
                           comoving_lower[axis] * primitive.magnetic[direction] *
                               inverse_lorentz;
  }
  state.flux[1 + direction] +=
      primitive.fluid.pressure + 0.5 * comoving_magnetic_squared;
  const Real magnetic_four_time = magnetic_projected_velocity * inverse_lapse;
  state.flux[4] =
      state.conserved[4] * transport_velocity -
      metric.lapse * magnetic_four_time * primitive.magnetic[direction] * inverse_lorentz +
      (primitive.fluid.pressure + 0.5 * comoving_magnetic_squared) *
          primitive.fluid.u[direction] * inverse_lorentz;
  for (int axis = 0; axis < 3; ++axis) {
    const Real transverse_transport = primitive.fluid.u[axis] * inverse_lorentz -
                                      metric.shift[axis] * inverse_lapse;
    state.induction[axis] = primitive.magnetic[axis] * transport_velocity -
                            primitive.magnetic[direction] * transverse_transport;
  }

  const Real gas_enthalpy =
      eos.EnthalpyDensity(primitive.fluid.density, primitive.fluid.pressure);
  const Real sound_squared =
      eos.SoundSpeedSquared(primitive.fluid.density, primitive.fluid.pressure);
  const Real alfven_squared =
      comoving_magnetic_squared / (comoving_magnetic_squared + gas_enthalpy);
  const Real fast_squared =
      sound_squared + alfven_squared - sound_squared * alfven_squared;
  const Real upper_time = lorentz * inverse_lapse;
  const Real upper_normal = primitive.fluid.u[direction] -
                            lorentz * metric.shift[direction] * inverse_lapse;
  const int normal = direction + 1;
  const Real a = upper_time * upper_time -
                 (metric.upper[0][0] + upper_time * upper_time) * fast_squared;
  const Real b = -2.0 *
                 (upper_time * upper_normal -
                  (metric.upper[0][normal] + upper_time * upper_normal) * fast_squared);
  const Real c = upper_normal * upper_normal -
                 (metric.upper[normal][normal] + upper_normal * upper_normal) * fast_squared;
  const Real a1 = b / a;
  const Real a0 = c / a;
  const Real root = sqrt(fmax(a1 * a1 - 4.0 * a0, 0.0));
  state.lambda_plus =
      a1 >= 0.0 ? -2.0 * a0 / (a1 + root) : (-a1 + root) / 2.0;
  state.lambda_minus =
      a1 >= 0.0 ? (-a1 - root) / 2.0 : -2.0 * a0 / (a1 - root);
  return state;
}

} // namespace pangu::nr

#endif
