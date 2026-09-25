#ifndef PANGU_Z4C_SYNC_GRHD_H_
#define PANGU_Z4C_SYNC_GRHD_H_

#include <string>

#include <parthenon/parthenon.hpp>

#include "geometry/geometry.h"
#include "z4c/coupling/stress_energy.h"
#include "relativity/relativistic_hydro.h"

namespace pangu::nr {

parthenon::TaskStatus
BuildSyncGRHDStressEnergyBlockTask(parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::TaskStatus
BuildSyncGRHDStressEnergyMeshTask(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus CalculateSyncGRHDFluxesMeshTask(
    parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus ApplySyncGRHDSourcesMeshTask(
    parthenon::MeshData<parthenon::Real>* current,
    parthenon::MeshData<parthenon::Real>* next, parthenon::Real dt);
parthenon::TaskStatus ApplySyncMatterSourcesMeshTask(
    parthenon::MeshData<parthenon::Real>* current,
    parthenon::MeshData<parthenon::Real>* next, parthenon::Real dt,
    const std::string& conserved_field);
void SyncGRHDConservedToPrimitiveBlock(parthenon::MeshBlockData<parthenon::Real>* data);
void SyncGRHDConservedToPrimitiveMesh(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus SyncGRHDConservedToPrimitiveMeshTask(
    parthenon::MeshData<parthenon::Real>* data);
parthenon::Real EstimateSyncGRHDTimestepBlock(
    parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::Real EstimateSyncGRHDTimestepMesh(parthenon::MeshData<parthenon::Real>* data);

// Dynamic-spacetime hydrodynamics uses ADM conserved variables rather than the
// mixed coordinate tensor evolved by PANGU's fixed-background GRHD route:
//   D     = sqrt(gamma) rho W,
//   S_i   = sqrt(gamma) h W u_i,
//   tau   = sqrt(gamma) (h W^2 - p - rho W).
// Here primitive.u stores the spatial projection W v^i, as in AthenaK.
KOKKOS_INLINE_FUNCTION relativity::HydroConservedState
ConvertSyncGRHDP2C(const relativity::HydroPrimitiveState& primitive,
                   const eos::RelativisticEOS& eos, const geometry::MetricPoint& metric) {
  relativity::HydroConservedState conserved{};
  parthenon::Real projected_lower[3]{};
  parthenon::Real projected_squared = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second)
      projected_lower[first] += metric.lower[first + 1][second + 1] * primitive.u[second];
    projected_squared += primitive.u[first] * projected_lower[first];
  }
  const parthenon::Real lorentz = sqrt(1.0 + projected_squared);
  const parthenon::Real enthalpy_lorentz =
      eos.EnthalpyDensity(primitive.density, primitive.pressure) * lorentz;
  const parthenon::Real spatial_volume = sqrt(metric.spatial_det);
  conserved.density = spatial_volume * primitive.density * lorentz;
  conserved.energy = spatial_volume *
                         (enthalpy_lorentz * lorentz - primitive.pressure) -
                     conserved.density;
  for (int axis = 0; axis < 3; ++axis)
    conserved.momentum[axis] = spatial_volume * enthalpy_lorentz * projected_lower[axis];
  return conserved;
}

struct SyncGRHDFluxState {
  parthenon::Real conserved[5]{};
  parthenon::Real flux[5]{};
  parthenon::Real lambda_plus = 0.0;
  parthenon::Real lambda_minus = 0.0;
};

// Build AthenaK's dynamic-spacetime Valencia flux at a face.  Conserved and
// flux arrays are both densitized, so the ordinary coordinate-speed HLL
// formula can be applied without any hidden lapse or volume factors.
KOKKOS_INLINE_FUNCTION SyncGRHDFluxState
BuildSyncGRHDFluxState(const relativity::HydroPrimitiveState& primitive,
                       const eos::RelativisticEOS& eos, const int direction,
                       const geometry::MetricPoint& metric) {
  SyncGRHDFluxState state{};
  const auto conserved = ConvertSyncGRHDP2C(primitive, eos, metric);
  state.conserved[0] = conserved.density;
  for (int axis = 0; axis < 3; ++axis)
    state.conserved[1 + axis] = conserved.momentum[axis];
  state.conserved[4] = conserved.energy;

  parthenon::Real projected_squared = 0.0;
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      projected_squared += metric.lower[first + 1][second + 1] * primitive.u[first] *
                           primitive.u[second];
  const parthenon::Real lorentz = sqrt(1.0 + projected_squared);
  const parthenon::Real transport_velocity =
      primitive.u[direction] / lorentz - metric.shift[direction] / metric.lapse;
  const parthenon::Real volume = sqrt(metric.spatial_det);
  state.flux[0] = metric.lapse * conserved.density * transport_velocity;
  for (int axis = 0; axis < 3; ++axis) {
    state.flux[1 + axis] =
        metric.lapse * (conserved.momentum[axis] * transport_velocity +
                        (axis == direction ? volume * primitive.pressure : 0.0));
  }
  state.flux[4] =
      metric.lapse *
      (conserved.energy * transport_velocity +
       volume * primitive.pressure * primitive.u[direction] / lorentz);

  parthenon::Real upper_four_velocity[4]{lorentz / metric.lapse, 0.0, 0.0, 0.0};
  for (int axis = 0; axis < 3; ++axis)
    upper_four_velocity[axis + 1] = primitive.u[axis] -
                                    lorentz * metric.shift[axis] / metric.lapse;
  const parthenon::Real sound_squared =
      eos.SoundSpeedSquared(primitive.density, primitive.pressure);
  const int normal = direction + 1;
  const parthenon::Real a =
      upper_four_velocity[0] * upper_four_velocity[0] -
      (metric.upper[0][0] + upper_four_velocity[0] * upper_four_velocity[0]) * sound_squared;
  const parthenon::Real b =
      -2.0 * (upper_four_velocity[0] * upper_four_velocity[normal] -
              (metric.upper[0][normal] +
               upper_four_velocity[0] * upper_four_velocity[normal]) *
                  sound_squared);
  const parthenon::Real c =
      upper_four_velocity[normal] * upper_four_velocity[normal] -
      (metric.upper[normal][normal] +
       upper_four_velocity[normal] * upper_four_velocity[normal]) *
          sound_squared;
  // Use AthenaK's cancellation-resistant quadratic form verbatim.  The
  // algebraically simpler quadratic formula loses avoidable ulps for nearly
  // hydrostatic states, and those differences enter the dissipative HLLE jump
  // term at every face and every RK stage.
  const parthenon::Real a1 = b / a;
  const parthenon::Real a0 = c / a;
  const parthenon::Real root = sqrt(fmax(a1 * a1 - 4.0 * a0, 0.0));
  state.lambda_plus =
      a1 >= 0.0 ? -2.0 * a0 / (a1 + root) : (-a1 + root) / 2.0;
  state.lambda_minus =
      a1 >= 0.0 ? (-a1 - root) / 2.0 : -2.0 * a0 / (a1 - root);
  return state;
}

KOKKOS_INLINE_FUNCTION relativity::HydroC2PResult
SolveSyncGRHDC2P(const relativity::HydroConservedState& conserved,
                 const eos::RelativisticEOS& eos, const parthenon::Real gamma_max,
                 const geometry::MetricPoint& metric) {
  const parthenon::Real inverse_volume = 1.0 / sqrt(metric.spatial_det);
  const parthenon::Real inverse_det = 1.0 / metric.spatial_det;
  parthenon::Real inverse[3][3]{};
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

  relativity::HydroConservedState local{};
  local.density = conserved.density * inverse_volume;
  local.energy = conserved.energy * inverse_volume;
  parthenon::Real momentum_lower[3]{};
  parthenon::Real momentum_squared = 0.0;
  for (int first = 0; first < 3; ++first) {
    momentum_lower[first] = conserved.momentum[first] * inverse_volume;
    for (int second = 0; second < 3; ++second) {
      local.momentum[first] += inverse[first][second] *
                               conserved.momentum[second] * inverse_volume;
      momentum_squared += inverse[first][second] * momentum_lower[first] *
                          conserved.momentum[second] * inverse_volume;
    }
  }
  return relativity::SolveSRHDC2PFromInvariants(local, momentum_squared, eos, gamma_max);
}

// Reconstruct the undensitized ADM projections using AthenaK's dynamic-GRMHD
// expression order.  E and S_i are read from the stage conserved state; S_ij
// uses the matching primitive velocity and pressure.  This makes a stale
// primitive/conserved pairing visible instead of silently recomputing all ten
// projections from only one representation.
KOKKOS_INLINE_FUNCTION StressEnergy
BuildSyncGRHDStressEnergy(const relativity::HydroPrimitiveState& primitive,
                          const relativity::HydroConservedState& conserved,
                          const geometry::MetricPoint& metric) {
  StressEnergy matter{};
  const parthenon::Real inverse_volume = 1.0 / sqrt(metric.spatial_det);
  parthenon::Real projected_lower[3]{};
  parthenon::Real projected_squared = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      projected_lower[first] += metric.lower[first + 1][second + 1] * primitive.u[second];
      projected_squared += primitive.u[first] * primitive.u[second] *
                           metric.lower[first + 1][second + 1];
    }
  }
  const parthenon::Real inverse_lorentz = 1.0 / sqrt(1.0 + projected_squared);
  matter.energy = (conserved.energy + conserved.density) * inverse_volume;
  for (int first = 0; first < 3; ++first) {
    matter.momentum[first] = conserved.momentum[first] * inverse_volume;
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      matter.stress[pair] = conserved.momentum[first] * inverse_volume *
                                projected_lower[second] * inverse_lorentz +
                            primitive.pressure * metric.lower[first + 1][second + 1];
    }
  }
  return matter;
}

struct SyncGRHDGeometryDerivatives {
  parthenon::Real lapse[3]{};
  parthenon::Real shift[3][3]{};          // derivative direction, upper shift component
  parthenon::Real spatial_metric[3][6]{}; // derivative direction, symmetric pair
  parthenon::Real extrinsic[6]{};
};

// Valencia/ADM geometric source rates for (D,S_i,tau).  The mass equation has
// no local geometric source.  The expression is the unmagnetized specialization
// of AthenaK DynGRMHDPS::AddCoordTermsEOS.
KOKKOS_INLINE_FUNCTION void
ComputeSyncGRHDSource(const StressEnergy& matter, const geometry::MetricPoint& metric,
                      const SyncGRHDGeometryDerivatives& derivatives,
                      parthenon::Real output[5]) {
  parthenon::Real inverse[3][3]{};
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

  parthenon::Real stress_upper[3][3]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      for (int lower_first = 0; lower_first < 3; ++lower_first) {
        for (int lower_second = 0; lower_second < 3; ++lower_second) {
          stress_upper[first][second] +=
              inverse[first][lower_first] * inverse[second][lower_second] *
              matter.stress[SpatialSymmetricComponent(lower_first, lower_second)];
        }
      }
    }
  }

  const parthenon::Real volume = sqrt(metric.spatial_det);
  output[0] = 0.0;
  output[4] = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      output[4] += volume *
                   (metric.lapse *
                        derivatives.extrinsic[SpatialSymmetricComponent(first, second)] *
                        stress_upper[first][second] -
                    inverse[first][second] * matter.momentum[first] *
                        derivatives.lapse[second]);
    }
  }
  for (int direction = 0; direction < 3; ++direction) {
    output[1 + direction] = -volume * matter.energy * derivatives.lapse[direction];
    for (int first = 0; first < 3; ++first) {
      output[1 + direction] +=
          volume * matter.momentum[first] * derivatives.shift[direction][first];
      for (int second = 0; second < 3; ++second) {
        output[1 + direction] +=
            0.5 * metric.lapse * volume * stress_upper[first][second] *
            derivatives.spatial_metric[direction]
                                              [SpatialSymmetricComponent(first, second)];
      }
    }
  }
}

} // namespace pangu::nr

#endif
