#ifndef PANGU_RELATIVITY_RELATIVISTIC_HYDRO_H_
#define PANGU_RELATIVITY_RELATIVISTIC_HYDRO_H_

#include <memory>

#include <parthenon/parthenon.hpp>

#include "eos/relativistic_eos.h"
#include "geometry/geometry.h"

namespace pangu::relativity {

enum class HydroMode { newtonian = 0, sr = 1, gr = 2 };

struct HydroPrimitiveState {
  parthenon::Real density = 0.0;
  parthenon::Real u[3]{};
  parthenon::Real pressure = 0.0;
};

struct HydroConservedState {
  parthenon::Real density = 0.0;
  parthenon::Real momentum[3]{};
  parthenon::Real energy = 0.0;
};

struct HydroC2PResult {
  HydroPrimitiveState primitive{};
  bool success = true;
  bool density_floor = false;
  bool pressure_floor = false;
  bool lorentz_ceiling = false;
  int iterations = 0;
};

KOKKOS_INLINE_FUNCTION void ComputeCoordinateLightSpeeds(const geometry::MetricPoint& metric,
                                                         const int direction, parthenon::Real& plus,
                                                         parthenon::Real& minus) {
  const int spatial = direction + 1;
  const parthenon::Real g0i = metric.lower[0][spatial];
  const parthenon::Real gii = metric.lower[spatial][spatial];
  const parthenon::Real discriminant = fmax(g0i * g0i - metric.lower[0][0] * gii, 0.0);
  const parthenon::Real root = sqrt(discriminant);
  plus = (-g0i + root) / gii;
  minus = (-g0i - root) / gii;
}

KOKKOS_INLINE_FUNCTION HydroConservedState ConvertSRHDP2C(const HydroPrimitiveState& primitive,
                                                          const eos::RelativisticEOS& eos) {
  HydroConservedState conserved{};
  const parthenon::Real lorentz =
      sqrt(1.0 + primitive.u[0] * primitive.u[0] + primitive.u[1] * primitive.u[1] +
           primitive.u[2] * primitive.u[2]);
  const parthenon::Real enthalpy_lorentz =
      eos.EnthalpyDensity(primitive.density, primitive.pressure) * lorentz;
  conserved.density = primitive.density * lorentz;
  conserved.energy = enthalpy_lorentz * lorentz - primitive.pressure - conserved.density;
  for (int axis = 0; axis < 3; ++axis)
    conserved.momentum[axis] = enthalpy_lorentz * primitive.u[axis];
  return conserved;
}

KOKKOS_INLINE_FUNCTION parthenon::Real
ComputeSRHDC2PResidual(const parthenon::Real z, const parthenon::Real density, const parthenon::Real q,
               const parthenon::Real r, const eos::RelativisticEOS& eos) {
  const parthenon::Real lorentz = sqrt(1.0 + z * z);
  const parthenon::Real rho = density / lorentz;
  parthenon::Real epsilon = lorentz * q - z * r + z * z / (1.0 + lorentz);
  epsilon = fmax(epsilon, eos.SpecificInternalEnergyFloor(rho));
  return z - r / eos.SpecificEnthalpyFromSpecificInternalEnergy(epsilon);
}

KOKKOS_INLINE_FUNCTION HydroC2PResult
SolveSRHDC2PFromInvariants(HydroConservedState conserved, const parthenon::Real momentum2,
                           const eos::RelativisticEOS& eos, const parthenon::Real gamma_max) {
  HydroC2PResult result{};
  if (!(conserved.density >= eos.density_floor)) {
    conserved.density = eos.density_floor;
    result.density_floor = true;
  }
  const parthenon::Real internal_energy_floor = eos.InternalEnergyDensityFloor();
  if (!(conserved.energy >= internal_energy_floor)) {
    conserved.energy = internal_energy_floor;
    result.pressure_floor = true;
  }
  const parthenon::Real q = conserved.energy / conserved.density;
  const parthenon::Real r = sqrt(momentum2) / conserved.density;
  constexpr parthenon::Real vmax = 0.9999999999995;
  const parthenon::Real kmax = 2.0 * vmax / (1.0 + vmax * vmax);
  const parthenon::Real k = fmin(kmax, r / (1.0 + q));
  parthenon::Real lower = 0.5 * k / sqrt(1.0 - 0.25 * k * k);
  parthenon::Real upper = k / sqrt(1.0 - k * k);
  parthenon::Real f_lower = ComputeSRHDC2PResidual(lower, conserved.density, q, r, eos);
  parthenon::Real f_upper = ComputeSRHDC2PResidual(upper, conserved.density, q, r, eos);
  parthenon::Real z = 0.5 * (lower + upper);
  // Match AthenaK's Galeazzi/Illinois inversion iteration cap exactly so that
  // failure classification and repaired states are reproducible.
  constexpr int max_iterations = 25;
  constexpr parthenon::Real tolerance = 1.0e-12;
  for (result.iterations = 0; result.iterations < max_iterations; ++result.iterations) {
    const parthenon::Real denominator = f_upper - f_lower;
    z = fabs(denominator) > 1.0e-30 ? (lower * f_upper - upper * f_lower) / denominator
                                    : 0.5 * (lower + upper);
    const parthenon::Real value = ComputeSRHDC2PResidual(z, conserved.density, q, r, eos);
    if (fabs(upper - lower) < tolerance || fabs(value) < tolerance)
      break;
    if (value * f_upper < 0.0) {
      lower = upper;
      f_lower = f_upper;
      upper = z;
      f_upper = value;
    } else {
      f_lower *= 0.5;
      upper = z;
      f_upper = value;
    }
  }
  if (result.iterations == max_iterations || !isfinite(z)) {
    result.success = false;
    result.primitive = {eos.density_floor, {0.0, 0.0, 0.0}, eos.pressure_floor};
    return result;
  }
  parthenon::Real lorentz = sqrt(1.0 + z * z);
  result.primitive.density = fmax(eos.density_floor, conserved.density / lorentz);
  parthenon::Real epsilon = lorentz * q - z * r + z * z / (1.0 + lorentz);
  const parthenon::Real epsilon_floor = eos.SpecificInternalEnergyFloor(result.primitive.density);
  if (!(epsilon >= epsilon_floor)) {
    epsilon = epsilon_floor;
    result.pressure_floor = true;
  }
  const parthenon::Real enthalpy = eos.SpecificEnthalpyFromSpecificInternalEnergy(epsilon);
  for (int axis = 0; axis < 3; ++axis)
    result.primitive.u[axis] = conserved.momentum[axis] / (conserved.density * enthalpy);
  result.primitive.pressure =
      eos.PressureFromSpecificInternalEnergy(result.primitive.density, epsilon);
  lorentz = sqrt(1.0 + result.primitive.u[0] * result.primitive.u[0] +
                 result.primitive.u[1] * result.primitive.u[1] +
                 result.primitive.u[2] * result.primitive.u[2]);
  if (lorentz > gamma_max) {
    const parthenon::Real scale = sqrt((gamma_max * gamma_max - 1.0) / (lorentz * lorentz - 1.0));
    for (int axis = 0; axis < 3; ++axis)
      result.primitive.u[axis] *= scale;
    result.lorentz_ceiling = true;
  }
  return result;
}

KOKKOS_INLINE_FUNCTION HydroC2PResult SolveSRHDC2P(HydroConservedState conserved,
                                                   const eos::RelativisticEOS& eos,
                                                   const parthenon::Real gamma_max) {
  const parthenon::Real momentum2 = conserved.momentum[0] * conserved.momentum[0] +
                                    conserved.momentum[1] * conserved.momentum[1] +
                                    conserved.momentum[2] * conserved.momentum[2];
  return SolveSRHDC2PFromInvariants(conserved, momentum2, eos, gamma_max);
}

KOKKOS_INLINE_FUNCTION HydroConservedState ConvertGRHDP2C(const HydroPrimitiveState& primitive,
                                                          const eos::RelativisticEOS& eos,
                                                          const geometry::MetricPoint& metric) {
  HydroConservedState conserved{};
  parthenon::Real spatial_u2 = 0.0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      spatial_u2 += metric.lower[a + 1][b + 1] * primitive.u[a] * primitive.u[b];
  const parthenon::Real lorentz = sqrt(1.0 + spatial_u2);
  parthenon::Real u_upper[4]{lorentz / metric.lapse, 0.0, 0.0, 0.0};
  for (int axis = 0; axis < 3; ++axis)
    u_upper[axis + 1] = primitive.u[axis] - metric.lapse * lorentz * metric.upper[0][axis + 1];
  parthenon::Real u_lower[4]{};
  for (int mu = 0; mu < 4; ++mu)
    for (int nu = 0; nu < 4; ++nu)
      u_lower[mu] += metric.lower[mu][nu] * u_upper[nu];
  const parthenon::Real enthalpy_u0 =
      eos.EnthalpyDensity(primitive.density, primitive.pressure) * u_upper[0];
  conserved.density = primitive.density * u_upper[0];
  conserved.energy = enthalpy_u0 * u_lower[0] + primitive.pressure + conserved.density;
  for (int axis = 0; axis < 3; ++axis)
    conserved.momentum[axis] = enthalpy_u0 * u_lower[axis + 1];
  conserved.density *= metric.gdet;
  conserved.energy *= metric.gdet;
  for (int axis = 0; axis < 3; ++axis)
    conserved.momentum[axis] *= metric.gdet;
  return conserved;
}

KOKKOS_INLINE_FUNCTION HydroC2PResult SolveGRHDC2P(const HydroConservedState& conserved,
                                                   const eos::RelativisticEOS& eos,
                                                   const parthenon::Real gamma_max,
                                                   const geometry::MetricPoint& metric) {
  const parthenon::Real inverse_gdet = 1.0 / metric.gdet;
  const HydroConservedState undensitized{conserved.density * inverse_gdet,
                                         {conserved.momentum[0] * inverse_gdet,
                                          conserved.momentum[1] * inverse_gdet,
                                          conserved.momentum[2] * inverse_gdet},
                                         conserved.energy * inverse_gdet};
  HydroConservedState sr{};
  sr.density = undensitized.density * metric.lapse;
  sr.energy = metric.upper[0][0] * (undensitized.energy - undensitized.density);
  for (int axis = 0; axis < 3; ++axis)
    sr.energy += metric.upper[0][axis + 1] * undensitized.momentum[axis];
  sr.energy *= -1.0 / metric.upper[0][0];
  sr.energy -= sr.density;
  parthenon::Real lowered[3];
  for (int axis = 0; axis < 3; ++axis)
    lowered[axis] = undensitized.momentum[axis] * metric.lapse;
  parthenon::Real momentum2 = 0.0;
  for (int a = 0; a < 3; ++a) {
    sr.momentum[a] = 0.0;
    for (int b = 0; b < 3; ++b) {
      const parthenon::Real spatial_inverse =
          metric.upper[a + 1][b + 1] -
          metric.upper[0][a + 1] * metric.upper[0][b + 1] / metric.upper[0][0];
      sr.momentum[a] += spatial_inverse * lowered[b];
    }
    momentum2 += lowered[a] * sr.momentum[a];
  }
  // The scalar inversion is locally special relativistic, but its momentum
  // norm must be formed with the spatial metric.  sr.momentum stores the raised
  // coordinate-basis components, exactly as TransformToSRHyd in AthenaK.
  return SolveSRHDC2PFromInvariants(sr, momentum2, eos, gamma_max);
}

parthenon::TaskStatus
CalculateFluxesBlockTask(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data);
void InitializeExcisionMasksMesh(parthenon::Mesh* mesh, parthenon::ParameterInput* pin,
                                 parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus CalculateFluxesMeshTask(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus
ApplyHydroFluxCorrectionBlockTask(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data,
                                  std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& base,
                                  parthenon::Real gam0, parthenon::Real gam1,
                                  parthenon::Real beta_dt);
parthenon::TaskStatus ApplyHydroFluxCorrectionMeshTask(parthenon::MeshData<parthenon::Real>* data,
                                                       parthenon::MeshData<parthenon::Real>* base,
                                                       parthenon::Real gam0, parthenon::Real gam1,
                                                       parthenon::Real beta_dt);
void ConservedToPrimitiveBlock(parthenon::MeshBlockData<parthenon::Real>* data);
void ConservedToPrimitiveMesh(parthenon::MeshData<parthenon::Real>* data);
parthenon::Real EstimateTimestepBlock(parthenon::MeshBlockData<parthenon::Real>* data);
parthenon::Real EstimateTimestepMesh(parthenon::MeshData<parthenon::Real>* data);
parthenon::TaskStatus
ApplySourcesBlockTask(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>>& data,
                      parthenon::Real dt);
parthenon::TaskStatus ApplySourcesMeshTask(parthenon::MeshData<parthenon::Real>* data,
                                           parthenon::Real dt);

} // namespace pangu::relativity

#endif
