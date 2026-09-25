#ifndef PANGU_Z4C_CONSTRAINTS_H_
#define PANGU_Z4C_CONSTRAINTS_H_

#include <cmath>

#include <parthenon/parthenon.hpp>

#include "z4c/core/component_indices.h"
#include "z4c/evolution/z4c_rhs.h"

namespace pangu::nr {

struct ConstraintValues {
  parthenon::Real values[kConstraintComponents]{};
};

// Physical ADM data and derivatives at one point. The input is deliberately
// independent of Parthenon fields so the same contraction is usable by a fused
// mesh kernel and by a host/device oracle.
struct ADMConstraintInput {
  parthenon::Real metric[6]{};
  parthenon::Real extrinsic[6]{};
  parthenon::Real dmetric[3][6]{};
  parthenon::Real ddmetric[6][6]{};
  parthenon::Real dextrinsic[3][6]{};
  parthenon::Real psi4 = 1.0;
  parthenon::Real dpsi4[3]{};
  parthenon::Real z[3]{};
  // Optional Z4c norm-squared supplied in the conformal metric.  A negative
  // value requests the physical-metric fallback used by standalone ADM
  // callers; the Z4c constraint task fills this with AthenaK's definition.
  parthenon::Real z_norm_squared = -1.0;
  parthenon::Real theta = 0.0;
  parthenon::Real energy = 0.0;
  parthenon::Real momentum[3]{};
};

KOKKOS_INLINE_FUNCTION void ContractedConformalChristoffel(
    const parthenon::Real physical_inverse[6], const parthenon::Real dmetric[3][6],
    const parthenon::Real psi4, const parthenon::Real dpsi4[3],
    parthenon::Real contracted_conformal[3]) {
  using Real = parthenon::Real;
  Real physical_gamma_lower[3][3][3]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      for (int lower = 0; lower < 3; ++lower) {
        physical_gamma_lower[lower][first][second] =
            0.5 * (rhs::SymmetricValue(dmetric[first], lower, second) +
                   rhs::SymmetricValue(dmetric[second], lower, first) -
                   rhs::SymmetricValue(dmetric[lower], first, second));
      }
    }
  }

  // Raising the first index must be a separate pass: each contraction reads
  // all three lower-index components.
  Real physical_gamma_upper[3][3][3]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      for (int upper = 0; upper < 3; ++upper) {
        for (int contracted = 0; contracted < 3; ++contracted) {
          physical_gamma_upper[upper][first][second] +=
              rhs::SymmetricValue(physical_inverse, upper, contracted) *
              physical_gamma_lower[contracted][first][second];
        }
      }
    }
  }

  for (int upper = 0; upper < 3; ++upper) {
    Real physical_contraction = 0.0;
    for (int first = 0; first < 3; ++first)
      for (int second = 0; second < 3; ++second)
        physical_contraction +=
            rhs::SymmetricValue(physical_inverse, first, second) *
            physical_gamma_upper[upper][first][second];
    contracted_conformal[upper] = psi4 * physical_contraction;
    for (int derivative = 0; derivative < 3; ++derivative)
      contracted_conformal[upper] +=
          0.5 * rhs::SymmetricValue(physical_inverse, upper, derivative) * dpsi4[derivative];
  }
}

KOKKOS_INLINE_FUNCTION void ComputeADMConstraints(const ADMConstraintInput& input,
                                                  ConstraintValues& output) {
  using Real = parthenon::Real;
  Real inverse[6]{};
  const Real determinant = rhs::SpatialDeterminant(input.metric);
  if (!(determinant > 0.0)) {
    for (int component = 0; component < kConstraintComponents; ++component)
      output.values[component] = Kokkos::Experimental::finite_max_v<Real>;
    return;
  }
  rhs::SpatialInverse(1.0 / determinant, input.metric, inverse);

  Real christoffel[3][3][3]{};
  for (int upper = 0; upper < 3; ++upper) {
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
        for (int contracted = 0; contracted < 3; ++contracted) {
          christoffel[upper][first][second] +=
              0.5 * rhs::SymmetricValue(inverse, upper, contracted) *
              (rhs::SymmetricValue(input.dmetric[first], contracted, second) +
               rhs::SymmetricValue(input.dmetric[second], contracted, first) -
               rhs::SymmetricValue(input.dmetric[contracted], first, second));
        }
      }
    }
  }

  // Contract the Ricci tensor in the same order as AthenaK's ADM diagnostic.
  // In particular, the second derivatives are combined before any discrete
  // derivative of an inverse metric or Christoffel symbol is formed.  These
  // two continuum-equivalent forms are not discretely equivalent near a
  // puncture, so retaining this ordering is essential for cross-code tests.
  Real christoffel_lower[3][3][3]{};
  for (int upper = 0; upper < 3; ++upper) {
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
        christoffel_lower[upper][first][second] =
            0.5 * (rhs::SymmetricValue(input.dmetric[first], upper, second) +
                   rhs::SymmetricValue(input.dmetric[second], upper, first) -
                   rhs::SymmetricValue(input.dmetric[upper], first, second));
      }
    }
  }
  Real ricci[6]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      Real value = 0.0;
      for (int contracted = 0; contracted < 3; ++contracted) {
        for (int fourth = 0; fourth < 3; ++fourth) {
          for (int upper = 0; upper < 3; ++upper) {
            value += rhs::SymmetricValue(inverse, contracted, fourth) *
                     christoffel[upper][first][contracted] *
                     christoffel_lower[upper][second][fourth];
            value -= rhs::SymmetricValue(inverse, contracted, fourth) *
                     christoffel[upper][first][second] *
                     christoffel_lower[upper][contracted][fourth];
          }
          value += 0.5 * rhs::SymmetricValue(inverse, contracted, fourth) *
                   (-rhs::SymmetricValue(input.ddmetric[SpatialSymmetricComponent(contracted, fourth)],
                                                           first, second) -
                    rhs::SymmetricValue(input.ddmetric[SpatialSymmetricComponent(first, second)],
                                                           contracted, fourth) +
                    rhs::SymmetricValue(input.ddmetric[SpatialSymmetricComponent(first, contracted)],
                                                           second, fourth) +
                    rhs::SymmetricValue(input.ddmetric[SpatialSymmetricComponent(second, contracted)],
                                                           first, fourth));
        }
      }
      ricci[SpatialSymmetricComponent(first, second)] = value;
    }
  }

  Real ricci_scalar = 0.0;
  Real trace_k = 0.0;
  Real k_squared = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      const Real inverse_value = rhs::SymmetricValue(inverse, first, second);
      ricci_scalar += inverse_value * rhs::SymmetricValue(ricci, first, second);
      trace_k += inverse_value * rhs::SymmetricValue(input.extrinsic, first, second);
      for (int third = 0; third < 3; ++third) {
        for (int fourth = 0; fourth < 3; ++fourth) {
          k_squared += inverse_value * rhs::SymmetricValue(inverse, third, fourth) *
                       rhs::SymmetricValue(input.extrinsic, first, third) *
                       rhs::SymmetricValue(input.extrinsic, second, fourth);
        }
      }
    }
  }

  // Form the covariant derivative of K_ij before raising an index.  This is
  // the ordering used by AthenaK (DK_ddd followed by DK_udd); differentiating
  // the mixed tensor directly is continuum-equivalent but has a different
  // round-off/truncation path on a puncture stencil.
  Real dK_cov[3][6]{};
  for (int derivative = 0; derivative < 3; ++derivative) {
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        Real value = rhs::SymmetricValue(input.dextrinsic[derivative], first, second);
        for (int contracted = 0; contracted < 3; ++contracted) {
          value -= christoffel[contracted][derivative][first] *
                   rhs::SymmetricValue(input.extrinsic, contracted, second);
          value -= christoffel[contracted][derivative][second] *
                   rhs::SymmetricValue(input.extrinsic, first, contracted);
        }
        dK_cov[derivative][SpatialSymmetricComponent(first, second)] = value;
      }
    }
  }
  Real momentum_upper[3]{};
  for (int upper = 0; upper < 3; ++upper) {
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
        Real divergence = 0.0;
        Real raised = 0.0;
        for (int contracted = 0; contracted < 3; ++contracted) {
          divergence += rhs::SymmetricValue(inverse, second, contracted) *
                        rhs::SymmetricValue(dK_cov[contracted], first, second);
          raised += rhs::SymmetricValue(inverse, upper, contracted) *
                    rhs::SymmetricValue(dK_cov[contracted], first, second);
        }
        momentum_upper[upper] += rhs::SymmetricValue(inverse, upper, first) * divergence -
                                 rhs::SymmetricValue(inverse, first, second) * raised;
      }
    }
  }

  Real momentum_constraint[3]{};
  for (int lower = 0; lower < 3; ++lower) {
    for (int upper = 0; upper < 3; ++upper)
      momentum_constraint[lower] +=
          rhs::SymmetricValue(input.metric, lower, upper) * momentum_upper[upper];
    momentum_constraint[lower] -= 8.0 * M_PI * input.momentum[lower];
  }

  Real momentum_norm = 0.0;
  Real z_norm = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      momentum_norm += rhs::SymmetricValue(inverse, first, second) * momentum_constraint[first] *
                       momentum_constraint[second];
      z_norm += rhs::SymmetricValue(inverse, first, second) * input.z[first] * input.z[second];
    }
  }
  const Real hamiltonian =
      ricci_scalar + trace_k * trace_k - k_squared - 16.0 * M_PI * input.energy;
  const Real momentum_magnitude = sqrt(fmax(0.0, momentum_norm));
  if (input.z_norm_squared >= 0.0) z_norm = input.z_norm_squared;
  const Real z_magnitude = sqrt(fmax(0.0, z_norm));
  output.values[Index(ConstraintComponent::hamiltonian)] = hamiltonian;
  output.values[Index(ConstraintComponent::momentum_norm)] = momentum_magnitude;
  output.values[Index(ConstraintComponent::z_norm)] = z_magnitude;
  output.values[Index(ConstraintComponent::momentum_x)] = momentum_constraint[0];
  output.values[Index(ConstraintComponent::momentum_y)] = momentum_constraint[1];
  output.values[Index(ConstraintComponent::momentum_z)] = momentum_constraint[2];
  output.values[Index(ConstraintComponent::theta)] = input.theta;
  output.values[Index(ConstraintComponent::combined)] =
      sqrt(fmax(0.0, hamiltonian * hamiltonian + momentum_norm +
                          input.theta * input.theta + 4.0 * z_norm));
}

// Build the physical ADM inputs directly from one Z4c point. This keeps the
// constraint task on the same finite-difference path as the vacuum RHS while
// avoiding persistent derivative fields.
// Evaluate the physical ADM derivatives from the derived ADM fields.  This is
// intentionally separate from the conformal Z4c derivative stencil: AthenaK's
// constraint operator differentiates gamma_ij and K_ij after the Z4c-to-ADM
// conversion, and using the same ordering avoids product-rule truncation terms
// near a puncture.
template <int Order, class ADMPack>
KOKKOS_INLINE_FUNCTION void LoadPhysicalADMConstraints(const ADMPack& adm, const int block,
                                                        const int k, const int j, const int i,
                                                        const parthenon::Real inverse_spacing[3],
                                                        ADMConstraintInput& input) {
  using Real = parthenon::Real;
  const auto metric_accessor = [&](const int component) {
    return rhs::ComponentAccessor<ADMPack>{adm, block, Index(ADMComponent::gxx) + component, k, j, i};
  };
  const auto extrinsic_accessor = [&](const int component) {
    return rhs::ComponentAccessor<ADMPack>{adm, block, Index(ADMComponent::kxx) + component, k, j, i};
  };
  for (int component = 0; component < 6; ++component) {
    input.metric[component] = adm(block, Index(ADMComponent::gxx) + component, k, j, i);
    input.extrinsic[component] = adm(block, Index(ADMComponent::kxx) + component, k, j, i);
    for (int derivative = 0; derivative < 3; ++derivative) {
      input.dmetric[derivative][component] =
          fd::First<Order>(derivative, inverse_spacing[derivative], metric_accessor(component));
      input.dextrinsic[derivative][component] = fd::First<Order>(
          derivative, inverse_spacing[derivative], extrinsic_accessor(component));
    }
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        const int derivative_pair = SpatialSymmetricComponent(first, second);
        if (first == second) {
          input.ddmetric[derivative_pair][component] =
              fd::Second<Order>(first, inverse_spacing[first], metric_accessor(component));
        } else {
          input.ddmetric[derivative_pair][component] = fd::MixedSecond<Order>(
              first, second, inverse_spacing[first], inverse_spacing[second],
              metric_accessor(component));
        }
      }
    }
  }
  const auto psi4_accessor = rhs::ComponentAccessor<ADMPack>{
      adm, block, Index(ADMComponent::psi4), k, j, i};
  input.psi4 = adm(block, Index(ADMComponent::psi4), k, j, i);
  for (int derivative = 0; derivative < 3; ++derivative)
    input.dpsi4[derivative] =
        fd::First<Order>(derivative, inverse_spacing[derivative], psi4_accessor);
}

template <int Order, class Z4cPack, class ADMPack>
KOKKOS_INLINE_FUNCTION void
ComputeZ4cConstraints(const Z4cPack& values, const ADMPack& adm, const int block, const int k,
                      const int j, const int i, const parthenon::Real inverse_spacing[3],
                      const parthenon::Real chi_psi_power, const StressEnergy& matter,
                      ConstraintValues& output) {
  using Real = parthenon::Real;
  rhs::PointState state{};
  rhs::PointDerivatives derivatives{};
  rhs::LoadPoint<Order>(values, block, k, j, i, inverse_spacing, state, derivatives);
  ADMConstraintInput input{};
  if (chi_psi_power == 0.0 || !(state.chi > 0.0)) {
    for (int component = 0; component < kConstraintComponents; ++component)
      output.values[component] = 1.0e300;
    return;
  }

  // Z_i = 1/2 tilde{gamma}_{ij} (Gamma-hat^j - Gamma-tilde^j).
  // Load physical ADM derivatives first.  AthenaK constructs its conformal
  // contracted Christoffel from the physical metric and psi^4 derivative,
  // rather than differentiating the conformal inverse directly.
  LoadPhysicalADMConstraints<Order>(adm, block, k, j, i, inverse_spacing, input);
  Real physical_inverse[6]{};
  const Real physical_determinant = rhs::SpatialDeterminant(input.metric);
  rhs::SpatialInverse(1.0 / physical_determinant, input.metric, physical_inverse);
  Real contracted_conformal[3]{};
  ContractedConformalChristoffel(physical_inverse, input.dmetric, input.psi4, input.dpsi4,
                                 contracted_conformal);
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      input.z[first] += 0.5 * rhs::SymmetricValue(state.metric, first, second) *
                        (state.gamma[second] - contracted_conformal[second]);
  Real z_squared = 0.0;
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      z_squared += 0.25 * rhs::SymmetricValue(state.metric, first, second) *
                   (state.gamma[first] - contracted_conformal[first]) *
                   (state.gamma[second] - contracted_conformal[second]);
  input.z_norm_squared = z_squared;
  input.theta = state.theta;
  input.energy = matter.energy;
  for (int axis = 0; axis < 3; ++axis)
    input.momentum[axis] = matter.momentum[axis];

  // Use the physical ADM fields for H and M, matching the AthenaK ordering.
  ComputeADMConstraints(input, output);
}

template <int Order, class Z4cPack, class ADMPack>
KOKKOS_INLINE_FUNCTION void
ComputeZ4cConstraints(const Z4cPack& values, const ADMPack& adm, const int block, const int k,
                      const int j, const int i, const parthenon::Real inverse_spacing[3],
                      const parthenon::Real chi_psi_power, ConstraintValues& output) {
  ComputeZ4cConstraints<Order>(values, adm, block, k, j, i, inverse_spacing, chi_psi_power,
                               StressEnergy{}, output);
}

} // namespace pangu::nr

#endif
