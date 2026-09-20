#ifndef PANGU_Z4C_Z4C_RHS_H_
#define PANGU_Z4C_Z4C_RHS_H_

#include <cmath>

#include <parthenon/parthenon.hpp>

#include "z4c/component_indices.h"
#include "z4c/finite_difference.h"
#include "z4c/stress_energy.h"

namespace pangu::nr {

struct Z4cOptions {
  parthenon::Real chi_psi_power = -4.0;
  parthenon::Real chi_div_floor = -1000.0;
  parthenon::Real chi_min_floor = 1.0e-12;
  bool floor_chi = false;
  parthenon::Real dissipation = 0.0;
  parthenon::Real damp_kappa1 = 0.0;
  parthenon::Real damp_kappa2 = 0.0;
  parthenon::Real lapse_oplog = 2.0;
  parthenon::Real lapse_harmonic_factor = 1.0;
  parthenon::Real lapse_harmonic = 0.0;
  parthenon::Real lapse_advect = 1.0;
  bool slow_start_lapse = false;
  parthenon::Real slow_start_amplitude = 0.6;
  parthenon::Real slow_start_time = 20.0;
  parthenon::Real slow_start_index = 1.0;
  parthenon::Real shift_gamma = 1.0;
  parthenon::Real shift_alpha2_gamma = 0.0;
  parthenon::Real shift_harmonic = 0.0;
  parthenon::Real shift_advect = 1.0;
  parthenon::Real shift_eta = 2.0;
  bool use_z4c = true;
};

namespace rhs {

using Real = parthenon::Real;

struct PointState {
  Real chi = 0.0;
  Real metric[6]{};
  Real khat = 0.0;
  Real a[6]{};
  Real gamma[3]{};
  Real theta = 0.0;
  Real alpha = 0.0;
  Real beta[3]{};
};

struct PointDerivatives {
  Real dalpha[3]{};
  Real dchi[3]{};
  Real dkhat[3]{};
  Real dtheta[3]{};
  Real dbeta[3][3]{};
  Real dgamma[3][3]{};
  Real dmetric[3][6]{};
  Real d_a[3][6]{};
  Real ddalpha[6]{};
  Real ddchi[6]{};
  Real ddbeta[6][3]{};
  Real ddmetric[6][6]{};
  Real lie_alpha = 0.0;
  Real lie_chi = 0.0;
  Real lie_khat = 0.0;
  Real lie_theta = 0.0;
  Real lie_beta[3]{};
  Real lie_gamma[3]{};
  Real lie_metric[6]{};
  Real lie_a[6]{};
};

KOKKOS_INLINE_FUNCTION Real SymmetricValue(const Real values[6], const int first,
                                           const int second) {
  return values[SpatialSymmetricComponent(first, second)];
}

KOKKOS_INLINE_FUNCTION Real SymmetricDerivative(const Real values[3][6], const int derivative,
                                                const int first, const int second) {
  return values[derivative][SpatialSymmetricComponent(first, second)];
}

KOKKOS_INLINE_FUNCTION Real DoubleSymmetricDerivative(const Real values[6][6],
                                                      const int first_derivative,
                                                      const int second_derivative, const int first,
                                                      const int second) {
  return values[SpatialSymmetricComponent(first_derivative, second_derivative)]
               [SpatialSymmetricComponent(first, second)];
}

template <class Pack> struct ComponentAccessor {
  Pack values;
  int block;
  int component;
  int k;
  int j;
  int i;

  KOKKOS_INLINE_FUNCTION Real operator()(const int dk, const int dj, const int di) const {
    return values(block, component, k + dk, j + dj, i + di);
  }
};

template <int Order, class Pack>
KOKKOS_INLINE_FUNCTION void LoadPoint(const Pack& values, const int block, const int k, const int j,
                                      const int i, const Real inverse_spacing[3], PointState& state,
                                      PointDerivatives& derivatives) {
  const auto accessor = [&](const int component) {
    return ComponentAccessor<Pack>{values, block, component, k, j, i};
  };
  state.chi = values(block, Index(Z4cComponent::chi), k, j, i);
  state.khat = values(block, Index(Z4cComponent::khat), k, j, i);
  state.theta = values(block, Index(Z4cComponent::theta), k, j, i);
  state.alpha = values(block, Index(Z4cComponent::alpha), k, j, i);
  for (int axis = 0; axis < 3; ++axis) {
    state.gamma[axis] = values(block, Index(Z4cComponent::gamx) + axis, k, j, i);
    state.beta[axis] = values(block, Index(Z4cComponent::betax) + axis, k, j, i);
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int symmetric = SpatialSymmetricComponent(first, second);
      state.metric[symmetric] = values(block, Index(Z4cComponent::gxx) + symmetric, k, j, i);
      state.a[symmetric] = values(block, Index(Z4cComponent::axx) + symmetric, k, j, i);
    }
  }

  for (int derivative = 0; derivative < 3; ++derivative) {
    derivatives.dalpha[derivative] = fd::First<Order>(derivative, inverse_spacing[derivative],
                                                      accessor(Index(Z4cComponent::alpha)));
    derivatives.dchi[derivative] = fd::First<Order>(derivative, inverse_spacing[derivative],
                                                    accessor(Index(Z4cComponent::chi)));
    derivatives.dkhat[derivative] = fd::First<Order>(derivative, inverse_spacing[derivative],
                                                     accessor(Index(Z4cComponent::khat)));
    derivatives.dtheta[derivative] = fd::First<Order>(derivative, inverse_spacing[derivative],
                                                      accessor(Index(Z4cComponent::theta)));
    for (int component = 0; component < 3; ++component) {
      derivatives.dbeta[derivative][component] =
          fd::First<Order>(derivative, inverse_spacing[derivative],
                           accessor(Index(Z4cComponent::betax) + component));
      derivatives.dgamma[derivative][component] = fd::First<Order>(
          derivative, inverse_spacing[derivative], accessor(Index(Z4cComponent::gamx) + component));
    }
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        const int symmetric = SpatialSymmetricComponent(first, second);
        derivatives.dmetric[derivative][symmetric] =
            fd::First<Order>(derivative, inverse_spacing[derivative],
                             accessor(Index(Z4cComponent::gxx) + symmetric));
        derivatives.d_a[derivative][symmetric] =
            fd::First<Order>(derivative, inverse_spacing[derivative],
                             accessor(Index(Z4cComponent::axx) + symmetric));
      }
    }
  }

  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int derivative_pair = SpatialSymmetricComponent(first, second);
      if (first == second) {
        derivatives.ddalpha[derivative_pair] =
            fd::Second<Order>(first, inverse_spacing[first], accessor(Index(Z4cComponent::alpha)));
        derivatives.ddchi[derivative_pair] =
            fd::Second<Order>(first, inverse_spacing[first], accessor(Index(Z4cComponent::chi)));
      } else {
        derivatives.ddalpha[derivative_pair] =
            fd::MixedSecond<Order>(first, second, inverse_spacing[first], inverse_spacing[second],
                                   accessor(Index(Z4cComponent::alpha)));
        derivatives.ddchi[derivative_pair] =
            fd::MixedSecond<Order>(first, second, inverse_spacing[first], inverse_spacing[second],
                                   accessor(Index(Z4cComponent::chi)));
      }
      for (int component = 0; component < 3; ++component) {
        if (first == second) {
          derivatives.ddbeta[derivative_pair][component] = fd::Second<Order>(
              first, inverse_spacing[first], accessor(Index(Z4cComponent::betax) + component));
        } else {
          derivatives.ddbeta[derivative_pair][component] =
              fd::MixedSecond<Order>(first, second, inverse_spacing[first], inverse_spacing[second],
                                     accessor(Index(Z4cComponent::betax) + component));
        }
      }
      for (int metric_first = 0; metric_first < 3; ++metric_first) {
        for (int metric_second = metric_first; metric_second < 3; ++metric_second) {
          const int metric_pair = SpatialSymmetricComponent(metric_first, metric_second);
          if (first == second) {
            derivatives.ddmetric[derivative_pair][metric_pair] = fd::Second<Order>(
                first, inverse_spacing[first], accessor(Index(Z4cComponent::gxx) + metric_pair));
          } else {
            derivatives.ddmetric[derivative_pair][metric_pair] = fd::MixedSecond<Order>(
                first, second, inverse_spacing[first], inverse_spacing[second],
                accessor(Index(Z4cComponent::gxx) + metric_pair));
          }
        }
      }
    }
  }

  for (int derivative = 0; derivative < 3; ++derivative) {
    derivatives.lie_alpha +=
        fd::AdvectiveFirst<Order>(derivative, inverse_spacing[derivative], state.beta[derivative],
                                  accessor(Index(Z4cComponent::alpha)));
    derivatives.lie_chi +=
        fd::AdvectiveFirst<Order>(derivative, inverse_spacing[derivative], state.beta[derivative],
                                  accessor(Index(Z4cComponent::chi)));
    derivatives.lie_khat +=
        fd::AdvectiveFirst<Order>(derivative, inverse_spacing[derivative], state.beta[derivative],
                                  accessor(Index(Z4cComponent::khat)));
    derivatives.lie_theta +=
        fd::AdvectiveFirst<Order>(derivative, inverse_spacing[derivative], state.beta[derivative],
                                  accessor(Index(Z4cComponent::theta)));
    for (int component = 0; component < 3; ++component) {
      derivatives.lie_beta[component] +=
          fd::AdvectiveFirst<Order>(derivative, inverse_spacing[derivative], state.beta[derivative],
                                    accessor(Index(Z4cComponent::betax) + component));
      derivatives.lie_gamma[component] +=
          fd::AdvectiveFirst<Order>(derivative, inverse_spacing[derivative], state.beta[derivative],
                                    accessor(Index(Z4cComponent::gamx) + component));
    }
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        const int symmetric = SpatialSymmetricComponent(first, second);
        derivatives.lie_metric[symmetric] += fd::AdvectiveFirst<Order>(
            derivative, inverse_spacing[derivative], state.beta[derivative],
            accessor(Index(Z4cComponent::gxx) + symmetric));
        derivatives.lie_a[symmetric] += fd::AdvectiveFirst<Order>(
            derivative, inverse_spacing[derivative], state.beta[derivative],
            accessor(Index(Z4cComponent::axx) + symmetric));
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION Real SpatialDeterminant(const Real metric[6]) {
  const Real gxx = metric[0];
  const Real gxy = metric[1];
  const Real gxz = metric[2];
  const Real gyy = metric[3];
  const Real gyz = metric[4];
  const Real gzz = metric[5];
  return -gxz * gxz * gyy + 2.0 * gxy * gxz * gyz - gyz * gyz * gxx - gxy * gxy * gzz +
         gxx * gyy * gzz;
}

KOKKOS_INLINE_FUNCTION void SpatialInverse(const Real inverse_determinant, const Real metric[6],
                                           Real inverse[6]) {
  const Real gxx = metric[0];
  const Real gxy = metric[1];
  const Real gxz = metric[2];
  const Real gyy = metric[3];
  const Real gyz = metric[4];
  const Real gzz = metric[5];
  inverse[0] = (-gyz * gyz + gyy * gzz) * inverse_determinant;
  inverse[1] = (gxz * gyz - gxy * gzz) * inverse_determinant;
  inverse[3] = (-gxz * gxz + gxx * gzz) * inverse_determinant;
  inverse[2] = (-gxz * gyy + gxy * gyz) * inverse_determinant;
  inverse[4] = (gxy * gxz - gxx * gyz) * inverse_determinant;
  inverse[5] = (-gxy * gxy + gxx * gyy) * inverse_determinant;
}

KOKKOS_INLINE_FUNCTION void ComputeVacuum(const PointState& state,
                                          const PointDerivatives& derivatives,
                                          const Z4cOptions& options, const Real time,
                                          Real output[kZ4cComponents]) {
  for (int component = 0; component < kZ4cComponents; ++component)
    output[component] = 0.0;

  Real inverse_metric[6]{};
  const Real determinant = SpatialDeterminant(state.metric);
  SpatialInverse(1.0 / determinant, state.metric, inverse_metric);

  Real gamma_lower[3][6]{};
  Real gamma_upper[3][6]{};
  Real contracted_gamma[3]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      for (int lowered = 0; lowered < 3; ++lowered) {
        gamma_lower[lowered][pair] =
            0.5 * (SymmetricDerivative(derivatives.dmetric, first, second, lowered) +
                   SymmetricDerivative(derivatives.dmetric, second, first, lowered) -
                   SymmetricDerivative(derivatives.dmetric, lowered, first, second));
      }
      for (int raised = 0; raised < 3; ++raised) {
        for (int lowered = 0; lowered < 3; ++lowered) {
          gamma_upper[raised][pair] +=
              SymmetricValue(inverse_metric, raised, lowered) * gamma_lower[lowered][pair];
        }
      }
    }
  }
  for (int raised = 0; raised < 3; ++raised) {
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
        contracted_gamma[raised] += SymmetricValue(inverse_metric, first, second) *
                                    gamma_upper[raised][SpatialSymmetricComponent(first, second)];
      }
    }
  }

  Real ricci[6]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      for (int third = 0; third < 3; ++third) {
        ricci[pair] +=
            0.5 * (SymmetricValue(state.metric, third, first) * derivatives.dgamma[second][third] +
                   SymmetricValue(state.metric, third, second) * derivatives.dgamma[first][third] +
                   contracted_gamma[third] *
                       (gamma_lower[first][SpatialSymmetricComponent(second, third)] +
                        gamma_lower[second][SpatialSymmetricComponent(first, third)]));
      }
      for (int third = 0; third < 3; ++third) {
        for (int fourth = 0; fourth < 3; ++fourth) {
          ricci[pair] -=
              0.5 * SymmetricValue(inverse_metric, third, fourth) *
              DoubleSymmetricDerivative(derivatives.ddmetric, third, fourth, first, second);
        }
      }
      for (int third = 0; third < 3; ++third) {
        for (int fourth = 0; fourth < 3; ++fourth) {
          for (int fifth = 0; fifth < 3; ++fifth) {
            ricci[pair] += SymmetricValue(inverse_metric, third, fourth) *
                           (gamma_upper[fifth][SpatialSymmetricComponent(third, first)] *
                                gamma_lower[second][SpatialSymmetricComponent(fifth, fourth)] +
                            gamma_upper[fifth][SpatialSymmetricComponent(third, second)] *
                                gamma_lower[first][SpatialSymmetricComponent(fifth, fourth)] +
                            gamma_upper[fifth][SpatialSymmetricComponent(first, fourth)] *
                                gamma_lower[fifth][SpatialSymmetricComponent(third, second)]);
          }
        }
      }
    }
  }

  const Real chi_guarded = state.chi > options.chi_div_floor ? state.chi : options.chi_div_floor;
  const Real inverse_psi4 = pow(chi_guarded, -4.0 / options.chi_psi_power);
  Real dphi[3]{};
  Real covariant_ddphi[6]{};
  Real ricci_phi[6]{};
  for (int first = 0; first < 3; ++first)
    dphi[first] = derivatives.dchi[first] / (chi_guarded * options.chi_psi_power);
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      covariant_ddphi[pair] = derivatives.ddchi[pair] / (chi_guarded * options.chi_psi_power) -
                              options.chi_psi_power * dphi[first] * dphi[second];
      for (int third = 0; third < 3; ++third) {
        covariant_ddphi[pair] -= gamma_upper[third][pair] * dphi[third];
      }
    }
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      ricci_phi[pair] = 4.0 * dphi[first] * dphi[second] - 2.0 * covariant_ddphi[pair];
      for (int third = 0; third < 3; ++third) {
        for (int fourth = 0; fourth < 3; ++fourth) {
          ricci_phi[pair] -=
              2.0 * state.metric[pair] * SymmetricValue(inverse_metric, third, fourth) *
              (SymmetricValue(covariant_ddphi, third, fourth) + 2.0 * dphi[third] * dphi[fourth]);
        }
      }
    }
  }

  Real covariant_ddalpha[6]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      covariant_ddalpha[pair] =
          derivatives.ddalpha[pair] - 2.0 * (dphi[first] * derivatives.dalpha[second] +
                                             dphi[second] * derivatives.dalpha[first]);
      for (int third = 0; third < 3; ++third) {
        covariant_ddalpha[pair] -= gamma_upper[third][pair] * derivatives.dalpha[third];
        for (int fourth = 0; fourth < 3; ++fourth) {
          covariant_ddalpha[pair] += 2.0 * state.metric[pair] *
                                     SymmetricValue(inverse_metric, third, fourth) * dphi[third] *
                                     derivatives.dalpha[fourth];
        }
      }
    }
  }
  Real trace_ddalpha = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      trace_ddalpha += inverse_psi4 * SymmetricValue(inverse_metric, first, second) *
                       SymmetricValue(covariant_ddalpha, first, second);
    }
  }

  Real aa_lower[6]{};
  Real a_upper[6]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      for (int third = 0; third < 3; ++third) {
        for (int fourth = 0; fourth < 3; ++fourth) {
          aa_lower[pair] += SymmetricValue(inverse_metric, third, fourth) *
                            SymmetricValue(state.a, first, third) *
                            SymmetricValue(state.a, fourth, second);
          a_upper[pair] += SymmetricValue(inverse_metric, first, third) *
                           SymmetricValue(inverse_metric, second, fourth) *
                           SymmetricValue(state.a, third, fourth);
        }
      }
    }
  }
  Real aa_trace = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second)
      aa_trace +=
          SymmetricValue(inverse_metric, first, second) * SymmetricValue(aa_lower, first, second);
  }
  Real divergence_a[3]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      divergence_a[first] -=
          1.5 * SymmetricValue(a_upper, first, second) * derivatives.dchi[second] / chi_guarded;
      divergence_a[first] -= (1.0 / 3.0) * SymmetricValue(inverse_metric, first, second) *
                             (2.0 * derivatives.dkhat[second] + derivatives.dtheta[second]);
    }
    for (int second = 0; second < 3; ++second) {
      for (int third = 0; third < 3; ++third) {
        divergence_a[first] += gamma_upper[first][SpatialSymmetricComponent(second, third)] *
                               SymmetricValue(a_upper, second, third);
      }
    }
  }

  Real ricci_scalar = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      ricci_scalar +=
          inverse_psi4 * SymmetricValue(inverse_metric, first, second) *
          (SymmetricValue(ricci, first, second) + SymmetricValue(ricci_phi, first, second));
    }
  }
  const Real trace_k = state.khat + 2.0 * state.theta;
  const Real hamiltonian_tilde = ricci_scalar + (2.0 / 3.0) * trace_k * trace_k - aa_trace;

  Real divergence_beta = 0.0;
  Real second_divergence_beta[3]{};
  for (int first = 0; first < 3; ++first) {
    divergence_beta += derivatives.dbeta[first][first];
    for (int second = 0; second < 3; ++second) {
      second_divergence_beta[first] +=
          (1.0 / 3.0) * derivatives.ddbeta[SpatialSymmetricComponent(first, second)][second];
    }
  }

  Real lie_chi =
      derivatives.lie_chi + (1.0 / 6.0) * options.chi_psi_power * chi_guarded * divergence_beta;
  Real lie_gamma[3]{};
  for (int first = 0; first < 3; ++first) {
    lie_gamma[first] =
        derivatives.lie_gamma[first] + (2.0 / 3.0) * contracted_gamma[first] * divergence_beta;
    for (int second = 0; second < 3; ++second) {
      lie_gamma[first] +=
          SymmetricValue(inverse_metric, first, second) * second_divergence_beta[second] -
          contracted_gamma[second] * derivatives.dbeta[second][first];
      for (int third = 0; third < 3; ++third) {
        lie_gamma[first] += SymmetricValue(inverse_metric, second, third) *
                            derivatives.ddbeta[SpatialSymmetricComponent(second, third)][first];
      }
    }
  }

  Real lie_metric[6]{};
  Real lie_a[6]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      lie_metric[pair] =
          derivatives.lie_metric[pair] - (2.0 / 3.0) * state.metric[pair] * divergence_beta;
      lie_a[pair] = derivatives.lie_a[pair] - (2.0 / 3.0) * state.a[pair] * divergence_beta;
      for (int third = 0; third < 3; ++third) {
        lie_metric[pair] +=
            derivatives.dbeta[first][third] * SymmetricValue(state.metric, second, third) +
            derivatives.dbeta[second][third] * SymmetricValue(state.metric, first, third);
        lie_a[pair] += derivatives.dbeta[second][third] * SymmetricValue(state.a, first, third) +
                       derivatives.dbeta[first][third] * SymmetricValue(state.a, second, third);
      }
    }
  }

  output[Index(Z4cComponent::khat)] =
      -trace_ddalpha + state.alpha * (aa_trace + (1.0 / 3.0) * trace_k * trace_k) +
      derivatives.lie_khat +
      options.damp_kappa1 * (1.0 - options.damp_kappa2) * state.alpha * state.theta;
  output[Index(Z4cComponent::chi)] =
      lie_chi - (1.0 / 6.0) * options.chi_psi_power * chi_guarded * state.alpha * trace_k;
  output[Index(Z4cComponent::theta)] =
      (derivatives.lie_theta +
       state.alpha * (0.5 * hamiltonian_tilde -
                      (2.0 + options.damp_kappa2) * options.damp_kappa1 * state.theta)) *
      static_cast<Real>(options.use_z4c);
  for (int first = 0; first < 3; ++first) {
    Real gamma_rhs =
        2.0 * state.alpha * divergence_a[first] + lie_gamma[first] -
        2.0 * state.alpha * options.damp_kappa1 * (state.gamma[first] - contracted_gamma[first]);
    for (int second = 0; second < 3; ++second) {
      gamma_rhs -= 2.0 * SymmetricValue(a_upper, first, second) * derivatives.dalpha[second];
    }
    output[Index(Z4cComponent::gamx) + first] = gamma_rhs;
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      output[Index(Z4cComponent::gxx) + pair] =
          -2.0 * state.alpha * state.a[pair] + lie_metric[pair];
      output[Index(Z4cComponent::axx) + pair] =
          inverse_psi4 *
              (-covariant_ddalpha[pair] + state.alpha * (ricci[pair] + ricci_phi[pair])) -
          (1.0 / 3.0) * state.metric[pair] * (-trace_ddalpha + state.alpha * ricci_scalar) +
          state.alpha * (trace_k * state.a[pair] - 2.0 * aa_lower[pair]) + lie_a[pair];
    }
  }

  const Real lapse_factor =
      options.lapse_oplog * options.lapse_harmonic_factor + options.lapse_harmonic * state.alpha;
  output[Index(Z4cComponent::alpha)] =
      options.lapse_advect * derivatives.lie_alpha - lapse_factor * state.alpha * state.khat;
  if (options.slow_start_lapse) {
    const Real w2 = state.chi > options.chi_min_floor ? state.chi : options.chi_min_floor;
    const Real w = sqrt(w2);
    const Real time_ratio = time / options.slow_start_time;
    output[Index(Z4cComponent::alpha)] += options.slow_start_amplitude * (w - state.alpha) *
                                          pow(w, options.slow_start_index) *
                                          exp(-0.5 * time_ratio * time_ratio);
  }
  for (int first = 0; first < 3; ++first) {
    Real beta_rhs = options.shift_gamma * state.gamma[first] +
                    options.shift_advect * derivatives.lie_beta[first] -
                    options.shift_eta * state.beta[first] +
                    options.shift_alpha2_gamma * state.alpha * state.alpha * state.gamma[first];
    for (int second = 0; second < 3; ++second) {
      beta_rhs += options.shift_harmonic * state.alpha * chi_guarded *
                  (0.5 * state.alpha * derivatives.dchi[second] - derivatives.dalpha[second]) *
                  SymmetricValue(inverse_metric, first, second);
    }
    output[Index(Z4cComponent::betax) + first] = beta_rhs;
  }
}

// Add the undensitized ADM matter projections to the vacuum Z4c right-hand
// side.  Keeping this as a separate point operator preserves the zero-cost
// vacuum kernel while giving analytic sources and the later GRHD/GRMHD
// adapters one field-level contract.  The expression order follows AthenaK's
// z4c_calcrhs.cpp matter branches.
KOKKOS_INLINE_FUNCTION void ApplyMatterSources(const PointState& state,
                                               const Z4cOptions& options,
                                               const StressEnergy& matter,
                                               Real output[kZ4cComponents]) {
  Real inverse_metric[6]{};
  const Real determinant = SpatialDeterminant(state.metric);
  SpatialInverse(1.0 / determinant, state.metric, inverse_metric);
  const Real chi_guarded = state.chi > options.chi_div_floor ? state.chi : options.chi_div_floor;
  const Real inverse_psi4 = pow(chi_guarded, -4.0 / options.chi_psi_power);

  Real stress_trace = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      stress_trace += inverse_psi4 * SymmetricValue(inverse_metric, first, second) *
                      SymmetricValue(matter.stress, first, second);
    }
  }

  output[Index(Z4cComponent::khat)] +=
      4.0 * M_PI * state.alpha * (stress_trace + matter.energy);
  output[Index(Z4cComponent::theta)] -=
      static_cast<Real>(options.use_z4c) * 8.0 * M_PI * state.alpha * matter.energy;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      output[Index(Z4cComponent::gamx) + first] -=
          16.0 * M_PI * state.alpha * SymmetricValue(inverse_metric, first, second) *
          matter.momentum[second];
    }
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      output[Index(Z4cComponent::axx) + pair] -=
          8.0 * M_PI * state.alpha *
          (inverse_psi4 * matter.stress[pair] -
           (1.0 / 3.0) * stress_trace * state.metric[pair]);
    }
  }
}

template <int Order, class Z4cPack, class StressEnergyPack>
KOKKOS_INLINE_FUNCTION void EvaluatePointWithMatter(
    const Z4cPack& values, const StressEnergyPack& stress_energy, const int block, const int k,
    const int j, const int i, const Real inverse_spacing[3], const Z4cOptions& options,
    const Real time, Real output[kZ4cComponents]) {
  PointState state{};
  PointDerivatives derivatives{};
  LoadPoint<Order>(values, block, k, j, i, inverse_spacing, state, derivatives);
  ComputeVacuum(state, derivatives, options, time, output);
  ApplyMatterSources(state, options, LoadStressEnergy(stress_energy, block, k, j, i), output);
  for (int component = 0; component < kZ4cComponents; ++component) {
    const auto component_values = ComponentAccessor<Z4cPack>{values, block, component, k, j, i};
    for (int direction = 0; direction < 3; ++direction) {
      output[component] +=
          options.dissipation *
          fd::KreissOliger<Order>(direction, inverse_spacing[direction], component_values);
    }
  }
}

template <int Order, class Pack>
KOKKOS_INLINE_FUNCTION void EvaluatePoint(const Pack& values, const int block, const int k,
                                          const int j, const int i, const Real inverse_spacing[3],
                                          const Z4cOptions& options, const Real time,
                                          Real output[kZ4cComponents]) {
  PointState state{};
  PointDerivatives derivatives{};
  LoadPoint<Order>(values, block, k, j, i, inverse_spacing, state, derivatives);
  ComputeVacuum(state, derivatives, options, time, output);
  for (int component = 0; component < kZ4cComponents; ++component) {
    const auto component_values = ComponentAccessor<Pack>{values, block, component, k, j, i};
    for (int direction = 0; direction < 3; ++direction) {
      output[component] +=
          options.dissipation *
          fd::KreissOliger<Order>(direction, inverse_spacing[direction], component_values);
    }
  }
}

} // namespace rhs
} // namespace pangu::nr

#endif
