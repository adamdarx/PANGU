#ifndef PANGU_Z4C_ADM_CONVERSION_H_
#define PANGU_Z4C_ADM_CONVERSION_H_

#include <cmath>

#include <parthenon/parthenon.hpp>

#include "z4c/component_indices.h"
#include "z4c/z4c_rhs.h"

namespace pangu::nr {

// Cell-centered ADM data used by the conversion boundary. The lapse, shift and
// Theta are kept here because they are part of the geometric state although the
// derived nr.adm field stores only the spatial metric, K_ij and psi^4.
struct ADMState {
  parthenon::Real metric[6]{};
  parthenon::Real extrinsic[6]{};
  parthenon::Real lapse = 0.0;
  parthenon::Real shift[3]{};
  parthenon::Real theta = 0.0;
};

struct ADMMetricDerivatives {
  parthenon::Real dmetric[3][6]{};
};

struct Z4cState {
  parthenon::Real values[kZ4cComponents]{};
};

KOKKOS_INLINE_FUNCTION bool ConvertZ4cToADM(const Z4cState& input,
                                            const parthenon::Real chi_psi_power, ADMState& output) {
  using Real = parthenon::Real;
  const Real chi = input.values[Index(Z4cComponent::chi)];
  if (!(chi > 0.0) || chi_psi_power == 0.0)
    return false;

  // chi = psi^chi_psi_power, hence psi^4 = chi^(4/chi_psi_power).
  const Real psi4 = pow(chi, 4.0 / chi_psi_power);
  const Real trace_k =
      input.values[Index(Z4cComponent::khat)] + 2.0 * input.values[Index(Z4cComponent::theta)];
  output.lapse = input.values[Index(Z4cComponent::alpha)];
  output.theta = input.values[Index(Z4cComponent::theta)];
  for (int axis = 0; axis < 3; ++axis)
    output.shift[axis] = input.values[Index(Z4cComponent::betax) + axis];
  for (int component = 0; component < 6; ++component) {
    const Real conformal_metric = input.values[Index(Z4cComponent::gxx) + component];
    const Real conformal_a = input.values[Index(Z4cComponent::axx) + component];
    output.metric[component] = psi4 * conformal_metric;
    output.extrinsic[component] = psi4 * (conformal_a + conformal_metric * trace_k / 3.0);
  }
  return output.lapse > 0.0;
}

KOKKOS_INLINE_FUNCTION bool ConvertADMToZ4c(const ADMState& input,
                                            const ADMMetricDerivatives& derivatives,
                                            const parthenon::Real chi_psi_power, Z4cState& output) {
  using Real = parthenon::Real;
  if (chi_psi_power == 0.0)
    return false;

  const Real determinant = rhs::SpatialDeterminant(input.metric);
  if (!(determinant > 0.0))
    return false;
  Real inverse_metric[6]{};
  rhs::SpatialInverse(1.0 / determinant, input.metric, inverse_metric);

  const Real chi = pow(determinant, chi_psi_power / 12.0);
  if (!(chi > 0.0))
    return false;
  Real conformal_metric[6]{};
  for (int component = 0; component < 6; ++component)
    conformal_metric[component] = chi * input.metric[component];
  const Real conformal_determinant = rhs::SpatialDeterminant(conformal_metric);
  if (!(conformal_determinant > 0.0))
    return false;
  Real inverse_conformal[6]{};
  rhs::SpatialInverse(1.0 / conformal_determinant, conformal_metric, inverse_conformal);

  Real trace_k = 0.0;
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      trace_k += rhs::SymmetricValue(inverse_metric, first, second) *
                 rhs::SymmetricValue(input.extrinsic, first, second);
    }
  }

  for (int component = 0; component < kZ4cComponents; ++component)
    output.values[component] = 0.0;
  output.values[Index(Z4cComponent::chi)] = chi;
  output.values[Index(Z4cComponent::khat)] = trace_k - 2.0 * input.theta;
  output.values[Index(Z4cComponent::theta)] = input.theta;
  output.values[Index(Z4cComponent::alpha)] = input.lapse;
  for (int axis = 0; axis < 3; ++axis)
    output.values[Index(Z4cComponent::betax) + axis] = input.shift[axis];

  for (int component = 0; component < 6; ++component) {
    output.values[Index(Z4cComponent::gxx) + component] = conformal_metric[component];
    output.values[Index(Z4cComponent::axx) + component] =
        chi * (input.extrinsic[component] - input.metric[component] * trace_k / 3.0);
  }

  // Gamma-tilde^i = -d_j gamma-tilde^{ij}. Deriving it from the physical
  // metric derivative keeps ADM->Z4c independent of a pre-existing Gamma field.
  Real d_conformal_inverse[3][6]{};
  for (int derivative = 0; derivative < 3; ++derivative) {
    Real d_conformal_metric[6]{};
    Real d_log_determinant = 0.0;
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
        d_log_determinant += rhs::SymmetricValue(inverse_metric, first, second) *
                             rhs::SymmetricValue(derivatives.dmetric[derivative], first, second);
      }
    }
    const Real dchi = chi * (chi_psi_power / 12.0) * d_log_determinant;
    for (int component = 0; component < 6; ++component) {
      d_conformal_metric[component] =
          dchi * input.metric[component] + chi * derivatives.dmetric[derivative][component];
    }
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        const int pair = SpatialSymmetricComponent(first, second);
        Real derivative_value = 0.0;
        for (int raised_first = 0; raised_first < 3; ++raised_first) {
          for (int raised_second = 0; raised_second < 3; ++raised_second) {
            derivative_value -=
                rhs::SymmetricValue(inverse_conformal, first, raised_first) *
                rhs::SymmetricValue(inverse_conformal, second, raised_second) *
                rhs::SymmetricValue(d_conformal_metric, raised_first, raised_second);
          }
        }
        d_conformal_inverse[derivative][pair] = derivative_value;
      }
    }
  }
  for (int upper = 0; upper < 3; ++upper) {
    Real contracted = 0.0;
    for (int derivative = 0; derivative < 3; ++derivative)
      contracted -= d_conformal_inverse[derivative][SpatialSymmetricComponent(upper, derivative)];
    output.values[Index(Z4cComponent::gamx) + upper] = contracted;
  }
  return input.lapse > 0.0;
}

} // namespace pangu::nr

#endif
