#ifndef PANGU_Z4C_NR_DIAGNOSTICS_H_
#define PANGU_Z4C_NR_DIAGNOSTICS_H_

#include <cmath>

#include <parthenon/parthenon.hpp>

#include "z4c/evolution/z4c_rhs.h"

namespace pangu::nr {

struct ComplexValue {
  parthenon::Real real = 0.0;
  parthenon::Real imag = 0.0;
};

// AthenaK's spin-weight -2 convention, specialized to the three l=2 modes
// required by NR-6.  The waveform projection multiplies Psi4 by conj(Y_lm).
KOKKOS_INLINE_FUNCTION ComplexValue SpinMinusTwoY2(const int m, const parthenon::Real theta,
                                                   const parthenon::Real phi) {
  using Real = parthenon::Real;
  constexpr Real pi = 3.141592653589793238462643383279502884;
  const Real cosine = cos(theta);
  Real amplitude = 0.0;
  if (m == 2) {
    amplitude = sqrt(5.0 / (64.0 * pi)) * (1.0 + cosine) * (1.0 + cosine);
  } else if (m == 0) {
    amplitude = sqrt(15.0 / (32.0 * pi)) * (1.0 - cosine * cosine);
  } else if (m == -2) {
    amplitude = sqrt(5.0 / (64.0 * pi)) * (1.0 - cosine) * (1.0 - cosine);
  } else {
    return {};
  }
  return {amplitude * cos(static_cast<Real>(m) * phi), amplitude * sin(static_cast<Real>(m) * phi)};
}

struct HorizonPoint {
  parthenon::Real expansion = 0.0;
  parthenon::Real area_jacobian = 0.0;
  parthenon::Real flow = 0.0; // AthenaK standard flow: H |grad F|
  parthenon::Real expansion_flow = 0.0;
  parthenon::Real shear_flow = 0.0;
  parthenon::Real normal_x = 0.0;
  parthenon::Real normal_y = 0.0;
  parthenon::Real normal_z = 0.0;
};

// Expansion of a star-shaped surface F=r-h(theta,phi)=0.  The radius and its
// angular derivatives are supplied by the real spherical-harmonic surface
// representation. metric_derivative[d][a][b] is partial_d gamma_ab.
KOKKOS_INLINE_FUNCTION HorizonPoint StarShapedSurfaceExpansion(
    const parthenon::Real metric[3][3], const parthenon::Real metric_derivative[3][3][3],
    const parthenon::Real extrinsic[3][3], const parthenon::Real radius,
    const parthenon::Real radius_theta, const parthenon::Real radius_phi,
    const parthenon::Real radius_theta_theta, const parthenon::Real radius_theta_phi,
    const parthenon::Real radius_phi_phi, const parthenon::Real theta, const parthenon::Real phi) {
  using Real = parthenon::Real;
  if (!(radius > 0.0))
    return {};

  Real symmetric[6]{};
  for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b)
      symmetric[SpatialSymmetricComponent(a, b)] = metric[a][b];
  const Real determinant = rhs::SpatialDeterminant(symmetric);
  if (!(determinant > 0.0))
    return {};
  Real inverse_symmetric[6]{};
  rhs::SpatialInverse(1.0 / determinant, symmetric, inverse_symmetric);
  Real inverse[3][3]{};
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      inverse[a][b] = rhs::SymmetricValue(inverse_symmetric, a, b);

  const Real sin_theta = sin(theta);
  const Real cos_theta = cos(theta);
  const Real sin_phi = sin(phi);
  const Real cos_phi = cos(phi);
  const Real x = radius * sin_theta * cos_phi;
  const Real y = radius * sin_theta * sin_phi;
  const Real z = radius * cos_theta;
  const Real cylindrical_radius = sqrt(x * x + y * y);
  if (!(cylindrical_radius > 0.0))
    return {};

  const Real inverse_radius = 1.0 / radius;
  const Real inverse_radius_squared = inverse_radius * inverse_radius;
  const Real inverse_radius_cubed = inverse_radius_squared * inverse_radius;
  const Real inverse_cylindrical = 1.0 / cylindrical_radius;
  const Real inverse_cylindrical_squared = inverse_cylindrical * inverse_cylindrical;
  const Real inverse_cylindrical_cubed = inverse_cylindrical_squared * inverse_cylindrical;

  Real dr[3] = {x * inverse_radius, y * inverse_radius, z * inverse_radius};
  Real dtheta[3] = {z * x * inverse_radius_squared * inverse_cylindrical,
                    z * y * inverse_radius_squared * inverse_cylindrical,
                    -cylindrical_radius * inverse_radius_squared};
  Real dphi[3] = {-y * inverse_cylindrical_squared, x * inverse_cylindrical_squared, 0.0};
  Real d2r[3][3]{};
  Real d2theta[3][3]{};
  Real d2phi[3][3]{};
  d2r[0][0] = inverse_radius - x * x * inverse_radius_cubed;
  d2r[0][1] = -x * y * inverse_radius_cubed;
  d2r[0][2] = -x * z * inverse_radius_cubed;
  d2r[1][1] = inverse_radius - y * y * inverse_radius_cubed;
  d2r[1][2] = -y * z * inverse_radius_cubed;
  d2r[2][2] = inverse_radius - z * z * inverse_radius_cubed;

  const Real inverse_radius_fourth = inverse_radius_squared * inverse_radius_squared;
  const Real inverse_cylindrical_fourth = inverse_cylindrical_squared * inverse_cylindrical_squared;
  d2theta[0][0] = z * (-2.0 * x * x * x * x - x * x * y * y + y * y * y * y + z * z * y * y) *
                  inverse_radius_fourth * inverse_cylindrical_cubed;
  d2theta[0][1] = -x * y * z * (3.0 * x * x + 3.0 * y * y + z * z) * inverse_radius_fourth *
                  inverse_cylindrical_cubed;
  d2theta[0][2] = x * (x * x + y * y - z * z) * inverse_radius_fourth * inverse_cylindrical;
  d2theta[1][1] = z * (-2.0 * y * y * y * y - y * y * x * x + x * x * x * x + z * z * x * x) *
                  inverse_radius_fourth * inverse_cylindrical_cubed;
  d2theta[1][2] = y * (x * x + y * y - z * z) * inverse_radius_fourth * inverse_cylindrical;
  d2theta[2][2] = 2.0 * z * cylindrical_radius * inverse_radius_fourth;

  d2phi[0][0] = 2.0 * y * x * inverse_cylindrical_fourth;
  d2phi[0][1] = (y * y - x * x) * inverse_cylindrical_fourth;
  d2phi[1][1] = -2.0 * y * x * inverse_cylindrical_fourth;
  for (int a = 0; a < 3; ++a) {
    for (int b = a + 1; b < 3; ++b) {
      d2r[b][a] = d2r[a][b];
      d2theta[b][a] = d2theta[a][b];
      d2phi[b][a] = d2phi[a][b];
    }
  }

  Real gradient[3]{};
  Real hessian[3][3]{};
  for (int a = 0; a < 3; ++a) {
    gradient[a] = dr[a] - radius_theta * dtheta[a] - radius_phi * dphi[a];
    for (int b = 0; b < 3; ++b) {
      hessian[a][b] = d2r[a][b] - radius_theta * d2theta[a][b] - radius_phi * d2phi[a][b] -
                      radius_theta_theta * dtheta[a] * dtheta[b] -
                      radius_theta_phi * (dtheta[a] * dphi[b] + dphi[a] * dtheta[b]) -
                      radius_phi_phi * dphi[a] * dphi[b];
    }
  }

  Real raised_gradient[3]{};
  Real gradient_norm_squared = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b)
      raised_gradient[a] += inverse[a][b] * gradient[b];
    gradient_norm_squared += raised_gradient[a] * gradient[a];
  }
  if (!(gradient_norm_squared > 0.0))
    return {};
  const Real gradient_norm = sqrt(gradient_norm_squared);
  const Real inverse_norm = 1.0 / gradient_norm;

  Real covariant_hessian[3][3]{};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      covariant_hessian[a][b] = hessian[a][b];
      for (int d = 0; d < 3; ++d)
        covariant_hessian[a][b] -=
            0.5 * raised_gradient[d] *
            (metric_derivative[a][b][d] + metric_derivative[b][a][d] - metric_derivative[d][a][b]);
    }
  }

  Real trace_k = 0.0;
  Real laplacian = 0.0;
  Real normal_k_normal = 0.0;
  Real normal_hessian_normal = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      trace_k += inverse[a][b] * extrinsic[a][b];
      laplacian += inverse[a][b] * covariant_hessian[a][b];
      normal_k_normal += raised_gradient[a] * raised_gradient[b] * extrinsic[a][b];
      normal_hessian_normal += raised_gradient[a] * raised_gradient[b] * covariant_hessian[a][b];
    }
  }
  const Real expansion = laplacian * inverse_norm + normal_k_normal * inverse_norm * inverse_norm -
                         normal_hessian_normal * inverse_norm * inverse_norm * inverse_norm -
                         trace_k;

  const Real tangent_theta[3] = {(radius_theta * sin_theta + radius * cos_theta) * cos_phi,
                                 (radius_theta * sin_theta + radius * cos_theta) * sin_phi,
                                 radius_theta * cos_theta - radius * sin_theta};
  const Real tangent_phi[3] = {(radius_phi * cos_phi - radius * sin_phi) * sin_theta,
                               (radius_phi * sin_phi + radius * cos_phi) * sin_theta,
                               radius_phi * cos_theta};
  Real q_theta_theta = 0.0;
  Real q_theta_phi = 0.0;
  Real q_phi_phi = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      q_theta_theta += metric[a][b] * tangent_theta[a] * tangent_theta[b];
      q_theta_phi += metric[a][b] * tangent_theta[a] * tangent_phi[b];
      q_phi_phi += metric[a][b] * tangent_phi[a] * tangent_phi[b];
    }
  }
  const Real area_squared = q_theta_theta * q_phi_phi - q_theta_phi * q_theta_phi;
  const Real sigma_parameter =
      inverse[0][0] + inverse[1][1] + inverse[2][2] - inverse[0][0] * dr[0] * dr[0] -
      inverse[1][1] * dr[1] * dr[1] - inverse[2][2] * dr[2] * dr[2] -
      2.0 * (inverse[0][1] * dr[0] * dr[1] + inverse[0][2] * dr[0] * dr[2] +
             inverse[1][2] * dr[1] * dr[2]) -
      raised_gradient[0] * raised_gradient[0] * inverse_norm * inverse_norm -
      raised_gradient[1] * raised_gradient[1] * inverse_norm * inverse_norm -
      raised_gradient[2] * raised_gradient[2] * inverse_norm * inverse_norm +
      raised_gradient[0] * raised_gradient[0] * inverse_norm * inverse_norm * dr[0] * dr[0] +
      raised_gradient[1] * raised_gradient[1] * inverse_norm * inverse_norm * dr[1] * dr[1] +
      raised_gradient[2] * raised_gradient[2] * inverse_norm * inverse_norm * dr[2] * dr[2] +
      2.0 * inverse_norm * inverse_norm *
          (raised_gradient[0] * raised_gradient[1] * dr[0] * dr[1] +
           raised_gradient[0] * raised_gradient[2] * dr[0] * dr[2] +
           raised_gradient[1] * raised_gradient[2] * dr[1] * dr[2]);
  const Real shear = 2.0 * radius * radius / sigma_parameter;
  return {expansion,
          area_squared > 0.0 ? sqrt(area_squared) : 0.0,
          expansion * gradient_norm,
          expansion,
          expansion * gradient_norm * shear,
          raised_gradient[0] * inverse_norm,
          raised_gradient[1] * inverse_norm,
          raised_gradient[2] * inverse_norm};
}

// Expansion of the outward normal of a coordinate sphere.  This is the l=0
// member of the fast-flow family used as the first GPU-friendly apparent-
// horizon finder.  metric_derivative[d][a][b] is partial_d gamma_ab.
KOKKOS_INLINE_FUNCTION HorizonPoint CoordinateSphereExpansion(
    const parthenon::Real metric[3][3], const parthenon::Real metric_derivative[3][3][3],
    const parthenon::Real extrinsic[3][3], const parthenon::Real radius,
    const parthenon::Real theta, const parthenon::Real phi) {
  return StarShapedSurfaceExpansion(metric, metric_derivative, extrinsic, radius, 0.0, 0.0, 0.0,
                                    0.0, 0.0, theta, phi);
}

void RunPostStepDiagnostics(parthenon::Mesh* mesh, parthenon::Real time, int cycle);

} // namespace pangu::nr

#endif
