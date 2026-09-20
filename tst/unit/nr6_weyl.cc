#include <algorithm>
#include <cmath>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "z4c/component_indices.h"
#include "z4c/nr_diagnostics.h"
#include "z4c/weyl.h"

using pangu::nr::ADMComponent;
using pangu::nr::ComputeWeylScalars;
using pangu::nr::Index;
using pangu::nr::SpatialSymmetricComponent;
using parthenon::Real;

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  constexpr int extent = 9;
  constexpr int center = extent / 2;
  Kokkos::View<Real*****> adm("NR-6 ADM fixture", 2, pangu::nr::kADMComponents, extent, extent,
                              extent);
  Kokkos::View<Real* [2]> errors("NR-6 Weyl errors", 2);
  Kokkos::View<Real*> diagnostic_errors("NR-6 diagnostic errors", 6);
  Kokkos::parallel_for(
      "NR-6 Weyl scalar fixtures", 2, KOKKOS_LAMBDA(const int test) {
        for (int k = 0; k < extent; ++k) {
          const Real z = static_cast<Real>(k - center);
          for (int j = 0; j < extent; ++j) {
            for (int i = 0; i < extent; ++i) {
              adm(test, Index(ADMComponent::gxx), k, j, i) = 1.0;
              adm(test, Index(ADMComponent::gyy), k, j, i) = 1.0;
              adm(test, Index(ADMComponent::gzz), k, j, i) = 1.0;
              if (test == 1) {
                constexpr Real amplitude = 0.02;
                adm(test, Index(ADMComponent::gxx), k, j, i) += amplitude * z * z;
                adm(test, Index(ADMComponent::gyy), k, j, i) -= amplitude * z * z;
              }
            }
          }
        }
        const Real inverse_spacing[3] = {1.0, 1.0, 1.0};
        const auto value = ComputeWeylScalars<2>(adm, test, center, center, center, inverse_spacing,
                                                 1.0, 0.0, 0.0);
        const Real expected_real = test == 0 ? 0.0 : 0.01;
        errors(test, 0) = fabs(value.real - expected_real);
        errors(test, 1) = fabs(value.imag);
      });
  Kokkos::parallel_for(
      "NR-6 waveform and horizon fixtures", 1, KOKKOS_LAMBDA(const int) {
        constexpr Real pi = 3.141592653589793238462643383279502884;
        const auto north = pangu::nr::SpinMinusTwoY2(2, 0.0, 0.37);
        diagnostic_errors(0) = fabs(north.real - sqrt(5.0 / (4.0 * pi)) * cos(0.74));
        diagnostic_errors(1) = fabs(north.imag - sqrt(5.0 / (4.0 * pi)) * sin(0.74));
        const auto equator = pangu::nr::SpinMinusTwoY2(0, 0.5 * pi, 1.2);
        diagnostic_errors(2) = fabs(equator.real - sqrt(15.0 / (32.0 * pi)));
        Real metric[3][3]{};
        Real derivative[3][3][3]{};
        Real extrinsic[3][3]{};
        for (int axis = 0; axis < 3; ++axis)
          metric[axis][axis] = 1.0;
        constexpr Real radius = 2.0;
        constexpr Real theta = 0.9;
        const auto horizon =
            pangu::nr::CoordinateSphereExpansion(metric, derivative, extrinsic, radius, theta, 1.1);
        diagnostic_errors(3) = fabs(horizon.expansion - 2.0 / radius);
        diagnostic_errors(4) = fabs(horizon.area_jacobian - radius * radius * sin(theta));

        // Independent Euclidean ellipsoid check for the angular-derivative
        // terms in F=r-h(theta,phi).  The reference is div(grad(G)/|grad(G)|)
        // for G=x^2/a^2+y^2/b^2+z^2/c^2-1.
        constexpr Real a = 1.2;
        constexpr Real b = 0.8;
        constexpr Real c = 1.0;
        constexpr Real ellipsoid_theta = 0.9;
        constexpr Real ellipsoid_phi = 1.1;
        const Real st = sin(ellipsoid_theta);
        const Real ct = cos(ellipsoid_theta);
        const Real sp = sin(ellipsoid_phi);
        const Real cp = cos(ellipsoid_phi);
        const Real inverse_a2 = 1.0 / (a * a);
        const Real inverse_b2 = 1.0 / (b * b);
        const Real inverse_c2 = 1.0 / (c * c);
        const Real angular = cp * cp * inverse_a2 + sp * sp * inverse_b2;
        const Real delta = inverse_b2 - inverse_a2;
        const Real q = st * st * angular + ct * ct * inverse_c2;
        const Real q_theta = 2.0 * st * ct * (angular - inverse_c2);
        const Real q_phi = 2.0 * st * st * sp * cp * delta;
        const Real q_theta_theta = 2.0 * (ct * ct - st * st) * (angular - inverse_c2);
        const Real q_theta_phi = 4.0 * st * ct * sp * cp * delta;
        const Real q_phi_phi = 2.0 * st * st * (cp * cp - sp * sp) * delta;
        const Real ellipsoid_radius = 1.0 / sqrt(q);
        const Real q_minus_three_halves = 1.0 / (q * sqrt(q));
        const Real q_minus_five_halves = q_minus_three_halves / q;
        const Real ellipsoid_radius_theta = -0.5 * q_minus_three_halves * q_theta;
        const Real ellipsoid_radius_phi = -0.5 * q_minus_three_halves * q_phi;
        const Real ellipsoid_radius_theta_theta = 0.75 * q_minus_five_halves * q_theta * q_theta -
                                                  0.5 * q_minus_three_halves * q_theta_theta;
        const Real ellipsoid_radius_theta_phi =
            0.75 * q_minus_five_halves * q_theta * q_phi - 0.5 * q_minus_three_halves * q_theta_phi;
        const Real ellipsoid_radius_phi_phi =
            0.75 * q_minus_five_halves * q_phi * q_phi - 0.5 * q_minus_three_halves * q_phi_phi;
        const auto ellipsoid = pangu::nr::StarShapedSurfaceExpansion(
            metric, derivative, extrinsic, ellipsoid_radius, ellipsoid_radius_theta,
            ellipsoid_radius_phi, ellipsoid_radius_theta_theta, ellipsoid_radius_theta_phi,
            ellipsoid_radius_phi_phi, ellipsoid_theta, ellipsoid_phi);
        const Real x = ellipsoid_radius * st * cp;
        const Real y = ellipsoid_radius * st * sp;
        const Real z = ellipsoid_radius * ct;
        const Real gradient_squared =
            4.0 * (x * x * inverse_a2 * inverse_a2 + y * y * inverse_b2 * inverse_b2 +
                   z * z * inverse_c2 * inverse_c2);
        const Real gradient = sqrt(gradient_squared);
        const Real hessian_contraction = 8.0 * (x * x * inverse_a2 * inverse_a2 * inverse_a2 +
                                                y * y * inverse_b2 * inverse_b2 * inverse_b2 +
                                                z * z * inverse_c2 * inverse_c2 * inverse_c2);
        const Real expected_ellipsoid = 2.0 * (inverse_a2 + inverse_b2 + inverse_c2) / gradient -
                                        hessian_contraction / (gradient * gradient * gradient);
        diagnostic_errors(5) = fabs(ellipsoid.expansion - expected_ellipsoid);
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
  Real maximum = 0.0;
  for (int test = 0; test < 2; ++test)
    for (int component = 0; component < 2; ++component)
      maximum = std::max(maximum, host(test, component));
  const auto diagnostic_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagnostic_errors);
  for (int component = 0; component < 6; ++component)
    maximum = std::max(maximum, diagnostic_host(component));
  if (maximum > 2.0e-13) {
    std::cerr << "NR-6 Weyl scalar mismatch: " << maximum << '\n';
    return 1;
  }
  std::cout << "NR-6 Weyl/waveform/horizon GPU fixtures: PASS; max_error=" << maximum << '\n';
  return 0;
}
