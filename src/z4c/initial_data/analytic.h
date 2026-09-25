#ifndef PANGU_Z4C_ANALYTIC_INITIAL_DATA_H_
#define PANGU_Z4C_ANALYTIC_INITIAL_DATA_H_

#include <parthenon/parthenon.hpp>

#include "z4c/core/adm_conversion.h"
#include "z4c/evolution/constraints.h"

namespace pangu::nr {

// Time-symmetric Schwarzschild data in isotropic Cartesian coordinates. The
// slice is conformally flat, K_ij=0, and the vacuum constraints vanish away
// from the puncture. The helper carries both derivatives needed by the ADM
// conversion and the second derivatives needed by the constraint contraction.
struct SchwarzschildIsotropicData {
  ADMState adm{};
  ADMMetricDerivatives derivatives{};
  ADMConstraintInput constraints{};
};

// Ingoing Schwarzschild Kerr--Schild data in Cartesian coordinates.  The
// spatial slice is regular at the horizon (but not at r=0), making it a useful
// one-hole initial-data test independent of puncture conformal factors.
struct SchwarzschildKerrSchildData {
  ADMState adm{};
  ADMMetricDerivatives derivatives{};
  ADMConstraintInput constraints{};
};

// Single boosted puncture data in the conformally flat construction used by
// AthenaK's z4c_boosted_puncture test (Cook, Living Rev. Rel. 3, 5 (2000),
// and the implementation notes cited by that test).  The first NR-5 slice
// intentionally restricts the boost to the x direction so that the discrete
// reference problem is unambiguous; transverse boosts are rejected by the
// problem generator until the corresponding rotated extrinsic-curvature
// expressions are added.
struct BoostedPunctureData {
  ADMState adm{};
};

KOKKOS_INLINE_FUNCTION bool BuildBoostedPuncture(const parthenon::Real x,
                                                const parthenon::Real y,
                                                const parthenon::Real z,
                                                const parthenon::Real mass,
                                                const parthenon::Real velocity_x,
                                                BoostedPunctureData& output) {
  using Real = parthenon::Real;
  const Real radius_squared = x * x + y * y + z * z;
  const Real radius = sqrt(radius_squared);
  const Real speed_squared = velocity_x * velocity_x;
  if (!(mass > 0.0) || !(radius > 0.0) || !(speed_squared < 1.0)) return false;

  output = BoostedPunctureData{};
  const Real lorentz_factor = 1.0 / sqrt(1.0 - speed_squared);
  const Real comoving_x = lorentz_factor * x;
  const Real comoving_radius = sqrt(comoving_x * comoving_x + y * y + z * z);
  if (!(comoving_radius > 0.0)) return false;

  const Real half_mass = 0.5 * mass;
  const Real psi = 1.0 + half_mass / comoving_radius;
  const Real psi4 = psi * psi * psi * psi;
  const Real alpha_isotropic = (1.0 - half_mass / comoving_radius) / psi;
  const Real psi_inverse_fourth = 1.0 / psi4;
  const Real boost_factor = sqrt(lorentz_factor * lorentz_factor *
                                 (1.0 - speed_squared * alpha_isotropic * alpha_isotropic *
                                           psi_inverse_fourth));
  if (!(boost_factor > 0.0) || !Kokkos::isfinite(boost_factor)) return false;

  output.adm.metric[SpatialSymmetricComponent(0, 0)] = psi4 * boost_factor * boost_factor;
  output.adm.metric[SpatialSymmetricComponent(1, 1)] = psi4;
  output.adm.metric[SpatialSymmetricComponent(2, 2)] = psi4;
  output.adm.shift[0] =
      (alpha_isotropic * alpha_isotropic - psi4) * velocity_x /
      (psi4 - alpha_isotropic * alpha_isotropic * speed_squared);
  output.adm.shift[1] = 0.0;
  output.adm.shift[2] = 0.0;
  // Match AthenaK's GaugePreCollapsedLapse call used by the moving-puncture
  // test: alpha = psi^{-2}, rather than the isotropic initial lapse.
  output.adm.lapse = 1.0 / (psi * psi);

  const Real alpha_prime = 4.0 * mass / ((mass + 2.0 * comoving_radius) *
                                         (mass + 2.0 * comoving_radius));
  const Real m_minus_2r = mass - 2.0 * comoving_radius;
  const Real m_plus_2r = mass + 2.0 * comoving_radius;
  const Real second_term =
      ((4.0 * speed_squared * m_minus_2r * m_minus_2r / (m_plus_2r * m_plus_2r * m_plus_2r)) +
       (4.0 * speed_squared * m_minus_2r / (m_plus_2r * m_plus_2r)) -
       (mass * m_plus_2r * m_plus_2r * m_plus_2r /
        (4.0 * comoving_radius * comoving_radius * comoving_radius * comoving_radius *
         comoving_radius))) /
      (psi * psi * psi * psi - speed_squared * m_minus_2r * m_minus_2r /
                                  (m_plus_2r * m_plus_2r));
  const Real common = alpha_prime - 0.5 * alpha_isotropic * second_term;
  output.adm.extrinsic[SpatialSymmetricComponent(0, 0)] =
      lorentz_factor * lorentz_factor * boost_factor * x * velocity_x / comoving_radius *
      (2.0 * alpha_prime - 0.5 * alpha_isotropic * second_term);
  output.adm.extrinsic[SpatialSymmetricComponent(1, 1)] =
      2.0 * lorentz_factor * lorentz_factor * x * velocity_x * alpha_isotropic *
      (-mass / (2.0 * comoving_radius * comoving_radius)) /
      (psi * boost_factor * comoving_radius);
  output.adm.extrinsic[SpatialSymmetricComponent(2, 2)] =
      output.adm.extrinsic[SpatialSymmetricComponent(1, 1)];
  output.adm.extrinsic[SpatialSymmetricComponent(0, 1)] =
      boost_factor * y * velocity_x / comoving_radius * common;
  output.adm.extrinsic[SpatialSymmetricComponent(0, 2)] =
      boost_factor * z * velocity_x / comoving_radius * common;
  output.adm.extrinsic[SpatialSymmetricComponent(1, 2)] = 0.0;
  return true;
}

KOKKOS_INLINE_FUNCTION bool BuildSchwarzschildIsotropic(const parthenon::Real x,
                                                        const parthenon::Real y,
                                                        const parthenon::Real z,
                                                        const parthenon::Real mass,
                                                        const parthenon::Real chi_psi_power,
                                                        SchwarzschildIsotropicData& output) {
  using Real = parthenon::Real;
  const Real radius_squared = x * x + y * y + z * z;
  const Real radius = sqrt(radius_squared);
  const Real half_mass = 0.5 * mass;
  if (!(mass >= 0.0) || !(radius > 0.0) || chi_psi_power == 0.0)
    return false;

  output = SchwarzschildIsotropicData{};
  const Real inverse_radius = 1.0 / radius;
  const Real psi = 1.0 + half_mass * inverse_radius;
  const Real conformal_factor = pow(psi, 4.0);
  const Real determinant = conformal_factor * conformal_factor * conformal_factor;
  const Real chi = pow(determinant, chi_psi_power / 12.0);
  const Real lapse = (1.0 - half_mass * inverse_radius) / psi;
  const Real coordinates[3] = {x, y, z};
  Real dpsi[3]{};
  Real ddpsi[6]{};
  for (int axis = 0; axis < 3; ++axis)
    dpsi[axis] = -half_mass * coordinates[axis] * inverse_radius * inverse_radius * inverse_radius;
  const Real inverse_radius_cubed = inverse_radius * inverse_radius * inverse_radius;
  const Real inverse_radius_fifth = inverse_radius_cubed * inverse_radius * inverse_radius;
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const Real kronecker = first == second ? 1.0 : 0.0;
      ddpsi[SpatialSymmetricComponent(first, second)] =
          -half_mass * (kronecker * inverse_radius_cubed -
                        3.0 * coordinates[first] * coordinates[second] * inverse_radius_fifth);
    }
  }

  Real dconformal[3]{};
  Real ddconformal[6]{};
  for (int derivative = 0; derivative < 3; ++derivative)
    dconformal[derivative] = 4.0 * psi * psi * psi * dpsi[derivative];
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      ddconformal[pair] =
          12.0 * psi * psi * dpsi[first] * dpsi[second] + 4.0 * psi * psi * psi * ddpsi[pair];
    }
  }

  output.adm.lapse = lapse;
  for (int component = 0; component < 6; ++component) {
    output.adm.metric[component] = 0.0;
    output.adm.extrinsic[component] = 0.0;
  }
  output.adm.metric[0] = output.adm.metric[3] = output.adm.metric[5] = conformal_factor;
  output.derivatives = ADMMetricDerivatives{};
  output.constraints = ADMConstraintInput{};
  for (int axis = 0; axis < 3; ++axis) {
    const int diagonal = SpatialSymmetricComponent(axis, axis);
    for (int derivative = 0; derivative < 3; ++derivative)
      output.derivatives.dmetric[derivative][diagonal] = dconformal[derivative];
  }
  output.constraints.metric[0] = output.constraints.metric[3] = output.constraints.metric[5] =
      conformal_factor;
  for (int derivative = 0; derivative < 3; ++derivative) {
    for (int component = 0; component < 6; ++component)
      output.constraints.dmetric[derivative][component] =
          output.derivatives.dmetric[derivative][component];
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int pair = SpatialSymmetricComponent(first, second);
      for (int axis = 0; axis < 3; ++axis) {
        const int diagonal = SpatialSymmetricComponent(axis, axis);
        output.constraints.ddmetric[pair][diagonal] = ddconformal[pair];
      }
    }
  }
  return true;
}

KOKKOS_INLINE_FUNCTION bool BuildSchwarzschildKerrSchild(const parthenon::Real x,
                                                         const parthenon::Real y,
                                                         const parthenon::Real z,
                                                         const parthenon::Real mass,
                                                         SchwarzschildKerrSchildData& output) {
  using Real = parthenon::Real;
  const Real radius_squared = x * x + y * y + z * z;
  const Real radius = sqrt(radius_squared);
  if (!(mass >= 0.0) || !(radius > 0.0))
    return false;

  output = SchwarzschildKerrSchildData{};
  const Real inverse_radius = 1.0 / radius;
  const Real q = 2.0 * mass * inverse_radius;
  const Real inverse_radius_squared = inverse_radius * inverse_radius;
  const Real n[3] = {x * inverse_radius, y * inverse_radius, z * inverse_radius};
  Real projector[3][3]{};
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      projector[first][second] = (first == second ? 1.0 : 0.0) - n[first] * n[second];

  // Derivatives of q=2M/r and n_i=x_i/r.
  Real dq[3]{};
  Real ddq[3][3]{};
  Real dn[3][3]{};
  Real ddn[3][3][3]{};
  for (int axis = 0; axis < 3; ++axis) {
    dq[axis] = -q * n[axis] * inverse_radius;
    for (int component = 0; component < 3; ++component)
      dn[axis][component] = projector[axis][component] * inverse_radius;
  }
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      ddq[first][second] =
          q * (3.0 * n[first] * n[second] - (first == second ? 1.0 : 0.0)) * inverse_radius_squared;
      for (int component = 0; component < 3; ++component)
        ddn[first][second][component] =
            -(projector[first][second] * n[component] + n[first] * projector[second][component] +
              projector[first][component] * n[second]) *
            inverse_radius_squared;
    }
  }

  Real metric[6]{};
  Real dmetric[3][6]{};
  Real ddmetric[6][6]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int component = SpatialSymmetricComponent(first, second);
      metric[component] = (first == second ? 1.0 : 0.0) + q * n[first] * n[second];
      for (int derivative = 0; derivative < 3; ++derivative) {
        dmetric[derivative][component] =
            dq[derivative] * n[first] * n[second] +
            q * (dn[derivative][first] * n[second] + n[first] * dn[derivative][second]);
        for (int second_derivative = derivative; second_derivative < 3; ++second_derivative) {
          const int derivative_pair = SpatialSymmetricComponent(derivative, second_derivative);
          ddmetric[derivative_pair][component] =
              ddq[derivative][second_derivative] * n[first] * n[second] +
              dq[derivative] * (dn[second_derivative][first] * n[second] +
                                n[first] * dn[second_derivative][second]) +
              dq[second_derivative] *
                  (dn[derivative][first] * n[second] + n[first] * dn[derivative][second]) +
              q * (ddn[derivative][second_derivative][first] * n[second] +
                   dn[derivative][first] * dn[second_derivative][second] +
                   dn[second_derivative][first] * dn[derivative][second] +
                   n[first] * ddn[derivative][second_derivative][second]);
        }
      }
    }
  }

  // For the ingoing Kerr--Schild sign convention, beta_i=q n_i and
  // beta^i=q n^i/(1+q), alpha=(1+q)^(-1/2).
  Real beta_cov[3]{};
  Real dbeta[3][3]{};
  Real ddbeta[3][3][3]{};
  for (int component = 0; component < 3; ++component)
    beta_cov[component] = q * n[component];
  for (int derivative = 0; derivative < 3; ++derivative) {
    for (int component = 0; component < 3; ++component) {
      dbeta[derivative][component] = dq[derivative] * n[component] + q * dn[derivative][component];
      for (int second_derivative = 0; second_derivative < 3; ++second_derivative)
        ddbeta[derivative][second_derivative][component] =
            ddq[derivative][second_derivative] * n[component] +
            dq[derivative] * dn[second_derivative][component] +
            dq[second_derivative] * dn[derivative][component] +
            q * ddn[derivative][second_derivative][component];
    }
  }

  Real inverse_metric[6]{};
  rhs::SpatialInverse(1.0 / rhs::SpatialDeterminant(metric), metric, inverse_metric);
  Real dinverse[3][6]{};
  for (int derivative = 0; derivative < 3; ++derivative) {
    for (int first = 0; first < 3; ++first) {
      for (int second = first; second < 3; ++second) {
        Real value = 0.0;
        for (int raised_first = 0; raised_first < 3; ++raised_first)
          for (int raised_second = 0; raised_second < 3; ++raised_second)
            value -= rhs::SymmetricValue(inverse_metric, first, raised_first) *
                     rhs::SymmetricValue(inverse_metric, second, raised_second) *
                     dmetric[derivative][SpatialSymmetricComponent(raised_first, raised_second)];
        dinverse[derivative][SpatialSymmetricComponent(first, second)] = value;
      }
    }
  }

  Real christoffel[3][3][3]{};
  Real dchristoffel[3][3][3][3]{};
  for (int upper = 0; upper < 3; ++upper) {
    for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
        for (int contracted = 0; contracted < 3; ++contracted) {
          const int first_pair = SpatialSymmetricComponent(first, contracted);
          const int second_pair = SpatialSymmetricComponent(second, contracted);
          const int contracted_pair = SpatialSymmetricComponent(first, second);
          const Real connection = dmetric[first][second_pair] + dmetric[second][first_pair] -
                                  dmetric[contracted][contracted_pair];
          christoffel[upper][first][second] +=
              0.5 * rhs::SymmetricValue(inverse_metric, upper, contracted) * connection;
          for (int derivative = 0; derivative < 3; ++derivative) {
            const int derivative_first = SpatialSymmetricComponent(derivative, first);
            const int derivative_second = SpatialSymmetricComponent(derivative, second);
            const int derivative_contracted = SpatialSymmetricComponent(derivative, contracted);
            const Real dconnection = ddmetric[derivative_first][second_pair] +
                                     ddmetric[derivative_second][first_pair] -
                                     ddmetric[derivative_contracted][contracted_pair];
            dchristoffel[derivative][upper][first][second] +=
                0.5 * dinverse[derivative][SpatialSymmetricComponent(upper, contracted)] *
                    connection +
                0.5 * rhs::SymmetricValue(inverse_metric, upper, contracted) * dconnection;
          }
        }
      }
    }
  }

  const Real lapse = 1.0 / sqrt(1.0 + q);
  const Real inverse_lapse = 1.0 / lapse;
  const Real factor = 0.5 * inverse_lapse;
  Real dfactor[3]{};
  for (int derivative = 0; derivative < 3; ++derivative)
    dfactor[derivative] = 0.25 * dq[derivative] * lapse;
  Real extrinsic[6]{};
  Real dextrinsic[3][6]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = first; second < 3; ++second) {
      const int component = SpatialSymmetricComponent(first, second);
      Real covariant_sum = dbeta[first][second] + dbeta[second][first];
      for (int upper = 0; upper < 3; ++upper)
        covariant_sum -= 2.0 * christoffel[upper][first][second] * beta_cov[upper];
      extrinsic[component] = factor * covariant_sum;
      for (int derivative = 0; derivative < 3; ++derivative) {
        Real derivative_sum = ddbeta[derivative][first][second] + ddbeta[derivative][second][first];
        for (int upper = 0; upper < 3; ++upper)
          derivative_sum -=
              2.0 * (dchristoffel[derivative][upper][first][second] * beta_cov[upper] +
                     christoffel[upper][first][second] * dbeta[derivative][upper]);
        dextrinsic[derivative][component] =
            dfactor[derivative] * covariant_sum + factor * derivative_sum;
      }
    }
  }

  output.adm.lapse = lapse;
  for (int axis = 0; axis < 3; ++axis)
    output.adm.shift[axis] = q * n[axis] / (1.0 + q);
  for (int component = 0; component < 6; ++component) {
    output.adm.metric[component] = metric[component];
    output.adm.extrinsic[component] = extrinsic[component];
    output.constraints.metric[component] = metric[component];
    output.constraints.extrinsic[component] = extrinsic[component];
    for (int derivative = 0; derivative < 3; ++derivative) {
      output.derivatives.dmetric[derivative][component] = dmetric[derivative][component];
      output.constraints.dmetric[derivative][component] = dmetric[derivative][component];
      output.constraints.dextrinsic[derivative][component] = dextrinsic[derivative][component];
    }
    for (int derivative_pair = 0; derivative_pair < 6; ++derivative_pair)
      output.constraints.ddmetric[derivative_pair][component] =
          ddmetric[derivative_pair][component];
  }
  output.constraints.energy = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    output.constraints.momentum[axis] = 0.0;
    output.constraints.z[axis] = 0.0;
  }
  return true;
}

} // namespace pangu::nr

#endif
