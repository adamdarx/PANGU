#ifndef PANGU_PGEN_RELATIVISTIC_MHD_WAVE_H_
#define PANGU_PGEN_RELATIVISTIC_MHD_WAVE_H_

#include <algorithm>
#include <cmath>
#include <limits>

#include <parthenon/parthenon.hpp>

#include "eos/ideal_gas.h"

namespace pangu::pgen {

// Relativistic ideal-MHD eigenmode used by the AthenaK dynamic-GRMHD linear-wave
// regression.  The algebra follows the Anile eigenvectors used by that public
// reference so initialization does not contaminate the cross-code evolution
// comparison.
struct RelativisticMHDWave {
  parthenon::Real density = 0.0;
  parthenon::Real pressure = 0.0;
  parthenon::Real velocity[3]{};
  parthenon::Real magnetic[3]{};
  parthenon::Real four_velocity[4]{};
  parthenon::Real comoving_magnetic[4]{};
  parthenon::Real delta_density = 0.0;
  parthenon::Real delta_pressure = 0.0;
  parthenon::Real delta_four_velocity[4]{};
  parthenon::Real delta_comoving_magnetic[4]{};
  parthenon::Real delta_transverse_magnetic[2]{};
  parthenon::Real wave_speed = 0.0;
};

// Rotation and wavelength geometry used by AthenaK's multidimensional linear
// wave.  The primitive eigenvector is constructed in the wave-aligned frame;
// this object maps it to the Cartesian mesh and supplies a single edge-vector
// potential whose discrete curl initializes every face field.
struct RelativisticMHDWaveGeometry {
  parthenon::Real cos_a2 = 1.0;
  parthenon::Real sin_a2 = 0.0;
  parthenon::Real cos_a3 = 1.0;
  parthenon::Real sin_a3 = 0.0;
  parthenon::Real wavelength = 1.0;
  parthenon::Real wave_number = 2.0 * 3.1415926535897932384626433832795;
};

namespace relativistic_mhd_wave {

inline parthenon::Real Square(const parthenon::Real value) { return value * value; }

inline RelativisticMHDWaveGeometry BuildGeometry(const parthenon::Real x1_length,
                                                 const parthenon::Real x2_length,
                                                 const parthenon::Real x3_length,
                                                 const bool along_x1,
                                                 const bool along_x2,
                                                 const bool along_x3) {
  RelativisticMHDWaveGeometry geometry{};
  if (!along_x1) {
    const parthenon::Real angle3 = std::atan(x1_length / x2_length);
    geometry.sin_a3 = std::sin(angle3);
    geometry.cos_a3 = std::cos(angle3);
    const parthenon::Real angle2 =
        std::atan(0.5 * (x1_length * geometry.cos_a3 +
                         x2_length * geometry.sin_a3) /
                  x3_length);
    geometry.sin_a2 = std::sin(angle2);
    geometry.cos_a2 = std::cos(angle2);
  }
  if (along_x2) {
    geometry.cos_a3 = 0.0;
    geometry.sin_a3 = 1.0;
    geometry.cos_a2 = 1.0;
    geometry.sin_a2 = 0.0;
  }
  if (along_x3) {
    geometry.cos_a3 = 0.0;
    geometry.sin_a3 = 1.0;
    geometry.cos_a2 = 0.0;
    geometry.sin_a2 = 1.0;
  }
  parthenon::Real wavelength = std::numeric_limits<float>::max();
  if (geometry.cos_a2 * geometry.cos_a3 > 0.0)
    wavelength = std::min(
        wavelength, x1_length * geometry.cos_a2 * geometry.cos_a3);
  if (geometry.cos_a2 * geometry.sin_a3 > 0.0)
    wavelength = std::min(
        wavelength, x2_length * geometry.cos_a2 * geometry.sin_a3);
  if (geometry.sin_a2 > 0.0)
    wavelength = std::min(wavelength, x3_length * geometry.sin_a2);
  geometry.wavelength = wavelength;
  geometry.wave_number = 2.0 * 3.1415926535897932384626433832795 / wavelength;
  return geometry;
}

KOKKOS_INLINE_FUNCTION parthenon::Real
WaveCoordinate(const parthenon::Real x1, const parthenon::Real x2,
               const parthenon::Real x3,
               const RelativisticMHDWaveGeometry& geometry) {
  return geometry.cos_a2 *
             (x1 * geometry.cos_a3 + x2 * geometry.sin_a3) +
         x3 * geometry.sin_a2;
}

template <int Component>
KOKKOS_INLINE_FUNCTION parthenon::Real
VectorPotential(const parthenon::Real x1, const parthenon::Real x2,
                const parthenon::Real x3, const RelativisticMHDWave& wave,
                const RelativisticMHDWaveGeometry& geometry) {
  const parthenon::Real x = WaveCoordinate(x1, x2, x3, geometry);
  const parthenon::Real y = -x1 * geometry.sin_a3 + x2 * geometry.cos_a3;
  const parthenon::Real ay =
      wave.magnetic[2] * x -
      wave.delta_transverse_magnetic[1] / geometry.wave_number *
          cos(geometry.wave_number * x);
  const parthenon::Real az =
      -wave.magnetic[1] * x +
      wave.delta_transverse_magnetic[0] / geometry.wave_number *
          cos(geometry.wave_number * x) +
      wave.magnetic[0] * y;
  if constexpr (Component == 0)
    return -ay * geometry.sin_a3 -
           az * geometry.sin_a2 * geometry.cos_a3;
  if constexpr (Component == 1)
    return ay * geometry.cos_a3 -
           az * geometry.sin_a2 * geometry.sin_a3;
  return az * geometry.cos_a2;
}

inline parthenon::Real QuadraticRoot(const parthenon::Real a1,
                                    const parthenon::Real a0,
                                    const bool greater_root) {
  if (a1 * a1 < 4.0 * a0) return -0.5 * a1;
  const parthenon::Real discriminant = std::sqrt(a1 * a1 - 4.0 * a0);
  if (greater_root)
    return a1 >= 0.0 ? -2.0 * a0 / (a1 + discriminant)
                     : 0.5 * (-a1 + discriminant);
  return a1 >= 0.0 ? 0.5 * (-a1 - discriminant)
                   : -2.0 * a0 / (a1 - discriminant);
}

inline parthenon::Real CubicRootReal(const parthenon::Real a2,
                                    const parthenon::Real a1,
                                    const parthenon::Real a0) {
  const parthenon::Real q = (a2 * a2 - 3.0 * a1) / 9.0;
  const parthenon::Real r =
      (2.0 * a2 * a2 * a2 - 9.0 * a1 * a2 + 27.0 * a0) / 54.0;
  if (r * r - q * q * q < 0.0) {
    const parthenon::Real theta = std::acos(r / std::sqrt(q * q * q));
    return -2.0 * std::sqrt(q) * std::cos(theta / 3.0) - a2 / 3.0;
  }
  const parthenon::Real a =
      -std::copysign(1.0, r) * std::cbrt(std::abs(r) + std::sqrt(r * r - q * q * q));
  const parthenon::Real b = a != 0.0 ? q / a : 0.0;
  return a + b - a2 / 3.0;
}

inline void QuarticRoots(const parthenon::Real a3, const parthenon::Real a2,
                         const parthenon::Real a1, const parthenon::Real a0,
                         parthenon::Real* first, parthenon::Real* second,
                         parthenon::Real* third, parthenon::Real* fourth) {
  const parthenon::Real b2 = a2 - 3.0 / 8.0 * Square(a3);
  const parthenon::Real b1 = a1 - 0.5 * a2 * a3 + 0.125 * a3 * Square(a3);
  const parthenon::Real b0 =
      a0 - 0.25 * a1 * a3 + 0.0625 * a2 * Square(a3) -
      3.0 / 256.0 * Square(Square(a3));
  const parthenon::Real z0 =
      CubicRootReal(-b2, -4.0 * b0, 4.0 * b0 * b2 - Square(b1));
  const parthenon::Real d1 = z0 - b2 > 0.0 ? std::sqrt(z0 - b2) : 0.0;
  const parthenon::Real e1 = -d1;
  const parthenon::Real radical = std::sqrt(Square(z0) / 4.0 - b0);
  const parthenon::Real d0 = b1 < 0.0 ? z0 / 2.0 + radical : z0 / 2.0 - radical;
  const parthenon::Real e0 = b1 < 0.0 ? z0 / 2.0 - radical : z0 / 2.0 + radical;
  const parthenon::Real y1 = QuadraticRoot(d1, d0, false);
  const parthenon::Real y2 = QuadraticRoot(d1, d0, true);
  const parthenon::Real y3 = QuadraticRoot(e1, e0, false);
  const parthenon::Real y4 = QuadraticRoot(e1, e0, true);
  *first = std::min(y1, y3) - a3 / 4.0;
  // Preserve AthenaK's operation order here.  Moving the common a3/4 shift
  // outside the min/max is algebraically equivalent, but changes the selected
  // magnetosonic eigenvalue by a few ulps and therefore changes the exact end
  // time of a one-period comparison.
  const parthenon::Real middle_first = std::max(y1, y3) - a3 / 4.0;
  const parthenon::Real middle_second = std::min(y2, y4) - a3 / 4.0;
  *second = std::min(middle_first, middle_second);
  *third = std::max(middle_first, middle_second);
  *fourth = std::max(y2, y4) - a3 / 4.0;
}

inline RelativisticMHDWave Build(const parthenon::Real density,
                                 const parthenon::Real pressure,
                                 const parthenon::Real velocity_x,
                                 const parthenon::Real velocity_y,
                                 const parthenon::Real velocity_z,
                                 const parthenon::Real magnetic_x,
                                 const parthenon::Real magnetic_y,
                                 const parthenon::Real magnetic_z,
                                 const parthenon::Real gamma,
                                 const parthenon::Real amplitude,
                                 const int wave_flag) {
  using parthenon::Real;
  RelativisticMHDWave wave{};
  wave.density = density;
  wave.pressure = pressure;
  wave.velocity[0] = velocity_x;
  wave.velocity[1] = velocity_y;
  wave.velocity[2] = velocity_z;
  wave.magnetic[0] = magnetic_x;
  wave.magnetic[1] = magnetic_y;
  wave.magnetic[2] = magnetic_z;
  const Real velocity_squared = Square(velocity_x) + Square(velocity_y) + Square(velocity_z);
  wave.four_velocity[0] = 1.0 / std::sqrt(1.0 - velocity_squared);
  for (int axis = 0; axis < 3; ++axis)
    wave.four_velocity[axis + 1] = wave.four_velocity[0] * wave.velocity[axis];
  wave.comoving_magnetic[0] = magnetic_x * wave.four_velocity[1] +
                              magnetic_y * wave.four_velocity[2] +
                              magnetic_z * wave.four_velocity[3];
  for (int axis = 0; axis < 3; ++axis) {
    wave.comoving_magnetic[axis + 1] =
        (wave.magnetic[axis] +
         wave.comoving_magnetic[0] * wave.four_velocity[axis + 1]) /
        wave.four_velocity[0];
  }
  const Real gas_enthalpy = eos::IdealGas{gamma}.EnthalpyDensity(density, pressure);
  const Real sound_squared = gamma * pressure / gas_enthalpy;
  Real magnetic_squared = -Square(wave.comoving_magnetic[0]);
  for (int mu = 1; mu < 4; ++mu) magnetic_squared += Square(wave.comoving_magnetic[mu]);
  const Real total_enthalpy = gas_enthalpy + magnetic_squared;

  if (wave_flag == 3) {
    wave.wave_speed = velocity_x;
    wave.delta_density = 1.0;
  } else if (wave_flag == 1 || wave_flag == 5) {
    const Real root_total_enthalpy = std::sqrt(total_enthalpy);
    const Real lambda_plus =
        (wave.comoving_magnetic[1] + root_total_enthalpy * wave.four_velocity[1]) /
        (wave.comoving_magnetic[0] + root_total_enthalpy * wave.four_velocity[0]);
    const Real lambda_minus =
        (wave.comoving_magnetic[1] - root_total_enthalpy * wave.four_velocity[1]) /
        (wave.comoving_magnetic[0] - root_total_enthalpy * wave.four_velocity[0]);
    Real sign = 1.0;
    if ((lambda_plus > lambda_minus && wave_flag == 1) ||
        (lambda_plus <= lambda_minus && wave_flag == 5))
      sign = -1.0;
    wave.wave_speed = sign > 0.0 ? lambda_plus : lambda_minus;
    Real alpha1[4]{wave.four_velocity[3], wave.wave_speed * wave.four_velocity[3], 0.0,
                   wave.four_velocity[0] - wave.wave_speed * wave.four_velocity[1]};
    Real alpha2[4]{-wave.four_velocity[2], -wave.wave_speed * wave.four_velocity[2],
                   wave.wave_speed * wave.four_velocity[1] - wave.four_velocity[0], 0.0};
    const Real g1 =
        (magnetic_y + wave.wave_speed * velocity_y /
                          (1.0 - wave.wave_speed * velocity_x) * magnetic_x) /
        wave.four_velocity[0];
    const Real g2 =
        (magnetic_z + wave.wave_speed * velocity_z /
                          (1.0 - wave.wave_speed * velocity_x) * magnetic_x) /
        wave.four_velocity[0];
    const Real norm = std::sqrt(Square(g1) + Square(g2));
    const Real f1 = norm == 0.0 ? 1.0 / std::sqrt(2.0) : g1 / norm;
    const Real f2 = norm == 0.0 ? 1.0 / std::sqrt(2.0) : g2 / norm;
    for (int mu = 0; mu < 4; ++mu) {
      wave.delta_four_velocity[mu] = f1 * alpha1[mu] + f2 * alpha2[mu];
      wave.delta_comoving_magnetic[mu] =
          -sign * root_total_enthalpy * wave.delta_four_velocity[mu];
    }
  } else {
    const Real factor_a = gas_enthalpy * (1.0 / sound_squared - 1.0);
    const Real factor_b = -(gas_enthalpy + magnetic_squared / sound_squared);
    const Real gamma2 = Square(wave.four_velocity[0]);
    const Real gamma4 = Square(gamma2);
    const Real coeff4 = factor_a * gamma4 - factor_b * gamma2 -
                        Square(wave.comoving_magnetic[0]);
    const Real coeff3 = -4.0 * factor_a * gamma4 * velocity_x +
                        2.0 * factor_b * gamma2 * velocity_x +
                        2.0 * wave.comoving_magnetic[0] * wave.comoving_magnetic[1];
    const Real coeff2 = 6.0 * factor_a * gamma4 * Square(velocity_x) +
                        factor_b * gamma2 * (1.0 - Square(velocity_x)) +
                        Square(wave.comoving_magnetic[0]) -
                        Square(wave.comoving_magnetic[1]);
    const Real coeff1 = -4.0 * factor_a * gamma4 * velocity_x * Square(velocity_x) -
                        2.0 * factor_b * gamma2 * velocity_x -
                        2.0 * wave.comoving_magnetic[0] * wave.comoving_magnetic[1];
    const Real coeff0 = factor_a * gamma4 * Square(Square(velocity_x)) +
                        factor_b * gamma2 * Square(velocity_x) +
                        Square(wave.comoving_magnetic[1]);
    Real lambda_fast_left = 0.0, lambda_slow_left = 0.0;
    Real lambda_slow_right = 0.0, lambda_fast_right = 0.0;
    QuarticRoots(coeff3 / coeff4, coeff2 / coeff4, coeff1 / coeff4, coeff0 / coeff4,
                 &lambda_fast_left, &lambda_slow_left, &lambda_slow_right,
                 &lambda_fast_right);
    Real lambda_other = 0.0;
    if (wave_flag == 0) {
      wave.wave_speed = lambda_fast_left;
      lambda_other = lambda_slow_left;
    } else if (wave_flag == 2) {
      wave.wave_speed = lambda_slow_left;
      lambda_other = lambda_fast_left;
    } else if (wave_flag == 4) {
      wave.wave_speed = lambda_slow_right;
      lambda_other = lambda_fast_right;
    } else {
      wave.wave_speed = lambda_fast_right;
      lambda_other = lambda_slow_right;
    }
    const Real root_total_enthalpy = std::sqrt(total_enthalpy);
    const Real lambda_plus =
        (wave.comoving_magnetic[1] + root_total_enthalpy * wave.four_velocity[1]) /
        (wave.comoving_magnetic[0] + root_total_enthalpy * wave.four_velocity[0]);
    const Real lambda_minus =
        (wave.comoving_magnetic[1] - root_total_enthalpy * wave.four_velocity[1]) /
        (wave.comoving_magnetic[0] - root_total_enthalpy * wave.four_velocity[0]);
    Real lambda_alfven = lambda_plus;
    Real sign = 1.0;
    if ((lambda_plus > lambda_minus && wave_flag < 3) ||
        (lambda_plus <= lambda_minus && wave_flag > 3)) {
      lambda_alfven = lambda_minus;
      sign = -1.0;
    }
    const Real a = wave.four_velocity[0] * (velocity_x - wave.wave_speed);
    const Real g = 1.0 - Square(wave.wave_speed);
    const Real b_over_a =
        -sign * std::sqrt(-factor_b - factor_a * Square(a) / g);
    Real alpha1[4]{wave.four_velocity[3], wave.wave_speed * wave.four_velocity[3], 0.0,
                   wave.four_velocity[0] - wave.wave_speed * wave.four_velocity[1]};
    Real alpha2[4]{-wave.four_velocity[2], -wave.wave_speed * wave.four_velocity[2],
                   wave.wave_speed * wave.four_velocity[1] - wave.four_velocity[0], 0.0};
    Real alpha11 = -Square(alpha1[0]);
    Real alpha12 = -alpha1[0] * alpha2[0];
    Real alpha22 = -Square(alpha2[0]);
    for (int mu = 1; mu < 4; ++mu) {
      alpha11 += Square(alpha1[mu]);
      alpha12 += alpha1[mu] * alpha2[mu];
      alpha22 += Square(alpha2[mu]);
    }
    const Real g1 =
        (magnetic_y + wave.wave_speed * velocity_y /
                          (1.0 - wave.wave_speed * velocity_x) * magnetic_x) /
        wave.four_velocity[0];
    const Real g2 =
        (magnetic_z + wave.wave_speed * velocity_z /
                          (1.0 - wave.wave_speed * velocity_x) * magnetic_x) /
        wave.four_velocity[0];
    const Real gram = alpha11 * alpha22 - Square(alpha12);
    const Real c1 = (g1 * alpha12 + g2 * alpha22) / gram * wave.four_velocity[0] *
                    (1.0 - wave.wave_speed * velocity_x);
    const Real c2 = -(g1 * alpha11 + g2 * alpha12) / gram * wave.four_velocity[0] *
                    (1.0 - wave.wave_speed * velocity_x);
    Real transverse[4]{};
    for (int mu = 0; mu < 4; ++mu)
      transverse[mu] = c1 * alpha1[mu] + c2 * alpha2[mu];
    const Real transverse_direction_norm = std::sqrt(Square(g1) + Square(g2));
    const Real f1 = transverse_direction_norm == 0.0
                        ? 1.0 / std::sqrt(2.0)
                        : g1 / transverse_direction_norm;
    const Real f2 = transverse_direction_norm == 0.0
                        ? 1.0 / std::sqrt(2.0)
                        : g2 / transverse_direction_norm;
    Real phi_plus_a_u[4]{};
    for (int mu = 0; mu < 4; ++mu) phi_plus_a_u[mu] = a * wave.four_velocity[mu];
    phi_plus_a_u[0] += wave.wave_speed;
    phi_plus_a_u[1] += 1.0;
    if (std::abs(wave.wave_speed - lambda_alfven) <=
        std::abs(lambda_other - lambda_alfven)) {
      const Real direction_denominator =
          std::sqrt(gram * (Square(f1) * alpha11 + 2.0 * f1 * f2 * alpha12 +
                            Square(f2) * alpha22));
      Real transverse_normalized[4]{};
      for (int mu = 0; mu < 4; ++mu)
        transverse_normalized[mu] =
            ((f1 * alpha12 + f2 * alpha22) * alpha1[mu] -
             (f1 * alpha11 + f2 * alpha12) * alpha2[mu]) /
            direction_denominator;
      Real transverse_norm = -Square(transverse[0]);
      for (int mu = 1; mu < 4; ++mu) transverse_norm += Square(transverse[mu]);
      transverse_norm = std::sqrt(transverse_norm);
      const Real denominator = Square(a) - (g + Square(a)) * sound_squared;
      wave.delta_pressure = denominator == 0.0
                                ? 0.0
                                : -(g + Square(a)) * sound_squared / denominator *
                                      transverse_norm;
      wave.delta_density = density / (gamma * pressure) * wave.delta_pressure;
      for (int mu = 0; mu < 4; ++mu) {
        wave.delta_four_velocity[mu] =
            -a * wave.delta_pressure /
                (gas_enthalpy * sound_squared * (g + Square(a))) *
                phi_plus_a_u[mu] -
            b_over_a / gas_enthalpy * transverse_normalized[mu];
        wave.delta_comoving_magnetic[mu] =
            -b_over_a * wave.delta_pressure / gas_enthalpy * wave.four_velocity[mu] -
            (1.0 + Square(a) / g) * transverse_normalized[mu];
      }
    } else {
      wave.delta_pressure = -1.0;
      wave.delta_density = density / (gamma * pressure) * wave.delta_pressure;
      const Real denominator = gas_enthalpy * Square(a) - magnetic_squared * g;
      Real transverse_reduced[4]{};
      if (denominator != 0.0)
        for (int mu = 0; mu < 4; ++mu)
          transverse_reduced[mu] = transverse[mu] / denominator;
      for (int mu = 0; mu < 4; ++mu) {
        wave.delta_four_velocity[mu] =
            a / (gas_enthalpy * sound_squared * (g + Square(a))) * phi_plus_a_u[mu] -
            b_over_a * g / gas_enthalpy * transverse_reduced[mu];
        wave.delta_comoving_magnetic[mu] =
            b_over_a / gas_enthalpy * wave.four_velocity[mu] -
            (1.0 + Square(a) / g) * g * transverse_reduced[mu];
      }
    }
  }

  Real perturbation_norm = Square(wave.delta_density) + Square(wave.delta_pressure);
  for (int mu = 0; mu < 4; ++mu)
    perturbation_norm += Square(wave.delta_four_velocity[mu]) +
                         Square(wave.delta_comoving_magnetic[mu]);
  perturbation_norm = std::sqrt(perturbation_norm);
  wave.delta_density /= perturbation_norm;
  wave.delta_pressure /= perturbation_norm;
  for (int mu = 0; mu < 4; ++mu) {
    wave.delta_four_velocity[mu] /= perturbation_norm;
    wave.delta_comoving_magnetic[mu] /= perturbation_norm;
  }

  const Real input_b0 = wave.comoving_magnetic[0];
  const Real input_u0 = wave.four_velocity[0];
  wave.magnetic[1] = wave.comoving_magnetic[2] * input_u0 -
                     input_b0 * wave.four_velocity[2];
  wave.magnetic[2] = wave.comoving_magnetic[3] * input_u0 -
                     input_b0 * wave.four_velocity[3];
  wave.delta_transverse_magnetic[0] =
      amplitude * ((wave.comoving_magnetic[2] * wave.delta_four_velocity[0] -
                    wave.comoving_magnetic[0] * wave.delta_four_velocity[2]) +
                   (wave.delta_comoving_magnetic[2] * wave.four_velocity[0] -
                    wave.delta_comoving_magnetic[0] * wave.four_velocity[2]));
  wave.delta_transverse_magnetic[1] =
      amplitude * ((wave.comoving_magnetic[3] * wave.delta_four_velocity[0] -
                    wave.comoving_magnetic[0] * wave.delta_four_velocity[3]) +
                   (wave.delta_comoving_magnetic[3] * wave.four_velocity[0] -
                    wave.delta_comoving_magnetic[0] * wave.four_velocity[3]));
  return wave;
}

} // namespace relativistic_mhd_wave
} // namespace pangu::pgen

#endif // PANGU_PGEN_RELATIVISTIC_MHD_WAVE_H_
