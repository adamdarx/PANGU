#pragma once
#include <parthenon/parthenon.hpp>

namespace pangu::plugin_api::scalar_field {
using parthenon::Real;

// Device-copyable policies; the PDE and metric provider do not depend on the
// analytic form of V or f. New policies implement Gradient/Derivative, then
// instantiate Force with those types in the plugin (no device virtual calls).
struct PolynomialPotential {
  Real mass_squared = 0, quartic = 0, sextic = 0;
  KOKKOS_INLINE_FUNCTION Real Value(Real rho2) const {
    return 0.5 * mass_squared * rho2 + 0.5 * quartic * rho2 * rho2 +
           sextic * rho2 * rho2 * rho2 / 6;
  }
  // rho2=sum_a phi_a^2. V=mass^2*rho2/2+quartic*rho2^2/2+sextic*rho2^3/6.
  KOKKOS_INLINE_FUNCTION Real Gradient(Real phi_a, Real rho2) const {
    return (mass_squared + 2 * quartic * rho2 + sextic * rho2 * rho2) * phi_a;
  }
};
struct AxionPotential {
  Real mass_squared = 0, decay_constant = 1;
  KOKKOS_INLINE_FUNCTION Real Value(Real phi) const {
    // Avoid cancellation for small phi/f_a.
    const Real s = sin(phi / (2 * decay_constant));
    return 2 * mass_squared * decay_constant * decay_constant * s * s;
  }
  KOKKOS_INLINE_FUNCTION Real Gradient(Real phi, Real) const {
    return mass_squared * decay_constant * sin(phi / decay_constant);
  }
};
struct Coupling {
  // f=phi (linear), phi^2/2 (quadratic), exp(alpha*phi^2) (exponential).
  int kind = 0;
  Real alpha = 1;
  KOKKOS_INLINE_FUNCTION Real Derivative(Real phi) const {
    if (kind == 0)
      return 1;
    if (kind == 1)
      return phi;
    return 2 * alpha * phi * exp(alpha * phi * phi);
  }
};
struct Invariants {
  Real gauss_bonnet = 0;
  Real electrovac_maxwell_squared = 0;
};
struct MaxwellCoupling {
  Real alpha = 0;
  bool linearized = false;
  int kind = 0; // 0: exp(alpha phi^2), 1: 1+alpha phi^2
  KOKKOS_INLINE_FUNCTION Real Inverse(Real phi) const {
    return kind == 0 ? exp(-alpha * phi * phi)
                     : 1 / (1 + alpha * phi * phi);
  }
  KOKKOS_INLINE_FUNCTION Real Source(Real phi, Real electrovac_F2) const {
    if (alpha == 0)
      return 0;
    Real ratio = 1;
    if (!linearized) {
      const Real a = alpha * phi * phi;
      ratio = kind == 0 ? exp(-a) : 1 / ((1 + a) * (1 + a));
    }
    // f'/(4 f^2) times the electrovac invariant. Maxwell Gauss law
    // E=Q/(r^2 f) is enforced before coupling to the scalar equation.
    return 0.5 * alpha * phi * electrovac_F2 * ratio;
  }
};
template <class Potential, class CurvatureCoupling = Coupling,
          class ElectromagneticCoupling = MaxwellCoupling>
struct Force {
  Potential potential;
  CurvatureCoupling coupling;
  Real eta = 0;
  ElectromagneticCoupling electromagnetic;
  // Convention: Box(phi)=V'(phi)-eta*f'(phi)*G, derived from
  // L=-1/2 (d phi)^2 - V + eta f G. Metric backreaction is absent here.
  KOKKOS_INLINE_FUNCTION Real operator()(Real phi_a, Real rho2,
                                         Invariants invariants) const {
    const Real potential_force = potential.Gradient(phi_a, rho2);
    // Do not evaluate exp(alpha phi^2) if the coupling is disabled.
    const Real gb =
        eta == 0 ? 0
                 : eta * coupling.Derivative(phi_a) * invariants.gauss_bonnet;
    // EMS: f=exp(alpha phi^2), Box(phi)=V'+f' F^2/4.
    // Spherical Maxwell Gauss law fixes E=Q/(r^2 f); hence f'/f^2
    // gives exp(-alpha phi^2), evaluated without overflowing exp(+...).
    const Real em =
        electromagnetic.Source(phi_a, invariants.electrovac_maxwell_squared);
    return potential_force - gb + em;
  }
};
struct SchwarzschildPoint {
  Real radius, redshift, gauss_bonnet;
};
KOKKOS_INLINE_FUNCTION SchwarzschildPoint SchwarzschildTortoise(Real x,
                                                                Real mass) {
  // Solve x/(2M)=1+exp(y)+y with y=log(r/(2M)-1).
  // Keeping F in terms of y avoids cancellation near the horizon.
  const Real a = x / (2 * mass);
  Real y = a > 2 ? log(a) : a - 1;
  for (int n = 0; n < 30; ++n) {
    const Real e = exp(y);
    const Real step = (1 + e + y - a) / (1 + e);
    y -= step;
    if (fabs(step) < 1.e-13)
      break;
  }
  const Real e = exp(y), r = 2 * mass * (1 + e);
  const Real r2 = r * r;
  return {r, e / (1 + e), 48 * mass * mass / (r2 * r2 * r2)};
}
KOKKOS_INLINE_FUNCTION Real LinearHair(Real r, Real mass, Real eta) {
  return 2 * eta / (mass * r) * (1 + mass / r + 4 * mass * mass / (3 * r * r));
}
struct RNPoint {
  Real radius, redshift, gauss_bonnet, geometric_potential, maxwell_squared;
};
KOKKOS_INLINE_FUNCTION RNPoint RNTortoise(Real x, Real mass, Real charge) {
  Real r, f;
  if (charge == 0) {
    const auto p = SchwarzschildTortoise(x, mass);
    r = p.radius;
    f = p.redshift;
  } else {
    const Real delta = 2 * sqrt(mass * mass - charge * charge);
    const Real rp = mass + delta / 2, rm = mass - delta / 2;
    const Real a = rp * rp / delta, b = rm * rm / delta;
    Real y = x > 2 * rp ? log(x / rp) : (x - rp) / a;
    for (int n = 0; n < 40; ++n) {
      const Real d = rp * exp(y);
      const Real radius = rp + d;
      const Real residual = radius + a * y - b * log((delta + d) / rm) - x;
      const Real step = residual * (delta + d) / (radius * radius);
      y -= step;
      if (fabs(step) < 1.e-13)
        break;
    }
    const Real d = rp * exp(y);
    r = rp + d;
    f = d * (delta + d) / (r * r);
  }
  const Real r2 = r * r, q2 = charge * charge;
  const Real fp = 2 * mass / r2 - 2 * q2 / (r2 * r);
  return {r, f,
          (48 * mass * mass - 96 * mass * q2 / r + 40 * q2 * q2 / r2) /
              (r2 * r2 * r2),
          f * fp / r, -2 * q2 / (r2 * r2)};
}
} // namespace pangu::plugin_api::scalar_field
