#pragma once

#include <parthenon/parthenon.hpp>
#include "pangu/plugin_api/scalar_field.h"
#include "z4c/coupling/stress_energy.h"
#include "z4c/evolution/z4c_rhs.h"

namespace pangu::plugins::scalar_field::ems {
using parthenon::Real;

// S=(16pi)^-1 integral sqrt(-g)[R-2(dphi)^2-4V-f(phi)F^2].
// P=n^a d_a phi, D^i=sqrt(gamma) f E^i, C^i=sqrt(gamma) B^i.
struct Model {
  plugin_api::scalar_field::PolynomialPotential polynomial;
  plugin_api::scalar_field::AxionPotential axion;
  bool use_axion = false;
  int coupling = 0;
  Real alpha = 0;
  KOKKOS_INLINE_FUNCTION Real Potential(const Real phi) const {
    return use_axion ? axion.Value(phi) : polynomial.Value(phi*phi);
  }
  KOKKOS_INLINE_FUNCTION Real Gradient(const Real phi) const {
    return use_axion ? axion.Gradient(phi, phi*phi) : polynomial.Gradient(phi, phi*phi);
  }
  KOKKOS_INLINE_FUNCTION Real F(const Real phi) const {
    return plugin_api::scalar_field::MaxwellCoupling{alpha,false,coupling}.Value(phi);
  }
  KOKKOS_INLINE_FUNCTION Real FPrime(const Real phi) const {
    return plugin_api::scalar_field::MaxwellCoupling{alpha,false,coupling}.Derivative(phi);
  }
};

KOKKOS_INLINE_FUNCTION int Epsilon(const int i, const int j, const int k) {
  return i == j || j == k || i == k ? 0 : ((i-j)*(j-k)*(k-i) / 2);
}

struct Geometry {
  Real metric[6]{}, inverse[6]{}, sqrt_det = 1, lapse = 1, shift[3]{}, trace_k = 0;
  KOKKOS_INLINE_FUNCTION void Invert() {
    const Real det = nr::rhs::SpatialDeterminant(metric);
    sqrt_det = sqrt(det);
    nr::rhs::SpatialInverse(1/det, metric, inverse);
  }
  KOKKOS_INLINE_FUNCTION Real G(const int i, const int j) const {
    return metric[nr::SpatialSymmetricComponent(i,j)];
  }
  KOKKOS_INLINE_FUNCTION Real Inv(const int i, const int j) const {
    return inverse[nr::SpatialSymmetricComponent(i,j)];
  }
};

struct Electromagnetic {
  Real e[3]{}, b[3]{}, e_lower[3]{}, b_lower[3]{}, e2=0, b2=0;
};

KOKKOS_INLINE_FUNCTION Electromagnetic PhysicalFields(
    const Geometry& g, const Real f, const Real d[3], const Real c[3]) {
  Electromagnetic out;
  for (int i=0;i<3;++i) {
    out.e[i]=d[i]/(g.sqrt_det*f);
    out.b[i]=c[i]/g.sqrt_det;
  }
  for (int i=0;i<3;++i) {
    for (int j=0;j<3;++j) {
      out.e_lower[i]+=g.G(i,j)*out.e[j];
      out.b_lower[i]+=g.G(i,j)*out.b[j];
    }
    out.e2+=out.e[i]*out.e_lower[i];
    out.b2+=out.b[i]*out.b_lower[i];
  }
  return out;
}

KOKKOS_INLINE_FUNCTION nr::StressEnergy Sources(
    const Geometry& g, const Model& model, const Real phi, const Real p,
    const Real gradient[3], const Electromagnetic& em) {
  Real grad2=0;
  for (int i=0;i<3;++i)
    for (int j=0;j<3;++j) grad2+=g.Inv(i,j)*gradient[i]*gradient[j];
  const Real v=model.Potential(phi), f=model.F(phi);
  // Native Z4c uses G_ab=8pi T_ab; the action above gives G_ab=2 Tcanonical_ab.
  constexpr Real inv_four_pi = 0.079577471545947667884441881686257;
  nr::StressEnergy out;
  out.energy=inv_four_pi*(0.5*(p*p+grad2)+v+0.5*f*(em.e2+em.b2));
  for (int i=0;i<3;++i) {
    Real poynting=0;
    for (int j=0;j<3;++j)
      for (int k=0;k<3;++k) poynting+=g.sqrt_det*Epsilon(i,j,k)*em.e[j]*em.b[k];
    out.momentum[i]=inv_four_pi*(-p*gradient[i]+f*poynting);
    for (int j=i;j<3;++j)
      out.stress[nr::SpatialSymmetricComponent(i,j)]=inv_four_pi*(gradient[i]*gradient[j]+
          g.G(i,j)*(0.5*(p*p-grad2)-v)+f*(0.5*g.G(i,j)*(em.e2+em.b2)-
          em.e_lower[i]*em.e_lower[j]-em.b_lower[i]*em.b_lower[j]));
  }
  return out;
}

KOKKOS_INLINE_FUNCTION Real ScalarForce(const Model& model, const Real phi,
                                       const Electromagnetic& em) {
  return model.Gradient(phi)+0.5*model.FPrime(phi)*(em.b2-em.e2);
}
} // namespace pangu::plugins::scalar_field::ems
