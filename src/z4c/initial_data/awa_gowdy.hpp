#ifndef AWA_EXACT_GOWDY_HPP_
#define AWA_EXACT_GOWDY_HPP_
#include <cmath>

#ifdef KOKKOS_INLINE_FUNCTION
#define AWA_GOWDY_INLINE KOKKOS_INLINE_FUNCTION
#else
#define AWA_GOWDY_INLINE inline
#endif

namespace awa_exact {
// arXiv:1810.12346, section 8. Keep elapsed driver time separate from the
// paper's harmonic coordinate time and from positive areal time.
struct GowdyTime {
  double areal, lapse_factor;
  static GowdyTime expanding(double elapsed) { return {1.0+elapsed,1.0}; }
  static GowdyTime contracting(double elapsed) {
    constexpr double c=0.0021195119214617, k=9.6707698127638;
    constexpr double tau_initial=-9.8753205829098;
    const double t=k*std::exp(-c*(tau_initial+elapsed));
    return {t,-c*t}; // signed lapse is essential to the paper's time orientation
  }
};

// Bessel functions are computed on the host once per initialisation or output,
// not per GPU cell. The resulting coefficients can be copied into a kernel.
struct GowdyCoefficients {
  double t,j0,j1,constant,lapse_factor;
  explicit GowdyCoefficients(GowdyTime clock):t(clock.areal),lapse_factor(clock.lapse_factor) {
    constexpr double pi=3.141592653589793238462643383279502884;
    j0=std::cyl_bessel_j(0,2*pi*t); j1=std::cyl_bessel_j(1,2*pi*t);
    const double a=std::cyl_bessel_j(0,2*pi), b=std::cyl_bessel_j(1,2*pi);
    constant=2*pi*pi*(a*a+b*b)-pi*a*b;
  }
};

// Physical ADM state in canonical order (xx,xy,xz,yy,yz,zz), with waves along z.
// K_ij retains the areal-time normal orientation even for signed-lapse collapse.
AWA_GOWDY_INLINE void GowdyADM(const GowdyCoefficients &c,double z,double g[6],
                              double curvature[6],double &lapse,double &gamma_z) {
  constexpr double pi=3.141592653589793238462643383279502884;
  const double co=std::cos(2*pi*z),si=std::sin(2*pi*z),p=c.j0*co;
  const double lambda=2*pi*pi*c.t*c.t*(c.j0*c.j0+c.j1*c.j1)
      -2*pi*c.t*c.j0*c.j1*co*co-c.constant;
  const double pt=-2*pi*c.j1*co;
  const double lt=4*pi*pi*c.t*(c.j1*c.j1*co*co+c.j0*c.j0*si*si);
  const double alpha=std::pow(c.t,-.25)*std::exp(lambda/4);
  for(int a=0;a<6;++a) {g[a]=0;curvature[a]=0;}
  g[0]=c.t*std::exp(p);g[3]=c.t*std::exp(-p);
  g[5]=std::pow(c.t,-.5)*std::exp(lambda/2);
  curvature[0]=-.5*g[0]*(1/c.t+pt)/alpha;
  curvature[3]=-.5*g[3]*(1/c.t-pt)/alpha;
  curvature[5]=-.25*g[5]*(lt-1/c.t)/alpha;
  lapse=c.lapse_factor*alpha;
  gamma_z=(4*pi*pi/3)*c.t*c.t*std::exp(-lambda/3)*c.j0*c.j1*std::sin(4*pi*z);
}
} // namespace awa_exact
#undef AWA_GOWDY_INLINE
#endif
