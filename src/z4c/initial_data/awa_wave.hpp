#ifndef AWA_EXACT_GAUGE_WAVE_HPP_
#define AWA_EXACT_GAUGE_WAVE_HPP_
#include <cmath>

#ifdef KOKKOS_INLINE_FUNCTION
#define AWA_EXACT_INLINE KOKKOS_INLINE_FUNCTION
#else
#define AWA_EXACT_INLINE inline
#endif

namespace awa_exact {
constexpr double pi = 3.141592653589793238462643383279502884;
AWA_EXACT_INLINE int pair(int a, int b) {
  if (a > b) { const int t=a; a=b; b=t; }
  return a==0 ? b : (a==1 ? b+2 : 5);
}

// Exact spatial rotation of a plane gauge wave. For diagonal propagation the
// unit direction is (0,-1,+1)/sqrt(2), wavelength 1/sqrt(2). Rotate beta as a
// vector, not as the unnormalised phase gradient. This preserves g_tt=-1+H
// for the shifted Kerr-Schild wave and its flat-spacetime coordinate transform.
AWA_EXACT_INLINE void GaugeWave(bool shifted, bool diagonal, double amplitude,
                               double x, double y, double z, double time,
                               double wavelength, double state[22]) {
  const double root2=std::sqrt(2.0);
  const double n[3]={diagonal ? 0.0 : 1.0,
                     diagonal ? -1.0/root2 : 0.0,
                     diagonal ? 1.0/root2 : 0.0};
  const double frequency=(diagonal ? root2 : 1.0)/wavelength;
  const double phase=2.0*pi*((diagonal ? z-y : x)/wavelength-frequency*time);
  const double h=amplitude*std::sin(phase);
  const double hc=pi*frequency*amplitude*std::cos(phase);
  const double sign=shifted ? 1.0 : -1.0;
  const double f=1.0+sign*h;
  const double chi=std::pow(f,-1.0/3.0);
  const double curvature=-hc/std::sqrt(f);
  const double trace=curvature/f;
  for (int a=0;a<22;++a) state[a]=0.0;
  state[0]=chi;
  state[7]=trace;
  state[18]=shifted ? 1.0/std::sqrt(f) : std::sqrt(f);
  for (int a=0;a<3;++a) {
    state[14+a]=(4.0/3.0)*sign*hc*n[a]*std::pow(f,-5.0/3.0);
    state[19+a]=shifted ? -h/f*n[a] : 0.0;
    for (int b=a;b<3;++b) {
      const int ab=pair(a,b);
      const double metric=(a==b ? 1.0 : 0.0)+sign*h*n[a]*n[b];
      state[1+ab]=chi*metric;
      state[8+ab]=chi*(curvature*n[a]*n[b]-metric*trace/3.0);
    }
  }
}
} // namespace awa_exact
#undef AWA_EXACT_INLINE
#endif
