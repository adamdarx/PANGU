// Shared AwA host diagnostic algorithm; keep the two repository copies identical.
#ifndef AWA_DIAGNOSTIC_CORE_HPP_
#define AWA_DIAGNOSTIC_CORE_HPP_
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "awa_wave.hpp"

namespace awa_diagnostic {
constexpr double pi = 3.141592653589793238462643383279502884;
struct Config {
  std::string family, variant;
  double amplitude = 0, final_time = 0;
  bool enabled() const { return !family.empty(); }
  Config() {
    const auto family_env = std::getenv("AWA_FAMILY");
    if (!family_env) return;
    family = family_env;
    const auto variant_env = std::getenv("AWA_VARIANT");
    variant = variant_env ? variant_env : "aligned";
    const auto a = std::getenv("AWA_AMPLITUDE");
    const auto t = std::getenv("AWA_FINAL_TIME");
    amplitude = a ? std::stod(a) : 0;
    final_time = t ? std::stod(t) : 0;
  }
};
inline int sym(int a, int b) {
  if (a > b) std::swap(a, b);
  return a == 0 ? b : (a == 1 ? b + 2 : 5);
}
template<class State>
std::array<double, 6> inverse(State state, int k, int j, int i) {
  const double a = state(1,k,j,i), b = state(2,k,j,i), c = state(3,k,j,i);
  const double d = state(4,k,j,i), e = state(5,k,j,i), f = state(6,k,j,i);
  const double det = a*d*f+2*b*c*e-a*e*e-d*c*c-f*b*b;
  return {(d*f-e*e)/det,(c*e-b*f)/det,(b*e-c*d)/det,
          (a*f-c*c)/det,(b*c-a*e)/det,(a*d-b*b)/det};
}
inline double analytic(const Config &c, int component, double x, double y,
                       double z, double time) {
  const bool diagonal = component == 0 || component == 3 || component == 5;
  if (c.family == "robust") return diagonal ? 1.0 : 0.0;
  if (c.variant=="diagonal" && (c.family=="gauge" || c.family=="shifted-gauge")) {
    double exact[22];
    awa_exact::GaugeWave(c.family=="shifted-gauge",true,c.amplitude,x,y,z,time,1.0,exact);
    return exact[1+component]/exact[0];
  }
  if (c.family == "linear") {
    const bool rotated = c.variant == "diagonal";
    const double q = rotated ? z-y-std::sqrt(2.0)*time : z-time;
    const double weights1[6] = {-1,0,0,1,0,0};
    const double weights2[6] = {-1,0,0,0.5,0.5,0.5};
    return (diagonal ? 1.0 : 0.0) +
        (rotated ? weights2[component] : weights1[component]) *
        c.amplitude*std::sin(2*pi*q);
  }
  const double sign = c.family == "gauge" ? -1.0 : 1.0;
  return (diagonal ? 1.0 : 0.0) +
         (component == 0 ? sign*c.amplitude*std::sin(2*pi*(x-time)) : 0.0);
}
template<class State, class Constraint>
void write(const Config &c, double time, int cycle, const std::array<int,3> &n,
           const std::array<double,3> &dx, State state, Constraint constraint) {
  const auto start = std::chrono::steady_clock::now();
  const int axis = c.family == "linear" || c.family == "robust" || c.variant=="diagonal" ? 2 : 0;
  const int profile_component = c.variant=="diagonal" &&
      (c.family=="gauge" || c.family=="shifted-gauge") ? 5 : 0;
  const int count = n[axis];
  std::vector<double> profile(count), exact(count);
  std::array<std::vector<double>,3> extra_profile, extra_exact;
  for (int a=0;a<3;++a) { extra_profile[a].resize(count); extra_exact[a].resize(count); }
  double metric_error=0, theta=0, h=0, all_error=0, det_error=0, atrace=0;
  double gtt_error=0, gtaxis_error=0, k_error=0;
  std::array<double,3> momentum{}, ztilde{};
  std::size_t nonfinite=0;
  std::size_t invalid_geometry=0;
  for (int k=0;k<n[2];++k) for(int j=0;j<n[1];++j) for(int i=0;i<n[0];++i) {
    const double x=(i+0.5)*dx[0], y=(j+0.5)*dx[1], z=(k+0.5)*dx[2];
    const double chi=state(0,k,j,i);
    invalid_geometry += !(chi>0.0);
    for (int component=0;component<22;++component) {
      const double v=state(component,k,j,i);
      nonfinite += !std::isfinite(v);
      const bool unit=component==0||component==1||component==4||component==6||component==18;
      all_error=std::max(all_error,std::abs(v-(unit ? 1.0 : 0.0)));
    }
    for (int a=0;a<6;++a)
      metric_error=std::max(metric_error,std::abs(state(1+a,k,j,i)/chi-
                                               analytic(c,a,x,y,z,time)));
    theta=std::max(theta,std::abs(state(17,k,j,i)));
    const double hv=constraint(1,k,j,i);
    nonfinite += !std::isfinite(hv);
    h=std::max(h,std::abs(hv));
    for (int a=0;a<3;++a) {
      const double mv=constraint(4+a,k,j,i);
      nonfinite += !std::isfinite(mv);
      momentum[a]=std::max(momentum[a],std::abs(mv));
      double divergence=0;
      for (int b=0;b<3;++b) {
        double deriv=0;
        const int offsets[4]={-2,-1,1,2};
        const double weights[4]={1.0,-8.0,8.0,-1.0};
        for (int q=0;q<4;++q) {
          int ii=i,jj=j,kk=k;
          if (b==0) ii+=offsets[q];
          if (b==1) jj+=offsets[q];
          if (b==2) kk+=offsets[q];
          deriv += weights[q]*inverse(state,kk,jj,ii)[sym(a,b)];
        }
        divergence+=deriv/(12.0*dx[b]);
      }
      const double zv=0.5*(state(14+a,k,j,i)+divergence);
      nonfinite += !std::isfinite(zv);
      ztilde[a]=std::max(ztilde[a],std::abs(zv));
    }
    const auto inv=inverse(state,k,j,i);
    double trace=0;
    for (int a=0;a<3;++a) for(int b=0;b<3;++b)
      trace+=inv[sym(a,b)]*state(8+sym(a,b),k,j,i);
    atrace=std::max(atrace,std::abs(trace));
    const double a=state(1,k,j,i),b=state(2,k,j,i),cc=state(3,k,j,i);
    const double d=state(4,k,j,i),e=state(5,k,j,i),f=state(6,k,j,i);
    invalid_geometry += !(a*d*f+2*b*cc*e-a*e*e-d*cc*cc-f*b*b>0.0);
    det_error=std::max(det_error,std::abs(a*d*f+2*b*cc*e-a*e*e-d*cc*cc-f*b*b-1));
    double gtt=-state(18,k,j,i)*state(18,k,j,i), gtaxis=0;
    for(int aa=0;aa<3;++aa) {
      gtaxis+=state(1+sym(axis,aa),k,j,i)/chi*state(19+aa,k,j,i);
      for(int bb=0;bb<3;++bb)
        gtt+=state(1+sym(aa,bb),k,j,i)/chi*state(19+aa,k,j,i)*state(19+bb,k,j,i);
    }
    const bool gauge_wave=c.family=="gauge"||c.family=="shifted-gauge";
    const bool rotated_gauge=gauge_wave && c.variant=="diagonal";
    const double q=rotated_gauge ? z-y-std::sqrt(2.0)*time : x-time;
    const double wave=gauge_wave ? c.amplitude*std::sin(2*pi*q) : 0.0;
    const double exact_gtt=-1.0+wave;
    const double exact_gtaxis=c.family=="shifted-gauge" ? -wave/(rotated_gauge ? std::sqrt(2.0) : 1.0) : 0.0;
    const double exact_k=gauge_wave ? -(rotated_gauge ? std::sqrt(2.0) : 1.0)*pi*c.amplitude*std::cos(2*pi*q)/
        std::pow(1.0+(c.family=="gauge" ? -wave : wave),1.5) : 0.0;
    const double physical_k=state(7,k,j,i)+2*state(17,k,j,i);
    gtt_error=std::max(gtt_error,std::abs(gtt-exact_gtt));
    gtaxis_error=std::max(gtaxis_error,std::abs(gtaxis-exact_gtaxis));
    k_error=std::max(k_error,std::abs(physical_k-exact_k));
    // A single physical line, with a cell-centred analytic reference.
    if ((axis==2 && i==0 && j==0) || (axis==0 && j==0 && k==0)) {
      const int p=axis==2 ? k : i;
      profile[p]=state(1+profile_component,k,j,i)/chi-1;
      exact[p]=analytic(c,profile_component,x,y,z,time)-1;
      extra_profile[0][p]=gtt+1; extra_exact[0][p]=exact_gtt+1;
      extra_profile[1][p]=gtaxis; extra_exact[1][p]=exact_gtaxis;
      extra_profile[2][p]=physical_k; extra_exact[2][p]=exact_k;
    }
  }
  std::complex<double> f0{},f1{},a1{};
  for (int p=0;p<count;++p) {
    const auto phase=std::polar(1.0,-2*pi*(p+0.5)/count);
    f0+=profile[p]/count;
    f1+=profile[p]*phase/static_cast<double>(count);
    a1+=exact[p]*phase/static_cast<double>(count);
  }
  double amplitude_error=0,phase_error=0,offset=0;
  if (c.family!="robust") {
    amplitude_error=std::abs(f1)/std::abs(a1)-1.0;
    phase_error=std::arg(f1*std::conj(a1));
    offset=std::abs(f0)/c.amplitude;
  }
  const bool initial=cycle==0;
  const bool final=time+1e-8*std::max(1.0,c.final_time)>=c.final_time;
  if (initial && std::getenv("AWA_SAVE_INITIAL")) {
    std::ofstream raw("awa_initial.bin",std::ios::binary);
    for(int k=0;k<n[2];++k) for(int j=0;j<n[1];++j) for(int i=0;i<n[0];++i)
      for(int a=0;a<22;++a) {
        const double value=state(a,k,j,i);
        raw.write(reinterpret_cast<const char *>(&value),sizeof(value));
      }
  }
  if (final && c.family!="robust") {
    std::ofstream spectrum("awa_spectrum.csv");
    spectrum << "mode,numerical_real,numerical_imag,analytic_real,analytic_imag,"
                "gtt_real,gtt_imag,gtt_analytic_real,gtt_analytic_imag,"
                "gtaxis_real,gtaxis_imag,gtaxis_analytic_real,gtaxis_analytic_imag,"
                "K_real,K_imag,K_analytic_real,K_analytic_imag\n"
             << std::setprecision(17);
    for(int mode=0;mode<=count/2;++mode) {
      std::complex<double> numerical{},reference{};
      for(int p=0;p<count;++p) {
        const auto phase=std::polar(1.0,-2*pi*mode*(p+0.5)/count);
        numerical+=profile[p]*phase/static_cast<double>(count);
        reference+=exact[p]*phase/static_cast<double>(count);
      }
      spectrum<<mode<<','<<numerical.real()<<','<<numerical.imag()<<','
              <<reference.real()<<','<<reference.imag();
      for(int a=0;a<3;++a) {
        std::complex<double> numerical_extra{}, reference_extra{};
        for(int p=0;p<count;++p) {
          const auto phase=std::polar(1.0,-2*pi*mode*(p+0.5)/count);
          numerical_extra+=extra_profile[a][p]*phase/static_cast<double>(count);
          reference_extra+=extra_exact[a][p]*phase/static_cast<double>(count);
        }
        spectrum<<','<<numerical_extra.real()<<','<<numerical_extra.imag()<<','
                <<reference_extra.real()<<','<<reference_extra.imag();
      }
      spectrum<<'\n';
    }
  }
  const double seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-start).count();
  std::ofstream out("awa_metrics.csv",initial ? std::ios::trunc : std::ios::app);
  if(initial)
    out<<"time,cycle,metric_linf,theta_linf,H_linf,Mx_linf,My_linf,Mz_linf,"
          "Ztilde_x_linf,Ztilde_y_linf,Ztilde_z_linf,all_state_linf,det_error,"
          "A_trace_linf,nonfinite,F0_real,F0_imag,F1_real,F1_imag,"
          "analytic_F1_real,analytic_F1_imag,amplitude_rel_error,phase_error,"
          "offset_rel,diagnostic_host_s,gtt_linf,gtaxis_linf,K_linf,invalid_geometry\n";
  out<<std::setprecision(17)<<time<<','<<cycle<<','<<metric_error<<','<<theta<<','<<h;
  for(const auto v:momentum)out<<','<<v;
  for(const auto v:ztilde)out<<','<<v;
  out<<','<<all_error<<','<<det_error<<','<<atrace<<','<<nonfinite<<','
     <<f0.real()<<','<<f0.imag()<<','<<f1.real()<<','<<f1.imag()<<','
     <<a1.real()<<','<<a1.imag()<<','<<amplitude_error<<','<<phase_error<<','
     <<offset<<','<<seconds<<','<<gtt_error<<','<<gtaxis_error<<','<<k_error<<','
     <<invalid_geometry<<'\n';
  out.flush();
  if (nonfinite || invalid_geometry)
    throw std::runtime_error("AwA sampled a non-finite state or invalid spatial geometry");
}
} // namespace awa_diagnostic
#endif

