#include <cmath>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "ems_stress_energy.h"

int main(int argc,char** argv) {
  Kokkos::ScopeGuard scope(argc,argv);
  using namespace pangu::plugins::scalar_field::ems;
  Kokkos::View<Real*> errors("EMS source independent oracle errors",4);
  Kokkos::parallel_for("EMS action and ADM projections",4,KOKKOS_LAMBDA(int test) {
    constexpr Real scale=12.566370614359172953850573533118;
    Geometry g; g.metric[0]=g.metric[3]=g.metric[5]=1; g.Invert();
    Model model; Real error=0;
    if(test==0) {
      const Real d[3]={2,0,0},c[3]={0,3,0},grad[3]{};
      const auto em=PhysicalFields(g,1,d,c);
      const auto s=Sources(g,model,0,0,grad,em);
      error=fabs(scale*s.energy-6.5)+fabs(scale*s.momentum[2]-6)+
            fabs(scale*s.stress[0]-2.5)+fabs(scale*s.stress[3]+2.5)+
            fabs(scale*s.stress[5]-6.5);
    } else if(test==1) {
      const Real d[3]{},c[3]{},grad[3]={1,2,3};
      model.polynomial={4,0.2,0.1};
      const auto s=Sources(g,model,0.5,2,grad,PhysicalFields(g,1,d,c));
      error=fabs(scale*s.energy-9.5065104166666667)+
            fabs(scale*s.momentum[0]+2)+fabs(scale*s.momentum[1]+4)+
            fabs(scale*s.momentum[2]+6)+fabs(scale*s.stress[0]+4.5065104166666667)+
            fabs(scale*s.stress[1]-2)+fabs(scale*s.stress[5]-3.4934895833333333);
    } else if(test==2) {
      g.metric[0]=4; g.metric[3]=9; g.metric[5]=16; g.Invert();
      const Real d[3]={48,0,0},c[3]={0,0,24},grad[3]{};
      const auto em=PhysicalFields(g,1,d,c);
      const auto s=Sources(g,model,0,0,grad,em);
      error=fabs(g.sqrt_det-24)+fabs(em.e2-16)+fabs(em.b2-16)+
            fabs(scale*s.energy-16)+fabs(scale*s.momentum[1]+48)+
            fabs(scale*s.stress[3]-144);
    } else {
      model.alpha=2.4; model.polynomial={0.04,0.1,0.2};
      const Real phi=0.3,h=1.e-4,d[3]={0.5,-0.2,0.1},c[3]={0.1,0.7,-0.4};
      const auto energy=[&](Real value) {
        return model.Potential(value)+0.5*(0.30/model.F(value)+0.66*model.F(value));
      };
      const Real derivative=(-energy(phi+2*h)+8*energy(phi+h)-8*energy(phi-h)+energy(phi-2*h))/(12*h);
      error=fabs(ScalarForce(model,phi,PhysicalFields(g,model.F(phi),d,c))-derivative);
    }
    errors(test)=error;
  });
  const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),errors);
  for(int test=0;test<4;++test) {
    if(!std::isfinite(host(test)) || host(test)>1.e-9) {
      std::cerr<<"EMS source/action oracle failed: "<<test<<" error="<<host(test)<<'\n';
      return 1;
    }
  }
  std::cout<<"EMS ADM stress, curved geometry and action-gradient oracles PASS\n";
}
