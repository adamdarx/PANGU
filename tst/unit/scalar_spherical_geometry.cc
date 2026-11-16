#include "plugins/scalar_field/ems_spherical_geometry.h"
#include "z4c/evolution/constraints.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace sph=pangu::plugins::scalar_field::spherical;
namespace nr=pangu::nr;
using parthenon::Real;

Real tensor(int component,const Real x[3]) {
  int a=0,b=0;
  for(int i=0;i<3;++i)for(int j=i;j<3;++j)
    if(nr::SpatialSymmetricComponent(i,j)==component){a=i;b=j;}
  const Real r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
  return (a==b ? 1+.2*r2+.03*r2*r2:0)+(.4+.07*r2)*x[a]*x[b];
}
struct RN {
  Real center,h;
  KOKKOS_INLINE_FUNCTION Real operator()(int,int component,int,int,int i) const {
    const Real r=center+(i-100)*h,c=(1-.9*.9)/4;
    const Real psi2=1+1/r+c/(r*r);
    if(component==0)return 1/(psi2*psi2);
    if(component==1 || component==4 || component==6)return 1;
    if(component==18)return (1-c/(r*r))/psi2;
    return 0;
  }
};
struct Output {Real values[22]{};KOKKOS_INLINE_FUNCTION Real& operator[](int n){return values[n];}};

int main() {
  const Real r=.8,h=.003;Real d[3][6]{},dd[6][6]{};
  const Real t=1+.2*r*r+.03*pow(r,4),s=1+.6*r*r+.10*pow(r,4);
  sph::TensorJet(s,1.2*r+.4*r*r*r,1.2+1.2*r*r,
                 t,.4*r+.12*r*r*r,.4+.36*r*r,r,d,dd);
  const Real w[5]={1,-8,0,8,-1},w2[5]={-1,16,-30,16,-1};
  Real error1=0,error2=0;
  for(int n=0;n<6;++n)for(int a=0;a<3;++a) {
    Real first=0;
    for(int i=-2;i<=2;++i) {Real x[3]={r,0,0};x[a]+=i*h;first+=w[i+2]*tensor(n,x)/(12*h);}
    error1=std::max(error1,fabs(first-d[a][n]));
    for(int b=a;b<3;++b) {
      Real second=0;
      if(a==b)for(int i=-2;i<=2;++i) {Real x[3]={r,0,0};x[a]+=i*h;second+=w2[i+2]*tensor(n,x)/(12*h*h);}
      else for(int i=-2;i<=2;++i)for(int j=-2;j<=2;++j) {Real x[3]={r,0,0};x[a]+=i*h;x[b]+=j*h;second+=w[i+2]*w[j+2]*tensor(n,x)/(144*h*h);}
      error2=std::max(error2,fabs(second-dd[nr::SpatialSymmetricComponent(a,b)][n]));
    }
  }
  if(error1>1.e-9 || error2>1.e-7)throw std::runtime_error("Spherical tensor jet disagrees with Cartesian polynomial oracle");
  std::cout<<"Tensor jet Cartesian polynomial errors: "<<error1<<' '<<error2<<'\n';
  Real previous=0;
  for(Real spacing:{.04,.02,.01}) {
    RN pack{r,spacing};nr::rhs::PointState state;nr::rhs::PointDerivatives derivatives;
    sph::LoadPoint(pack,0,0,100,0,10000,spacing,r,1,0,state,derivatives);
    nr::Z4cOptions options;options.use_z4c=false;
    options.gauge.lapse=nr::gauge::Lapse::one_plus_log;
    options.gauge.shift=nr::gauge::Shift::gamma_driver;
    nr::StressEnergy matter;
    matter.energy=.9*.9*state.chi*state.chi/(8*M_PI*pow(r,4));
    matter.stress[0]=-matter.energy/state.chi;
    matter.stress[3]=matter.stress[5]=matter.energy/state.chi;
    Output out;nr::rhs::ComputeVacuum(state,derivatives,options,0,out);
    nr::rhs::ApplyMatterSources(state,options,matter,out);
    Real error=0;for(Real value:out.values)error=std::max(error,fabs(value));
    std::cout<<"Static RN h="<<spacing<<" maximum RHS="<<error;
    if(previous>0) {
      const Real order=log(previous/error)/log(2.);std::cout<<" order="<<order;
      if(order<3.7)throw std::runtime_error("Static RN Einstein equilibrium did not converge at fourth order");
    }
    std::cout<<'\n';previous=error;
  }
  return 0;
}
