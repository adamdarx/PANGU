#include "ems_coupled.h"
#include "ems_spherical_geometry.h"
#include "ems_stress_energy.h"
#include "driver/contributor.h"
#include "interface/update.hpp"
#include "z4c/evolution/classical_rk4.h"
#include "z4c/evolution/constraints.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace pangu::plugins::scalar_field {
namespace {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using spherical::RadialJet;
using spherical::RadialValue;
constexpr const char* label="plugin.scalar_field";
struct Model {
  int fields=2;
  Real mass=2,charge=1,em_alpha=160,amplitude=1.e-6,center=4,width=1,phase=0;
  Real radius=30000,map_a=.07,cfl=.2,dissipation=.05,diagnostic_dt=1,diagnostic_outer=50;
  Real lapse_floor=1.e-10;
  bool minkowski=false;
  nr::Z4cOptions nr{};
};
struct Map {Real r,jac,jac2;};
KOKKOS_INLINE_FUNCTION Map Radius(const Model& p,Real x) {
  if(p.map_a==0) return {p.radius*x,p.radius,0};
  const Real scale=p.radius/sinh(1/p.map_a);
  return {scale*sinh(x/p.map_a),scale*cosh(x/p.map_a)/p.map_a,
          scale*sinh(x/p.map_a)/(p.map_a*p.map_a)};
}
template<class P> KOKKOS_INLINE_FUNCTION
ems::Geometry Geometry(const P& z,int k,int j,int i) {
  ems::Geometry g;
  for(int n=0;n<6;++n) g.metric[n]=z(0,1+n,k,j,i)/z(0,0,k,j,i);
  g.Invert();g.lapse=z(0,18,k,j,i);g.shift[0]=z(0,19,k,j,i);
  g.trace_k=z(0,7,k,j,i)+2*z(0,17,k,j,i);return g;
}
template<class U,class Z> KOKKOS_INLINE_FUNCTION
nr::StressEnergy Matter(const Model& p,const U& u,const Z& z,int k,int j,int i,
                       int lo,int hi,Real hx,const Map& map) {
  const auto g=Geometry(z,k,j,i);
  Real rho2=0,P2=0,X2=0,PX=0;
  for(int a=0;a<p.fields;++a) {
    const Real phi=u(0,2*a,k,j,i),P=u(0,2*a+1,k,j,i);
    Real X,XX;RadialJet(RadialValue<U>{u,2*a,k,j,lo,hi},i,hx,map.jac,map.jac2,X,XX);
    rho2+=phi*phi;P2+=P*P;X2+=X*X;PX+=P*X;
  }
  const Real f=exp(p.em_alpha*rho2),E=p.charge/(map.r*map.r*g.sqrt_det*f);
  const Real em_energy=.5*f*g.metric[0]*E*E,grad2=X2*g.inverse[0];
  constexpr Real c=1/(4*M_PI);
  nr::StressEnergy m;
  m.energy=c*(.5*(P2+grad2)+em_energy);m.momentum[0]=-c*PX;
  m.stress[0]=c*(X2+g.metric[0]*(.5*(P2-grad2)-em_energy));
  m.stress[3]=c*g.metric[3]*(.5*(P2-grad2)+em_energy);
  m.stress[5]=m.stress[3];return m;
}

void Refresh(MeshData<Real>* data) {
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Model>("spherical_model");
  const auto z=data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto a=data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto tm=data->PackVariables(std::vector<std::string>{"nr.tmunu"});
  const auto c=data->PackVariables(std::vector<std::string>{"nr.constraints"});
  const auto radial=data->PackVariables(std::vector<std::string>{"ems.radial"});
  const auto phi=data->PackVariables(std::vector<std::string>{"scalar.phi"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int lo=ib.s,hi=ib.e,j=data->GetBoundsJ(IndexDomain::interior).s,
            k=data->GetBoundsK(IndexDomain::interior).s;
  const Real hx=data->GetBlockData(0)->GetBlockPointer()->coords.Dxc<X1DIR>(k,j,lo);
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"spherical ADM and EMS sources",
      DevExecSpace(),lo,hi,KOKKOS_LAMBDA(int i) {
    const auto map=Radius(p,u.GetCoords(0).Xc<1>(i));const auto g=Geometry(z,k,j,i);
    for(int n=0;n<6;++n) {
      a(0,n,k,j,i)=g.metric[n];
      a(0,6+n,k,j,i)=(z(0,8+n,k,j,i)+z(0,1+n,k,j,i)*g.trace_k/3)/z(0,0,k,j,i);
    }
    a(0,12,k,j,i)=1/z(0,0,k,j,i);
    const auto matter=Matter(p,u,z,k,j,i,lo,hi,hx,map);
    for(int n=0;n<6;++n) tm(0,n,k,j,i)=matter.stress[n];
    tm(0,6,k,j,i)=matter.energy;
    for(int n=0;n<3;++n) tm(0,7+n,k,j,i)=matter.momentum[n];
    Real rho2=0;
    for(int n=0;n<p.fields;++n) {phi(0,n,k,j,i)=u(0,2*n,k,j,i);rho2+=u(0,2*n,k,j,i)*u(0,2*n,k,j,i);}
    radial(0,0,k,j,i)=map.r;radial(0,1,k,j,i)=map.r*sqrt(g.metric[3]);
    radial(0,2,k,j,i)=p.charge/(map.r*map.r*g.sqrt_det*exp(p.em_alpha*rho2));
    radial(0,3,k,j,i)=p.charge;
  });
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"spherical native ADM constraints",
      DevExecSpace(),lo,hi,KOKKOS_LAMBDA(int i) {
    const auto map=Radius(p,u.GetCoords(0).Xc<1>(i));
    nr::ADMConstraintInput in;
    Real first[13]{},second[13]{};
    for(int n=0;n<13;++n) RadialJet(RadialValue<decltype(a)>{a,n,k,j,lo,hi},i,hx,map.jac,map.jac2,first[n],second[n]);
    for(int n=0;n<6;++n) {in.metric[n]=a(0,n,k,j,i);in.extrinsic[n]=a(0,n+6,k,j,i);}
    spherical::TensorJet(in.metric[0],first[0],second[0],in.metric[3],first[3],second[3],map.r,in.dmetric,in.ddmetric);
    Real unused[6][6]{};
    spherical::TensorJet(in.extrinsic[0],first[6],second[6],in.extrinsic[3],first[9],second[9],map.r,in.dextrinsic,unused);
    in.psi4=a(0,12,k,j,i);in.dpsi4[0]=first[12];in.theta=z(0,17,k,j,i);
    Real inverse[6]{},contracted[3]{};
    nr::rhs::SpatialInverse(1/nr::rhs::SpatialDeterminant(in.metric),in.metric,inverse);
    nr::ContractedConformalChristoffel(inverse,in.dmetric,in.psi4,in.dpsi4,contracted);
    in.z[0]=.5*z(0,1,k,j,i)*(z(0,14,k,j,i)-contracted[0]);
    in.z_norm_squared=.25*z(0,1,k,j,i)*pow(z(0,14,k,j,i)-contracted[0],2);
    in.energy=tm(0,6,k,j,i);in.momentum[0]=tm(0,7,k,j,i);
    nr::ConstraintValues out;nr::ComputeADMConstraints(in,out);
    for(int n=0;n<8;++n)c(0,n,k,j,i)=out.values[n];
    const Real R=radial(0,1,k,j,i),Kt=in.extrinsic[3]/in.metric[3];
    const Real Rprime=sqrt(in.metric[3])*(1+.5*map.r*first[3]/in.metric[3]);
    radial(0,4,k,j,i)=2*(Rprime/(R*sqrt(in.metric[0]))-Kt);
    radial(0,5,k,j,i)=.5*R*(1-Rprime*Rprime/in.metric[0]+Kt*Kt*R*R)+p.charge*p.charge/(2*R);
    radial(0,6,k,j,i)=map.jac;
  });
}

TaskStatus RHS(MeshData<Real>* data,Real time) {
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Model>("spherical_model");
  const auto z=data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto a=data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto zr=data->PackVariables(std::vector<std::string>{"nr.rhs"});
  const auto ur=data->PackVariables(std::vector<std::string>{"scalar.rhs"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int lo=ib.s,hi=ib.e,j=data->GetBoundsJ(IndexDomain::interior).s,k=data->GetBoundsK(IndexDomain::interior).s;
  const Real hx=data->GetBlockData(0)->GetBlockPointer()->coords.Dxc<X1DIR>(k,j,lo);
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"spherical BSSN and scalar RK RHS",
      DevExecSpace(),lo,hi,KOKKOS_LAMBDA(int i) {
    const auto map=Radius(p,z.GetCoords(0).Xc<1>(i));
    nr::rhs::PointState state;nr::rhs::PointDerivatives derivatives;
    spherical::LoadPoint(z,k,j,i,lo,hi,hx,map.r,map.jac,map.jac2,state,derivatives);
    auto output=nr::rhs::PointOutput<decltype(zr)>{zr,0,k,j,i};
    nr::rhs::ComputeVacuum(state,derivatives,p.nr,time,output);
    nr::rhs::ApplyMatterSources(state,p.nr,Matter(p,u,z,k,j,i,lo,hi,hx,map),output);
    Real grprime,unused,gtprime;RadialJet(RadialValue<decltype(a)>{a,0,k,j,lo,hi},i,hx,map.jac,map.jac2,grprime,unused);
    RadialJet(RadialValue<decltype(a)>{a,3,k,j,lo,hi},i,hx,map.jac,map.jac2,gtprime,unused);
    const auto g=Geometry(z,k,j,i);Real rho2=0;
    for(int n=0;n<p.fields;++n)rho2+=pow(u(0,2*n,k,j,i),2);
    const Real f=exp(p.em_alpha*rho2),E=p.charge/(map.r*map.r*g.sqrt_det*f);
    for(int n=0;n<p.fields;++n) {
      const auto q=RadialValue<decltype(u)>{u,2*n,k,j,lo,hi};
      const auto P=RadialValue<decltype(u)>{u,2*n+1,k,j,lo,hi};
      Real X,XX;RadialJet(q,i,hx,map.jac,map.jac2,X,XX);
      const Real lap=(XX+(2/map.r+gtprime/g.metric[3]-.5*grprime/g.metric[0])*X)/g.metric[0];
      const Real source=p.em_alpha*q(i)*f*g.metric[0]*E*E;
      ur(0,2*n,k,j,i)=nr::fd::AdvectiveFirst<4>(0,1/hx,g.shift[0]/map.jac,spherical::ShiftedValue<decltype(q)>{q,i})+g.lapse*P(i);
      ur(0,2*n+1,k,j,i)=nr::fd::AdvectiveFirst<4>(0,1/hx,g.shift[0]/map.jac,spherical::ShiftedValue<decltype(P)>{P,i})+
          g.lapse*(g.trace_k*P(i)+lap+source)+derivatives.dalpha[0]*X/g.metric[0];
    }
    const Real speed=fmax(fabs(g.shift[0])+sqrt(2*fabs(g.lapse)/g.metric[0]),
                         sqrt(4*p.nr.gauge.shift_gamma/(3*state.metric[0])));
    for(int kind=0;kind<2;++kind) {
      const int count=kind==0 ? 22:2*p.fields;
      for(int n=0;n<count;++n) {
        const int parity=kind==0 && (n==14 || n==19) ? -1:1;
        // Both packs have the same VariablePack type; choose individual values.
        const auto qz=RadialValue<decltype(z)>{z,n,k,j,lo,hi,parity};
        const auto qu=RadialValue<decltype(u)>{u,n,k,j,lo,hi,parity};
        const auto at=[&](int m){return kind==0 ? qz(m):qu(m);};
        Real value=kind==0 ? zr(0,n,k,j,i):ur(0,n,k,j,i);
        const Real d6=at(i-3)-6*at(i-2)+15*at(i-1)-20*at(i)+15*at(i+1)-6*at(i+2)+at(i+3);
        value+=p.dissipation*speed*d6/(64*hx*map.jac);
        if(i>=hi-2) {
          Real d,dd;RadialJet(at,i,hx,map.jac,map.jac2,d,dd);
          const Real background=kind==0 && (n==0 || n==1 || n==4 || n==6 || n==18) ? 1:0;
          value=-(g.lapse/sqrt(g.metric[0])-g.shift[0])*(d+(at(i)-background)/map.r);
        }
        if(kind==0)zr(0,n,k,j,i)=value;else ur(0,n,k,j,i)=value;
      }
    }
  });return TaskStatus::complete;
}

Real Timestep(MeshData<Real>* data);
TaskStatus Update(MeshData<Real>* data,int stage,Real dt) {
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Model>("spherical_model");
  for(const auto& prefix:std::vector<std::string>{"nr","scalar"}) {
    const auto u=data->PackVariables(std::vector<std::string>{prefix=="nr" ? "nr.z4c":"scalar.state"});
    const auto rhs=data->PackVariables(std::vector<std::string>{prefix+".rhs"});
    const auto initial=data->PackVariables(std::vector<std::string>{prefix+".initial"});
    const auto sum=data->PackVariables(std::vector<std::string>{prefix+".rk_sum"});
    const auto ib=data->GetBoundsI(IndexDomain::interior);
    const int k=data->GetBoundsK(IndexDomain::interior).s,j=data->GetBoundsJ(IndexDomain::interior).s,
              count=prefix=="nr" ? 22:2*p.fields;
    parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"spherical classical RK4 update",
        DevExecSpace(),ib.s,ib.e,KOKKOS_LAMBDA(int i) {
      for(int n=0;n<count;++n) {
        if(stage==1)initial(0,n,k,j,i)=u(0,n,k,j,i);
        sum(0,n,k,j,i)=nr::classical_rk4::Accumulate(stage,rhs(0,n,k,j,i),sum(0,n,k,j,i));
        u(0,n,k,j,i)=nr::classical_rk4::Update(stage,initial(0,n,k,j,i),rhs(0,n,k,j,i),sum(0,n,k,j,i),dt);
      }
    });
  }
  const auto z=data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int k=data->GetBoundsK(IndexDomain::interior).s,j=data->GetBoundsJ(IndexDomain::interior).s;
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"spherical algebraic BSSN projection",
      DevExecSpace(),ib.s,ib.e,KOKKOS_LAMBDA(int i) {
    // Apply the existing puncture regularization at every RK substage. Do not
    // replace nonfinite values: the state check must still diagnose them.
    if(p.nr.floor_chi && z(0,0,k,j,i)<p.nr.chi_min_floor)
      z(0,0,k,j,i)=p.nr.chi_min_floor;
    // 1+log preserves the lapse sign in the continuum. High-order advection
    // and dissipation at the unresolved puncture need its positive floor too.
    if(z(0,18,k,j,i)<p.lapse_floor)z(0,18,k,j,i)=p.lapse_floor;
    Real s=z(0,1,k,j,i),t=.5*(z(0,4,k,j,i)+z(0,6,k,j,i));
    const Real normalization=cbrt(1/(s*t*t));s*=normalization;t*=normalization;
    z(0,1,k,j,i)=s;z(0,4,k,j,i)=z(0,6,k,j,i)=t;
    Real Ar=z(0,8,k,j,i),At=.5*(z(0,11,k,j,i)+z(0,13,k,j,i));
    const Real trace=Ar/s+2*At/t;
    z(0,8,k,j,i)=Ar-s*trace/3;z(0,11,k,j,i)=z(0,13,k,j,i)=At-t*trace/3;
    for(int n: {2,3,5,9,10,12,15,16,20,21})z(0,n,k,j,i)=0;
    if(!p.nr.use_z4c)z(0,17,k,j,i)=0;
  });
  // Inspect each RK substage before a bad state enters the next RHS.
  Timestep(data);
  return TaskStatus::complete;
}

Real Timestep(MeshData<Real>* data) {
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Model>("spherical_model");
  const auto z=data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int j=data->GetBoundsJ(IndexDomain::interior).s,k=data->GetBoundsK(IndexDomain::interior).s;
  const Real hx=data->GetBlockData(0)->GetBlockPointer()->coords.Dxc<X1DIR>(k,j,ib.s);
  Real minimum=std::numeric_limits<Real>::max();int invalid=0;
  parthenon::par_reduce(parthenon::loop_pattern_flatrange_tag,"spherical causal CFL",DevExecSpace(),
      ib.s,ib.e,KOKKOS_LAMBDA(int i,Real& value) {
    const Real alpha=z(0,18,k,j,i),chi=z(0,0,k,j,i),s=z(0,1,k,j,i);
    const auto map=Radius(p,z.GetCoords(0).Xc<1>(i));
    const Real speed=fabs(z(0,19,k,j,i))+fmax(fabs(alpha)*sqrt(chi/s),
        fmax(sqrt(2*fabs(alpha)*chi/s),sqrt(4*p.nr.gauge.shift_gamma/(3*s))));
    value=fmin(value,hx*map.jac/fmax(speed,1.e-12));
  },Kokkos::Min<Real>(minimum));
  parthenon::par_reduce(parthenon::loop_pattern_flatrange_tag,"spherical finite state",DevExecSpace(),
      ib.s,ib.e,KOKKOS_LAMBDA(int i,int& bad) {
    if(!(z(0,0,k,j,i)>0 && z(0,1,k,j,i)>0 && z(0,4,k,j,i)>0 && z(0,18,k,j,i)>0))++bad;
    for(int n=0;n<22;++n)if(!(fabs(z(0,n,k,j,i))<1.e300))++bad;
    for(int n=0;n<2*p.fields;++n)if(!(fabs(u(0,n,k,j,i))<1.e300))++bad;
  },Kokkos::Sum<int>(invalid));
  if(invalid!=0) {
    Kokkos::View<Real**> failed("spherical failed state",ib.e-ib.s+1,23);
    parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"retain spherical failed state",
        DevExecSpace(),ib.s,ib.e,KOKKOS_LAMBDA(int i) {
      failed(i-ib.s,0)=Radius(p,z.GetCoords(0).Xc<1>(i)).r;
      for(int n=0;n<22;++n)failed(i-ib.s,n+1)=z(0,n,k,j,i);
    });
    const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),failed);
    std::ofstream out("spherical_invalid_state.csv");out<<std::setprecision(17)<<"r";
    for(int n=0;n<22;++n)out<<",z"<<n;out<<'\n';
    for(int i=0;i<=ib.e-ib.s;++i) {out<<host(i,0);for(int n=1;n<23;++n)out<<','<<host(i,n);out<<'\n';}
  }
  PARTHENON_REQUIRE(invalid==0,"Invalid spherical BSSN/EMS state; retain output for inspection");
  return p.cfl*minimum;
}

void Initialize(Mesh*,ParameterInput*,MeshData<Real>* data) {
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Model>("spherical_model");
  const auto z=data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int j=data->GetBoundsJ(IndexDomain::interior).s,k=data->GetBoundsK(IndexDomain::interior).s;
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"isotropic RN spherical initial state",
      DevExecSpace(),ib.s,ib.e,KOKKOS_LAMBDA(int i) {
    const Real r=Radius(p,z.GetCoords(0).Xc<1>(i)).r;
    const Real psi2=1+p.mass/r+(p.mass*p.mass-p.charge*p.charge)/(4*r*r);
    for(int n=0;n<22;++n)z(0,n,k,j,i)=0;
    z(0,0,k,j,i)=1/(psi2*psi2);z(0,18,k,j,i)=1/psi2;
    z(0,1,k,j,i)=z(0,4,k,j,i)=z(0,6,k,j,i)=1;
    for(int n=0;n<2*p.fields;++n)u(0,n,k,j,i)=0;
    const Real profile=p.amplitude*(exp(-pow((r-p.center)/p.width,2))+exp(-pow((r+p.center)/p.width,2)));
    u(0,0,k,j,i)=profile*cos(p.phase);
    if(p.fields==2)u(0,2,k,j,i)=profile*sin(p.phase);
  });Refresh(data);
}

void Diagnostics(Mesh* mesh,Real time,bool force=false) {
  auto package=mesh->packages.Get(label);const auto p=package->Param<Model>("spherical_model");
  if(!force && time-package->Param<Real>("last_diagnostic")<p.diagnostic_dt*(1-1.e-10))return;
  auto& data=mesh->mesh_data.Get();const auto r=data->PackVariables(std::vector<std::string>{"ems.radial"});
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto c=data->PackVariables(std::vector<std::string>{"nr.constraints"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int j=data->GetBoundsJ(IndexDomain::interior).s,k=data->GetBoundsK(IndexDomain::interior).s;
  Kokkos::View<Real**> buffer("spherical diagnostics",ib.e-ib.s+1,13);
  const auto z=data->PackVariables(std::vector<std::string>{"nr.z4c"});
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"spherical diagnostics copy",DevExecSpace(),
      ib.s,ib.e,KOKKOS_LAMBDA(int i) {
    const int n=i-ib.s;for(int m=0;m<7;++m)buffer(n,m)=r(0,m,k,j,i);
    buffer(n,7)=u(0,0,k,j,i);buffer(n,8)=p.fields==2 ? u(0,2,k,j,i):0;
    buffer(n,9)=c(0,1,k,j,i);buffer(n,10)=c(0,2,k,j,i);
    buffer(n,11)=z(0,0,k,j,i);
    buffer(n,12)=z(0,18,k,j,i);
  });
  const auto s=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),buffer);
  const int count=s.extent(0);int bracket=-1;
  for(int i=1;i<count;++i)if(s(i-1,4)<=0 && s(i,4)>0)bracket=i;
  const auto interp=[&](int column,Real x,int near) {
    const int first=std::clamp(near-3,0,count-6);Real value=0;
    for(int i=first;i<first+6;++i) {
      Real w=1;for(int j=first;j<first+6;++j)if(i!=j)w*=(x-s(j,0))/(s(i,0)-s(j,0));
      value+=w*s(i,column);
    }return value;
  };
  Real rh=0,Rh=0,ph1=0,ph2=0;
  if(bracket>=0) {
    Real left=s(bracket-1,0),right=s(bracket,0);
    for(int n=0;n<60;++n) {const Real mid=.5*(left+right);if(interp(4,mid,bracket)>0)right=mid;else left=mid;}
    rh=.5*(left+right);Rh=interp(1,rh,bracket);ph1=interp(7,rh,bracket);ph2=interp(8,rh,bracket);
  }
  Real H2=0,M2=0,volume=0,Hmax=0,minimum_chi=std::numeric_limits<Real>::max();
  Real minimum_lapse=std::numeric_limits<Real>::max();
  int mass_index=0,floor_cells=0,floor_exterior_cells=0;
  int lapse_floor_cells=0,lapse_floor_exterior_cells=0;
  for(int i=0;i<count;++i) {
    minimum_chi=std::min(minimum_chi,s(i,11));
    if(p.nr.floor_chi && s(i,11)<=p.nr.chi_min_floor*(1+1.e-10)) {
      ++floor_cells;if(s(i,0)>rh)++floor_exterior_cells;
    }
    minimum_lapse=std::min(minimum_lapse,s(i,12));
    if(s(i,12)<=p.lapse_floor*(1+1.e-10)) {
      ++lapse_floor_cells;if(s(i,0)>rh)++lapse_floor_exterior_cells;
    }
    if(fabs(s(i,0)-p.diagnostic_outer)<fabs(s(mass_index,0)-p.diagnostic_outer))mass_index=i;
    if(s(i,0)<=1.1*rh || s(i,0)>p.diagnostic_outer)continue;
    const Real weight=s(i,0)*s(i,0)*s(i,6);volume+=weight;
    H2+=weight*s(i,9)*s(i,9);M2+=weight*s(i,10)*s(i,10);Hmax=std::max(Hmax,fabs(s(i,9)));
  }
  std::ofstream out(package->Param<std::string>("history_file"),std::ios::app);
  out<<std::setprecision(17)<<time<<','<<rh<<','<<Rh<<','<<Rh/2<<','<<ph1<<','<<ph2<<','
     <<hypot(ph1,ph2)<<','<<s(mass_index,5)<<','<<sqrt(H2/std::max(volume,1.e-300))<<','
     <<sqrt(M2/std::max(volume,1.e-300))<<','<<Hmax<<','<<minimum_chi<<','
     <<floor_cells<<','<<floor_exterior_cells<<','<<minimum_lapse<<','
     <<lapse_floor_cells<<','<<lapse_floor_exterior_cells<<'\n';
  package->UpdateParam("last_diagnostic",time);
}

void BeforeLoop(Mesh* mesh,ParameterInput*,parthenon::SimTime& tm) {
  const auto path=mesh->packages.Get(label)->Param<std::string>("history_file");
  std::vector<std::string> keep;std::ifstream old(path);std::string row;
  std::getline(old,row);
  while(std::getline(old,row)) {try{if(std::stod(row)<tm.time-1.e-12)keep.push_back(row);}catch(...) {}}
  std::ofstream out(path);out<<"time,horizon_coordinate_radius,horizon_areal_radius,irreducible_mass,phi1_h,phi2_h,abs_phi_h,charged_mass_at_diagnostic_radius,H_exterior_RMS,M_exterior_RMS,H_exterior_max,minimum_chi,chi_floor_cells,chi_floor_exterior_cells,minimum_lapse,lapse_floor_cells,lapse_floor_exterior_cells\n";
  for(const auto& line:keep)out<<line<<'\n';out.close();Diagnostics(mesh,tm.time,true);
}
void PostStep(Mesh* mesh,const parthenon::SimTime& tm) {Diagnostics(mesh,tm.time+tm.dt,tm.time+tm.dt>=tm.tlim-1.e-12);}
void Configure(Mesh* mesh,parthenon::LowStorageIntegrator& rk) {
  PARTHENON_REQUIRE(mesh->ndim==1 && mesh->nbtotal==1 && parthenon::Globals::nranks==1 && !mesh->multilevel,
                    "Spherical EMS uses one radial MeshBlock and one rank");
  PARTHENON_REQUIRE(rk.GetName()=="rk4","Spherical EMS uses classical RK4");
  rk.nstages=4;rk.nbuffers=2;rk.gam0={0,0,0,0};rk.gam1={1,1,1,1};
  rk.beta={.5,.5,1,1./6};rk.delta={0,0,0,0};rk.c={0,.5,.5,1};
  rk.buffer_name={"base","base","base"};rk.stage_name={"base","base","base","base","base"};
}
TaskCollection Build(driver::StageBuildContext& context) {
  TaskCollection collection;auto& tasks=collection.AddRegion(1)[0];
  auto& data=context.mesh->mesh_data.Add("base",context.mesh->GetDefaultBlockPartitions().front());
  const Real time=context.time->time+context.integrator->c[context.stage-1]*context.integrator->dt;
  const auto rhs=tasks.AddTask(TaskID(0),RHS,data.get(),time);
  const auto update=tasks.AddTask(rhs,Update,data.get(),context.stage,context.integrator->dt);
  const auto refresh=tasks.AddTask(update,parthenon::Update::FillDerived<MeshData<Real>>,data.get());
  if(context.stage==4)tasks.AddTask(refresh,parthenon::Update::EstimateTimestep<MeshData<Real>>,data.get());
  return collection;
}
} // namespace

std::shared_ptr<parthenon::StateDescriptor> InitializeSphericalEMS(plugin::InitContext& context) {
  auto& pin=context.input;const std::string block="plugin/scalar_field";Model p;
  const std::set<std::string> allowed={"backend","coordinates","fields","bh_mass","bh_charge","em_alpha","em_coupling","initial_data","amplitude","center","width","phase","outer_radius","radial_map_a","cfl","dissipation","diagnostic_interval","diagnostic_outer_radius","history_file","lapse_min_floor"};
  for(const auto& key:pin.GetParameterNames(block))PARTHENON_REQUIRE(allowed.count(key),"Unknown spherical EMS parameter: "+key);
  p.fields=pin.GetOrAddInteger(block,"fields",2);p.mass=pin.GetOrAddReal(block,"bh_mass",2);p.charge=pin.GetOrAddReal(block,"bh_charge",1);
  p.em_alpha=pin.GetOrAddReal(block,"em_alpha",160);p.amplitude=pin.GetOrAddReal(block,"amplitude",1.e-6);
  p.center=pin.GetOrAddReal(block,"center",4);p.width=pin.GetOrAddReal(block,"width",1);p.phase=pin.GetOrAddReal(block,"phase",0);
  p.radius=pin.GetOrAddReal(block,"outer_radius",30000);p.map_a=pin.GetOrAddReal(block,"radial_map_a",.07);
  p.cfl=pin.GetOrAddReal(block,"cfl",.2);p.dissipation=pin.GetOrAddReal(block,"dissipation",.05);
  p.diagnostic_dt=pin.GetOrAddReal(block,"diagnostic_interval",1);p.diagnostic_outer=pin.GetOrAddReal(block,"diagnostic_outer_radius",50);
  const auto initial=pin.GetOrAddString(block,"initial_data","gaussian");
  PARTHENON_REQUIRE(initial=="gaussian" || initial=="reissner_nordstrom" || initial=="minkowski","Unknown spherical initial data");
  if(initial!="gaussian")p.amplitude=0;
  if(initial=="minkowski")p.mass=p.charge=0;
  PARTHENON_REQUIRE(pin.GetOrAddString(block,"em_coupling","exponential")=="exponential","Spherical reproduction currently uses exp(alpha*|Phi|^2)");
  for(Real v:{p.mass,p.charge,p.em_alpha,p.amplitude,p.center,p.width,p.phase,p.radius,p.map_a,p.cfl,p.dissipation,p.diagnostic_dt,p.diagnostic_outer})PARTHENON_REQUIRE(std::isfinite(v),"Nonfinite spherical parameter");
  PARTHENON_REQUIRE((p.fields==1 || p.fields==2) && p.mass>=fabs(p.charge) && (p.mass>fabs(p.charge) || p.mass==0) && p.width>0 && p.radius>0 && p.map_a>=0 && p.cfl>0 && p.cfl<=.5 && p.em_alpha>=0 && p.dissipation>=0 && p.dissipation<=1 && p.diagnostic_dt>0 && p.diagnostic_outer>0,"Invalid spherical reproduction parameters");
  PARTHENON_REQUIRE(pin.GetReal("parthenon/mesh","x1min")==0 && pin.GetReal("parthenon/mesh","x1max")==1 && pin.GetInteger("parthenon/mesh","nx1")>=32,"Spherical computational radius spans[0,1], nx1>=32");
  const std::string nrblock="numerical_relativity";
  PARTHENON_REQUIRE(pin.GetOrAddReal(nrblock,"chi_psi_power",-4)==-4,"Spherical NR requires chi=psi^-4");
  p.nr.use_z4c=pin.GetOrAddBoolean(nrblock,"use_z4c",false);
  p.nr.floor_chi=pin.GetOrAddBoolean(nrblock,"floor_chi",true);
  p.nr.chi_min_floor=pin.GetOrAddReal(nrblock,"chi_min_floor",1.e-12);
  p.lapse_floor=pin.GetOrAddReal(block,"lapse_min_floor",1.e-10);
  PARTHENON_REQUIRE(std::isfinite(p.nr.chi_min_floor) && p.nr.chi_min_floor>0,
                    "Spherical chi_min_floor must be finite and positive");
  PARTHENON_REQUIRE(std::isfinite(p.lapse_floor) && p.lapse_floor>0,
                    "Spherical lapse_min_floor must be finite and positive");
  p.nr.gauge.lapse=nr::gauge::Lapse::one_plus_log;p.nr.gauge.shift=nr::gauge::Shift::gamma_driver;
  p.nr.gauge.shift_gamma=pin.GetOrAddReal(nrblock,"shift_gamma",.75);
  p.nr.gauge.shift_eta=pin.GetOrAddReal(nrblock,"shift_eta",p.mass>0 ? 2/p.mass:2);
  p.nr.damp_kappa1=pin.GetOrAddReal(nrblock,"damp_kappa1",0);
  PARTHENON_REQUIRE(p.nr.gauge.shift_gamma>=0 && p.nr.gauge.shift_eta>=0 && p.nr.damp_kappa1>=0,"Invalid spherical gauge parameters");
  auto package=std::make_shared<StateDescriptor>(label);package->AddParam<Model>("spherical_model",p);
  package->AddParam<std::string>("coordinates","spherical");package->AddParam<Real>("last_diagnostic",-std::numeric_limits<Real>::max(),true);
  package->AddParam<std::string>("history_file",pin.GetOrAddString(block,"history_file","ems_spherical_history.csv"));
  const Metadata independent({Metadata::Cell,Metadata::Independent,Metadata::Restart},std::vector<int>{22});
  package->AddField("nr.z4c",independent);
  package->AddField("scalar.state",Metadata({Metadata::Cell,Metadata::Independent,Metadata::Restart},std::vector<int>{2*p.fields}));
  const auto derived=[](int n){return Metadata({Metadata::Cell,Metadata::Derived,Metadata::OneCopy},std::vector<int>{n});};
  for(const auto& prefix:std::vector<std::string>{"nr","scalar"})for(const auto& suffix:std::vector<std::string>{"rhs","initial","rk_sum"})package->AddField(prefix+"."+suffix,derived(prefix=="nr" ? 22:2*p.fields));
  package->AddField("nr.adm",derived(13));package->AddField("nr.tmunu",derived(10));package->AddField("nr.constraints",derived(8));
  package->AddField("ems.radial",derived(7));package->AddField("scalar.phi",derived(p.fields));
  package->AddParam<driver::StageContribution>(std::string(driver::stage_contribution_key),{"scalar_ems_spherical",200,[](const driver::StageBuildContext&){return true;},Build,Configure,PostStep});
  package->PostProblemGeneratorMesh=Initialize;package->FillDerivedMesh=Refresh;package->EstimateTimestepMesh=Timestep;package->UserWorkBeforeLoopMesh=BeforeLoop;
  return package;
}
} // namespace pangu::plugins::scalar_field
