#include "plugin.h"
#include "driver/contributor.h"
#include "interface/update.hpp"
#include "pangu/plugin_api/scalar_field.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <vector>

namespace pangu::plugins::scalar_collapse {
namespace {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
constexpr const char *label = "plugin.scalar_collapse";

// S=integral[R-2(dphi)^2-4V]/(16 pi), ds^2=-a^2dt^2+(dr+a z dt)^2+r^2dOmega^2.
// P=a^-1 phi_t-z phi_r, m=r z^2/2. No black-hole seed or electric charge.
struct Model {
  Real amplitude=.18, center=8, width=1.5, cfl=.2, dissipation=.05;
  Real diagnostic_interval=.25, tolerance=1.e-11;
  int iterations=24;
  plugin_api::scalar_field::PolynomialPotential potential;
  KOKKOS_INLINE_FUNCTION Real Phi0(Real r) const {
    return amplitude*(exp(-pow((r-center)/width,2))+exp(-pow((r+center)/width,2)));
  }
  KOKKOS_INLINE_FUNCTION Real X0(Real r) const {
    return -2*amplitude/(width*width)*((r-center)*exp(-pow((r-center)/width,2))+
                                              (r+center)*exp(-pow((r+center)/width,2)));
  }
  KOKKOS_INLINE_FUNCTION Real Source(Real r, Real phi, Real P, Real X, Real m) const {
    if (r==0) return 0;
    return r*r*(.5*(P*P+X*X)+sqrt(fmax(0.,2*m/r))*P*X+potential.Value(phi*phi));
  }
};

template<class Q> KOKKOS_INLINE_FUNCTION
Real D1(Q q,int i,int lo,int hi,Real h,bool origin,int parity=1) {
  if ((origin || i>=lo+2) && i<=hi-2) {
    const auto at=[&](int n) { return n<lo ? parity*q(2*lo-1-n):q(n); };
    return (at(i-2)-8*at(i-1)+8*at(i+1)-at(i+2))/(12*h);
  }
  if (i<lo+2) {
    if (i==lo) return (-25*q(lo)+48*q(lo+1)-36*q(lo+2)+16*q(lo+3)-3*q(lo+4))/(12*h);
    return (-3*q(lo)-10*q(lo+1)+18*q(lo+2)-6*q(lo+3)+q(lo+4))/(12*h);
  }
  if (i==hi) return (25*q(hi)-48*q(hi-1)+36*q(hi-2)-16*q(hi-3)+3*q(hi-4))/(12*h);
  return (3*q(hi)+10*q(hi-1)-18*q(hi-2)+6*q(hi-3)-q(hi-4))/(12*h);
}

template<class Q> KOKKOS_INLINE_FUNCTION
Real Integral(Q q,int i,int lo,int hi,Real h) {
  if (i==lo) return h*(9*q(lo)+19*q(lo+1)-5*q(lo+2)+q(lo+3))/24;
  if (i==hi-1) return h*(9*q(hi)+19*q(hi-1)-5*q(hi-2)+q(hi-3))/24;
  return h*(-q(i-1)+13*q(i)+13*q(i+1)-q(i+2))/24;
}

int FirstActive(MeshData<Real>* data) {
  const auto mask=data->PackVariables(std::vector<std::string>{"collapse.active"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int j=data->GetBoundsJ(IndexDomain::interior).s,k=data->GetBoundsK(IndexDomain::interior).s;
  int first=ib.e+1;
  parthenon::par_reduce(parthenon::loop_pattern_flatrange_tag,"collapse active boundary",
      DevExecSpace(),ib.s,ib.e,KOKKOS_LAMBDA(int i,int& value) {
        if(mask(0,0,k,j,i)>.5 && i<value) value=i;
      },Kokkos::Min<int>(first));
  PARTHENON_REQUIRE(first<=ib.e-8,"insufficient active radial cells in scalar collapse");
  return first;
}

void Refresh(MeshData<Real>* data) {
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Model>("model");
  auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  auto g=data->PackVariables(std::vector<std::string>{"collapse.geometry"});
  auto w=data->PackVariables(std::vector<std::string>{"collapse.work"});
  auto c=data->PackVariables(std::vector<std::string>{"collapse.constraints"});
  auto phi=data->PackVariables(std::vector<std::string>{"scalar.phi"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int lo=FirstActive(data),hi=ib.e,j=data->GetBoundsJ(IndexDomain::interior).s,
            k=data->GetBoundsK(IndexDomain::interior).s;
  const Real h=data->GetBlockData(0)->GetBlockPointer()->coords.Dxc<X1DIR>(k,j,lo);
  const bool origin=lo==ib.s;
  if(!origin)parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"masked collapse derived fields",
      DevExecSpace(),ib.s,lo-1,KOKKOS_LAMBDA(int i) {
        for(int v=0;v<4;++v)g(0,v,k,j,i)=0;
        for(int v=0;v<5;++v)w(0,v,k,j,i)=0;
        for(int v=0;v<3;++v)c(0,v,k,j,i)=0;
        phi(0,0,k,j,i)=0;
      });
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"collapse scalar gradient",
      DevExecSpace(),lo,hi,KOKKOS_LAMBDA(int i) {
        w(0,0,k,j,i)=D1([&](int n){return u(0,0,k,j,n);},i,lo,hi,h,origin);
      });
  for(int iteration=0;iteration<p.iterations;++iteration) {
    parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"collapse mass source",
        DevExecSpace(),lo,hi,KOKKOS_LAMBDA(int i) {
          w(0,3,k,j,i)=p.Source(u.GetCoords(0).Xc<1>(i),u(0,0,k,j,i),u(0,1,k,j,i),
                              w(0,0,k,j,i),u(0,2,k,j,i));
        });
    Kokkos::parallel_scan("collapse radial constraint",Kokkos::RangePolicy<DevExecSpace>(0,hi-lo+1),
        KOKKOS_LAMBDA(int n,Real& sum,bool final) {
          const int i=lo+n;
          if(n>0) sum+=Integral([&](int ii){return w(0,3,k,j,ii);},i-1,lo,hi,h);
          if(final) {
            // Integral from the regular center to the first cell center,
            // exact for r^2,r^4,r^6. This fixes m(0)=0 without a seed mass.
            const Real anchor=origin ? h*(299.*w(0,3,k,j,lo)/1680-
                43.*w(0,3,k,j,lo+1)/30240+w(0,3,k,j,lo+2)/16800):u(0,2,k,j,lo);
            const Real next=anchor+sum;
            w(0,4,k,j,i)=next-u(0,2,k,j,i);
            u(0,2,k,j,i)=next;
          }
        });
  }
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"collapse lapse source",
      DevExecSpace(),lo,hi,KOKKOS_LAMBDA(int i) {
        const Real r=u.GetCoords(0).Xc<1>(i),z=sqrt(fmax(0.,2*u(0,2,k,j,i)/r));
        g(0,1,k,j,i)=z;
        w(0,1,k,j,i)=z>0 ? -r*u(0,1,k,j,i)*w(0,0,k,j,i)/z:0;
      });
  Kokkos::parallel_scan("collapse lapse constraint",Kokkos::RangePolicy<DevExecSpace>(0,hi-lo+1),
      KOKKOS_LAMBDA(int n,Real& sum,bool final) {
        const int i=hi-n;
        if(i<hi) sum+=Integral([&](int ii){return w(0,1,k,j,ii);},i,lo,hi,h);
        if(final) g(0,0,k,j,i)=exp(-sum);
      });
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"collapse geometry and flux",
      DevExecSpace(),lo,hi,KOKKOS_LAMBDA(int i) {
        const Real r=u.GetCoords(0).Xc<1>(i),z=g(0,1,k,j,i),a=g(0,0,k,j,i);
        g(0,2,k,j,i)=1-z*z;g(0,3,k,j,i)=u(0,2,k,j,i);
        w(0,2,k,j,i)=r*r*a*(w(0,0,k,j,i)+z*u(0,1,k,j,i));
        phi(0,0,k,j,i)=u(0,0,k,j,i);
      });
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"collapse independent constraints",
      DevExecSpace(),lo,hi,KOKKOS_LAMBDA(int i) {
        const Real r=u.GetCoords(0).Xc<1>(i);
        c(0,0,k,j,i)=D1([&](int n){return u(0,2,k,j,n);},i,lo,hi,h,origin,-1)-
            p.Source(r,u(0,0,k,j,i),u(0,1,k,j,i),w(0,0,k,j,i),u(0,2,k,j,i));
        c(0,1,k,j,i)=D1([&](int n){return g(0,0,k,j,n);},i,lo,hi,h,origin)/g(0,0,k,j,i)-w(0,1,k,j,i);
        c(0,2,k,j,i)=0;
      });
}

void InitializeFields(Mesh*,ParameterInput*,MeshData<Real>* data) {
  PARTHENON_REQUIRE(data->NumBlocks()==1,"scalar collapse requires one radial block");
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Model>("model");
  auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  auto mask=data->PackVariables(std::vector<std::string>{"collapse.active"});
  auto g=data->PackVariables(std::vector<std::string>{"collapse.geometry"});
  auto c=data->PackVariables(std::vector<std::string>{"collapse.constraints"});
  auto phi=data->PackVariables(std::vector<std::string>{"scalar.phi"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int j=data->GetBoundsJ(IndexDomain::interior).s,k=data->GetBoundsK(IndexDomain::interior).s;
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"regular collapse initial scalar",
      DevExecSpace(),ib.s,ib.e,KOKKOS_LAMBDA(int i) {
        const Real r=u.GetCoords(0).Xc<1>(i);
        u(0,0,k,j,i)=p.Phi0(r);u(0,1,k,j,i)=0;mask(0,0,k,j,i)=1;
        for(int v=0;v<3;++v)c(0,v,k,j,i)=0;
        for(int v=0;v<4;++v)g(0,v,k,j,i)=0;
        phi(0,0,k,j,i)=u(0,0,k,j,i);
      });
  Kokkos::parallel_for("regular initial Hamiltonian constraint",Kokkos::RangePolicy<DevExecSpace>(0,1),
      KOKKOS_LAMBDA(int) {
        Real mass=0,last=0;
        for(int i=ib.s;i<=ib.e;++i) {
          const Real r=u.GetCoords(0).Xc<1>(i),step=r-last;
          const auto source=[&](Real x,Real m){return p.Source(x,p.Phi0(x),0,p.X0(x),m);};
          const Real k1=source(last,mass),k2=source(last+step/2,mass+step*k1/2),
                     k3=source(last+step/2,mass+step*k2/2),k4=source(r,mass+step*k3);
          mass+=step*(k1+2*k2+2*k3+k4)/6;u(0,2,k,j,i)=mass;last=r;
        }
      });
  Refresh(data);
  Real trapped=0;
  parthenon::par_reduce(parthenon::loop_pattern_flatrange_tag,"initial no-horizon check",
      DevExecSpace(),ib.s,ib.e,KOKKOS_LAMBDA(int i,Real& value) {
        value=fmax(value,1-g(0,2,k,j,i));
      },Kokkos::Max<Real>(trapped));
  PARTHENON_REQUIRE(trapped<1,"scalar-collapse initial data already contain a trapped sphere");
}

TaskStatus RHS(MeshData<Real>* data) {
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Model>("model");
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto g=data->PackVariables(std::vector<std::string>{"collapse.geometry"});
  const auto w=data->PackVariables(std::vector<std::string>{"collapse.work"});
  auto f=data->PackVariables(std::vector<std::string>{"collapse.rhs"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int lo=FirstActive(data),hi=ib.e,j=data->GetBoundsJ(IndexDomain::interior).s,
            k=data->GetBoundsK(IndexDomain::interior).s;
  const Real h=data->GetBlockData(0)->GetBlockPointer()->coords.Dxc<X1DIR>(k,j,lo);const bool origin=lo==ib.s;
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"Einstein-scalar collapse RHS",
      DevExecSpace(),lo,hi,KOKKOS_LAMBDA(int i) {
        const Real r=u.GetCoords(0).Xc<1>(i),a=g(0,0,k,j,i),z=g(0,1,k,j,i),P=u(0,1,k,j,i),X=w(0,0,k,j,i);
        f(0,0,k,j,i)=a*(P+z*X);
        f(0,1,k,j,i)=D1([&](int n){return w(0,2,k,j,n);},i,lo,hi,h,origin,-1)/(r*r)-
            a*p.potential.Gradient(u(0,0,k,j,i),pow(u(0,0,k,j,i),2));
        f(0,2,k,j,i)=r*r*a*(P+z*X)*(z*P+X);
        if(i>hi-2) {
          const Real speed=a*(1-z);
          f(0,0,k,j,i)=-speed*(X+u(0,0,k,j,i)/r);
          f(0,1,k,j,i)=-speed*(D1([&](int n){return u(0,1,k,j,n);},i,lo,hi,h,false)+P/r);
        }
        if((origin || i>=lo+3) && i<=hi-3)
          for(int v=0;v<2;++v) {
            const auto at=[&](int n){return u(0,v,k,j,n<lo?2*lo-1-n:n);};
            const Real d6=at(i-3)-6*at(i-2)+15*at(i-1)-20*at(i)+15*at(i+1)-6*at(i+2)+at(i+3);
            f(0,v,k,j,i)+=p.dissipation*d6/(64*h);
          }
      });
  return TaskStatus::complete;
}

TaskStatus Update(MeshData<Real>* data,int stage,Real dt) {
  auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto f=data->PackVariables(std::vector<std::string>{"collapse.rhs"});
  auto initial=data->PackVariables(std::vector<std::string>{"collapse.initial"});
  auto sum=data->PackVariables(std::vector<std::string>{"collapse.sum"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int lo=FirstActive(data),j=data->GetBoundsJ(IndexDomain::interior).s,k=data->GetBoundsK(IndexDomain::interior).s;
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"collapse classical RK4",
      DevExecSpace(),lo,ib.e,KOKKOS_LAMBDA(int i) {
        for(int v=0;v<3;++v) {
          if(stage==1){initial(0,v,k,j,i)=u(0,v,k,j,i);sum(0,v,k,j,i)=0;}
          sum(0,v,k,j,i)+=(stage==1||stage==4?1./6:1./3)*f(0,v,k,j,i);
          u(0,v,k,j,i)=initial(0,v,k,j,i)+dt*(stage==4?sum(0,v,k,j,i):(stage<3?.5:1)*f(0,v,k,j,i));
        }
      });return TaskStatus::complete;
}

TaskStatus CompleteStep(MeshData<Real>* data,Real dt) {
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto g=data->PackVariables(std::vector<std::string>{"collapse.geometry"});
  const auto initial=data->PackVariables(std::vector<std::string>{"collapse.initial"});
  const auto sum=data->PackVariables(std::vector<std::string>{"collapse.sum"});
  auto c=data->PackVariables(std::vector<std::string>{"collapse.constraints"});
  auto mask=data->PackVariables(std::vector<std::string>{"collapse.active"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int lo=FirstActive(data),hi=ib.e,j=data->GetBoundsJ(IndexDomain::interior).s,
            k=data->GetBoundsK(IndexDomain::interior).s;
  int next=lo;
  parthenon::par_reduce(parthenon::loop_pattern_flatrange_tag,"causal excision candidate",
      DevExecSpace(),lo,hi-8,KOKKOS_LAMBDA(int i,int& value) {
        if(g(0,2,k,j,i)<-.1 && i>value)value=i;
      },Kokkos::Max<int>(next));
  // A max reduction with no candidate returns the reducer identity.
  next=std::max(lo,next);
  if(next>lo) {
    parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"activate trapped-region excision",
        DevExecSpace(),lo,next-1,KOKKOS_LAMBDA(int i){mask(0,0,k,j,i)=0;});
    Refresh(data);
  }
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"collapse temporal Einstein residual",
      DevExecSpace(),next,hi,KOKKOS_LAMBDA(int i) {
        c(0,2,k,j,i)=(u(0,2,k,j,i)-initial(0,2,k,j,i))/dt-sum(0,2,k,j,i);
      });
  return TaskStatus::complete;
}

Real Timestep(MeshData<Real>* data) {
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Model>("model");
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto g=data->PackVariables(std::vector<std::string>{"collapse.geometry"});
  const auto w=data->PackVariables(std::vector<std::string>{"collapse.work"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int lo=FirstActive(data),j=data->GetBoundsJ(IndexDomain::interior).s,k=data->GetBoundsK(IndexDomain::interior).s;
  const Real h=data->GetBlockData(0)->GetBlockPointer()->coords.Dxc<X1DIR>(k,j,lo);Real rate=0;
  parthenon::par_reduce(parthenon::loop_pattern_flatrange_tag,"collapse causal timestep and health",
      DevExecSpace(),lo,ib.e,KOKKOS_LAMBDA(int i,Real& value) {
        const Real a=g(0,0,k,j,i),z=g(0,1,k,j,i),phi=u(0,0,k,j,i),P=u(0,1,k,j,i),m=u(0,2,k,j,i);
        const Real mass_scale=fmax(fabs(u(0,2,k,j,ib.e)),1.e-300);
        const Real frequency=sqrt(p.potential.mass_squared+6*p.potential.quartic*phi*phi+
                                   5*p.potential.sextic*pow(phi,4));
        const bool good=isfinite(a)&&a>0&&isfinite(z)&&isfinite(phi)&&isfinite(P)&&isfinite(m)&&
            m>=-128*std::numeric_limits<Real>::epsilon()*mass_scale &&
            fabs(w(0,4,k,j,i))<=p.tolerance*(1+fabs(m)) &&
            (i!=lo || lo==ib.s || z>1);
        value=good?fmax(value,a*((1+z)/h+frequency)):INFINITY;
      },Kokkos::Max<Real>(rate));
  PARTHENON_REQUIRE(std::isfinite(rate)&&rate>0,
                    "collapse state nonfinite, constraint unconverged, negative mass, or excision untrapped");
  return p.cfl/rate;
}

void Diagnostics(Mesh* mesh,Real time,bool force=false) {
  auto package=mesh->packages.Get(label);const auto p=package->Param<Model>("model");
  if(!force && time-package->Param<Real>("last_diagnostic")<p.diagnostic_interval*(1-1e-10))return;
  auto& data=mesh->mesh_data.Get();const int lo=FirstActive(data.get());
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto g=data->PackVariables(std::vector<std::string>{"collapse.geometry"});
  const auto w=data->PackVariables(std::vector<std::string>{"collapse.work"});
  const auto c=data->PackVariables(std::vector<std::string>{"collapse.constraints"});
  const auto ib=data->GetBoundsI(IndexDomain::interior);
  const int hi=ib.e,j=data->GetBoundsJ(IndexDomain::interior).s,k=data->GetBoundsK(IndexDomain::interior).s;
  Kokkos::View<Real**> snap("collapse diagnostic samples",hi-lo+1,8);
  parthenon::par_for(parthenon::loop_pattern_flatrange_tag,"collapse diagnostic copy",
      DevExecSpace(),lo,hi,KOKKOS_LAMBDA(int i) {
        const int n=i-lo;const Real r=u.GetCoords(0).Xc<1>(i);
        snap(n,0)=r;snap(n,1)=u(0,0,k,j,i);snap(n,2)=u(0,2,k,j,i);snap(n,3)=g(0,2,k,j,i);
        snap(n,4)=c(0,0,k,j,i);snap(n,5)=c(0,1,k,j,i);snap(n,6)=c(0,2,k,j,i);
        const Real a=g(0,0,k,j,i),z=g(0,1,k,j,i),P=u(0,1,k,j,i),X=w(0,0,k,j,i);
        snap(n,7)=-r*r*a*(P+z*X)*(z*P+X);
      });
  const auto s=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),snap);
  const int n=s.extent(0);Real rh=0,ph=0,max_compactness=0,max_c=0,max_a=0,max_t=0;
  for(int i=0;i<n;++i) {
    for(int v=0;v<8;++v)PARTHENON_REQUIRE(std::isfinite(s(i,v)),"nonfinite collapse diagnostic");
    max_compactness=std::max(max_compactness,1-s(i,3));
    if(i+1<n && s(i,3)<=0 && s(i+1,3)>0) {
      const int start=std::max(0,std::min(n-4,i-1));
      const auto interpolate=[&](Real r,int component) {
        Real value=0;
        for(int a=start;a<start+4;++a){Real weight=1;
          for(int b=start;b<start+4;++b)if(a!=b)weight*=(r-s(b,0))/(s(a,0)-s(b,0));
          value+=weight*s(a,component);
        }return value;
      };
      Real left=s(i,0),right=s(i+1,0);
      for(int step=0;step<40;++step){const Real middle=(left+right)/2;
        if(interpolate(middle,3)<=0)left=middle;else right=middle;
      }
      rh=(left+right)/2;ph=interpolate(rh,1);
    }
  }
  PARTHENON_REQUIRE(lo==ib.s || rh>0,"excised collapse domain has no outer apparent horizon");
  for(int i=3;i+3<n;++i)if(s(i,0)>rh) {
    max_c=std::max(max_c,fabs(s(i,4)));max_a=std::max(max_a,fabs(s(i,5)));max_t=std::max(max_t,fabs(s(i,6)));
  }
  std::ofstream out(package->Param<std::string>("history_file"),std::ios::app);
  PARTHENON_REQUIRE(out.good(),"cannot open scalar-collapse history");
  out<<std::setprecision(17)<<time<<','<<(rh>0)<<','<<rh<<','<<ph<<','<<rh/2<<','<<max_compactness<<','
     <<lo-ib.s<<','<<s(0,0)<<','<<s(n-1,2)<<','<<s(n-1,7)<<','<<max_c<<','<<max_a<<','<<max_t<<'\n';
  out.flush();PARTHENON_REQUIRE(out.good(),"cannot write scalar-collapse history");
  package->UpdateParam<Real>("last_diagnostic",time);
}

void BeforeLoop(Mesh* mesh,ParameterInput*,parthenon::SimTime& tm) {
  const auto path=mesh->packages.Get(label)->Param<std::string>("history_file");
  std::ifstream old(path);std::vector<std::string> retained;std::string line;
  if(old.good()){std::getline(old,line);while(std::getline(old,line)){
    std::istringstream in(line);Real t;if(in>>t && t<tm.time-1.e-12)retained.push_back(line);
  }}old.close();std::ofstream out(path);
  out<<"time,horizon_found,horizon_radius,horizon_phi,irreducible_mass,max_compactness,excised_cells,inner_radius,outer_mass,outgoing_luminosity,mass_constraint_max,lapse_constraint_max,einstein_time_constraint_max\n";
  for(const auto& row:retained)out<<row<<'\n';out.close();Diagnostics(mesh,tm.time,true);
}

void PostStep(Mesh* mesh,const parthenon::SimTime& tm) {
  const Real time=tm.time+tm.dt;
  Diagnostics(mesh,time,time>=tm.tlim-1.e-12*std::max(1.,tm.tlim));
}

void Configure(Mesh* mesh,parthenon::LowStorageIntegrator& rk) {
  PARTHENON_REQUIRE(mesh->ndim==1&&mesh->nbtotal==1&&parthenon::Globals::nranks==1&&!mesh->multilevel,
                    "scalar collapse requires a uniform one-block/rank radial mesh");
  PARTHENON_REQUIRE(rk.GetName()=="rk4","scalar collapse requires classical RK4");
  rk.nstages=4;rk.nbuffers=2;rk.gam0={0,0,0,0};rk.gam1={1,1,1,1};
  rk.beta={.5,.5,1,1./6};rk.delta={0,0,0,0};rk.c={0,.5,.5,1};
  rk.buffer_name={"base","base","base"};rk.stage_name={"base","base","base","base","base"};
}

TaskCollection Build(driver::StageBuildContext& context) {
  auto* mesh=context.mesh;const auto partitions=mesh->GetDefaultBlockPartitions();
  TaskCollection collection;auto& tasks=collection.AddRegion(1)[0];auto& data=mesh->mesh_data.Add("base",partitions.front());
  const auto rhs=tasks.AddTask(TaskID(0),RHS,data.get());
  const auto update=tasks.AddTask(rhs,Update,data.get(),context.stage,context.integrator->dt);
  const auto refresh=tasks.AddTask(update,parthenon::Update::FillDerived<MeshData<Real>>,data.get());
  if(context.stage==4) {
    const auto complete=tasks.AddTask(refresh,CompleteStep,data.get(),context.integrator->dt);
    tasks.AddTask(complete,parthenon::Update::EstimateTimestep<MeshData<Real>>,data.get());
  }
  return collection;
}
} // namespace

std::shared_ptr<parthenon::StateDescriptor> Plugin::Initialize(plugin::InitContext& context) {
  auto& pin=context.input;const std::string section="plugin/scalar_collapse";
  const std::set<std::string> keys{"amplitude","center","width","cfl","dissipation","diagnostic_interval",
      "constraint_iterations","constraint_tolerance","mass","quartic","sextic","history_file"};
  for(const auto& key:pin.GetParameterNames(section))
    PARTHENON_REQUIRE(keys.count(key)!=0,"unknown scalar_collapse parameter: "+key);
  PARTHENON_REQUIRE(pin.GetString("parthenon/job","problem_id")=="scalar_collapse",
                    "scalar_collapse requires problem_id=scalar_collapse");
  PARTHENON_REQUIRE(pin.GetReal("parthenon/mesh","x1min")==0 &&
                        pin.GetReal("parthenon/mesh","x1max")>0 &&
                        pin.GetInteger("parthenon/mesh","nx1")>=32 &&
                        pin.GetInteger("parthenon/meshblock","nx1")==pin.GetInteger("parthenon/mesh","nx1"),
                    "scalar collapse requires the regular center at x1min=0 and one radial block");
  Model p;
  p.amplitude=pin.GetOrAddReal(section,"amplitude",.18);p.center=pin.GetOrAddReal(section,"center",8);
  p.width=pin.GetOrAddReal(section,"width",1.5);p.cfl=pin.GetOrAddReal(section,"cfl",.2);
  p.dissipation=pin.GetOrAddReal(section,"dissipation",.05);
  p.diagnostic_interval=pin.GetOrAddReal(section,"diagnostic_interval",.25);
  p.iterations=pin.GetOrAddInteger(section,"constraint_iterations",24);
  p.tolerance=pin.GetOrAddReal(section,"constraint_tolerance",1.e-11);
  const Real mass=pin.GetOrAddReal(section,"mass",0),quartic=pin.GetOrAddReal(section,"quartic",0),
             sextic=pin.GetOrAddReal(section,"sextic",0);
  p.potential={mass*mass,quartic,sextic};
  for(Real x:{p.amplitude,p.center,p.width,p.cfl,p.dissipation,p.diagnostic_interval,p.tolerance,mass,quartic,sextic})
    PARTHENON_REQUIRE(std::isfinite(x),"nonfinite scalar-collapse parameter");
  PARTHENON_REQUIRE(p.center>=0&&p.width>0&&p.cfl>0&&p.cfl<=.4&&p.dissipation>=0&&p.dissipation<=1&&
      p.diagnostic_interval>0&&p.tolerance>0&&p.iterations>=2&&p.iterations<=64&&mass>=0&&quartic>=0&&sextic>=0,
      "invalid scalar-collapse parameters");
  auto package=std::make_shared<StateDescriptor>(label);package->AddParam<Model>("model",p);
  package->AddParam<Real>("last_diagnostic",-std::numeric_limits<Real>::max(),true);
  package->AddParam<std::string>("history_file",pin.GetOrAddString(section,"history_file","collapse_history.csv"));
  package->AddParam<driver::StageContribution>(std::string(driver::stage_contribution_key),
      {"scalar_collapse",200,[](const driver::StageBuildContext&){return true;},Build,Configure,PostStep});
  const auto derived=[](int n){return Metadata({Metadata::Cell,Metadata::Derived,Metadata::OneCopy},std::vector<int>{n});};
  package->AddField("scalar.state",Metadata({Metadata::Cell,Metadata::Independent,Metadata::Restart},std::vector<int>{3}));
  package->AddField("collapse.active",Metadata({Metadata::Cell,Metadata::Independent,Metadata::Restart}));
  package->AddField("scalar.phi",derived(1));package->AddField("collapse.geometry",derived(4));
  package->AddField("collapse.work",derived(5));package->AddField("collapse.constraints",derived(3));
  package->AddField("collapse.rhs",derived(3));package->AddField("collapse.initial",derived(3));package->AddField("collapse.sum",derived(3));
  package->PostProblemGeneratorMesh=InitializeFields;package->FillDerivedMesh=Refresh;
  package->EstimateTimestepMesh=Timestep;package->UserWorkBeforeLoopMesh=BeforeLoop;
  return package;
}
} // namespace pangu::plugins::scalar_collapse
