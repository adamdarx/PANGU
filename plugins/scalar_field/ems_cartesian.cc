#include "ems_coupled.h"
#include "ems_stress_energy.h"

#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pangu/plugin_api/synchronized_matter.h"
#include "z4c/core/finite_difference.h"
#include "z4c/evolution/classical_rk4.h"
#include "z4c/evolution/package.h"

namespace pangu::plugins::scalar_field {
using namespace ems;
using namespace parthenon::package::prelude;
namespace {
constexpr const char* label = "plugin.scalar_field";
struct Parameters {
  Model model;
  int order=4, initial=0;
  Real amplitude=0, wave[3]{1,1,1}, polarization[3]{1,-1,0};
  Real mass=1, charge=0, cfl=0.25, dissipation=0;
};

template <class Pack> struct Value {
  Pack pack;
  int block, component, k,j,i;
  KOKKOS_INLINE_FUNCTION Real operator()(int dk,int dj,int di) const {
    return pack(block,component,k+dk,j+dj,i+di);
  }
};

template <class APack,class ZPack>
KOKKOS_INLINE_FUNCTION Geometry GetGeometry(const APack& a,const ZPack& z,
                                           int b,int k,int j,int i) {
  Geometry g;
  for(int n=0;n<6;++n) g.metric[n]=a(b,n,k,j,i);
  g.Invert();
  g.lapse=z(b,nr::Index(nr::Z4cComponent::alpha),k,j,i);
  g.trace_k=z(b,nr::Index(nr::Z4cComponent::khat),k,j,i)+
             2*z(b,nr::Index(nr::Z4cComponent::theta),k,j,i);
  for(int n=0;n<3;++n) g.shift[n]=z(b,nr::Index(nr::Z4cComponent::betax)+n,k,j,i);
  return g;
}

template <class Pack>
KOKKOS_INLINE_FUNCTION void Spacing(const Pack& pack,int b,int k,int j,int i,Real h[3]) {
  const auto& c=pack.GetCoords(b);
  h[0]=1/c.template Dxc<X1DIR>(k,j,i);
  h[1]=1/c.template Dxc<X2DIR>(k,j,i);
  h[2]=1/c.template Dxc<X3DIR>(k,j,i);
}

template<int Order> TaskStatus PrepareOrder(MeshData<Real>* data,Real) {
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Parameters>("parameters");
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto z=data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto a=data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto flux=data->PackVariables(std::vector<std::string>{"ems.flux"});
  const auto phi_field=data->PackVariables(std::vector<std::string>{"scalar.phi"});
  const auto tm=data->PackVariables(std::vector<std::string>{"nr.tmunu"});
  const auto constraints=data->PackVariables(std::vector<std::string>{"ems.gauss"});
  const auto ei=data->GetBoundsI(IndexDomain::entire),ej=data->GetBoundsJ(IndexDomain::entire),
             ek=data->GetBoundsK(IndexDomain::entire);
  parthenon::par_for(DEFAULT_LOOP_PATTERN,"3D EMS Maxwell fluxes",DevExecSpace(),
      0,data->NumBlocks()-1,ek.s,ek.e,ej.s,ej.e,ei.s,ei.e,
      KOKKOS_LAMBDA(int b,int k,int j,int i) {
        phi_field(b,0,k,j,i)=u(b,0,k,j,i);
        const auto g=GetGeometry(a,z,b,k,j,i);
        const Real f=p.model.F(u(b,0,k,j,i));
        Real d[3],c[3];
        for(int n=0;n<3;++n) { d[n]=u(b,2+n,k,j,i); c[n]=u(b,5+n,k,j,i); }
        const auto em=PhysicalFields(g,f,d,c);
        for(int n=0;n<3;++n) for(int axis=0;axis<3;++axis) {
          Real df=g.shift[axis]*d[n]-g.shift[n]*d[axis];
          Real bf=g.shift[axis]*c[n]-g.shift[n]*c[axis];
          for(int l=0;l<3;++l) {
            df+=g.lapse*f*Epsilon(n,axis,l)*em.b_lower[l];
            bf-=g.lapse*Epsilon(n,axis,l)*em.e_lower[l];
          }
          flux(b,3*n+axis,k,j,i)=df;
          flux(b,9+3*n+axis,k,j,i)=bf;
        }
      });
  const auto ib=data->GetBoundsI(IndexDomain::interior),jb=data->GetBoundsJ(IndexDomain::interior),
             kb=data->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(DEFAULT_LOOP_PATTERN,"3D EMS Einstein source and Gauss constraints",DevExecSpace(),
      0,data->NumBlocks()-1,kb.s,kb.e,jb.s,jb.e,ib.s,ib.e,
      KOKKOS_LAMBDA(int b,int k,int j,int i) {
        const auto g=GetGeometry(a,z,b,k,j,i);
        Real h[3],grad[3],d[3],c[3],divd=0,divb=0;
        Spacing(u,b,k,j,i,h);
        for(int n=0;n<3;++n) {
          grad[n]=nr::fd::First<Order>(n,h[n],Value<decltype(u)>{u,b,0,k,j,i});
          d[n]=u(b,2+n,k,j,i); c[n]=u(b,5+n,k,j,i);
          divd+=nr::fd::First<Order>(n,h[n],Value<decltype(u)>{u,b,2+n,k,j,i});
          divb+=nr::fd::First<Order>(n,h[n],Value<decltype(u)>{u,b,5+n,k,j,i});
        }
        const Real phi=u(b,0,k,j,i);
        const auto em=PhysicalFields(g,p.model.F(phi),d,c);
        const auto sources=Sources(g,p.model,phi,u(b,1,k,j,i),grad,em);
        for(int n=0;n<6;++n) tm(b,n,k,j,i)=sources.stress[n];
        tm(b,6,k,j,i)=sources.energy;
        for(int n=0;n<3;++n) tm(b,7+n,k,j,i)=sources.momentum[n];
        constraints(b,0,k,j,i)=divd/g.sqrt_det;
        constraints(b,1,k,j,i)=divb/g.sqrt_det;
      });
  int invalid=0;
  const Real largest=std::numeric_limits<Real>::max();
  parthenon::par_reduce(DEFAULT_LOOP_PATTERN,"3D EMS finite-state validation",DevExecSpace(),
      0,data->NumBlocks()-1,kb.s,kb.e,jb.s,jb.e,ib.s,ib.e,
      KOKKOS_LAMBDA(int b,int k,int j,int i,int& bad) {
        const Real conformal=z(b,nr::Index(nr::Z4cComponent::chi),k,j,i);
        if(!(conformal>0 && conformal<=largest)) ++bad;
        Real metric[6];for(int n=0;n<6;++n) metric[n]=a(b,n,k,j,i);
        if(!(metric[0]>0 && metric[0]*metric[3]-metric[1]*metric[1]>0 &&
             nr::rhs::SpatialDeterminant(metric)>0)) ++bad;
        for(int n=0;n<8;++n) if(!(fabs(u(b,n,k,j,i))<=largest)) ++bad;
        for(int n=0;n<10;++n) if(!(fabs(tm(b,n,k,j,i))<=largest)) ++bad;
        for(int n=0;n<2;++n) if(!(fabs(constraints(b,n,k,j,i))<=largest)) ++bad;
      },Kokkos::Sum<int>(invalid));
  if(invalid!=0) {
    // Preserve the failed state. Report geometric causes before a NaN in its
    // matter projections obscures a puncture positivity failure.
    Real minimum_chi=largest,minimum_determinant=largest;
    parthenon::par_reduce(DEFAULT_LOOP_PATTERN,"EMS failed-state minimum conformal variable",DevExecSpace(),
        0,data->NumBlocks()-1,kb.s,kb.e,jb.s,jb.e,ib.s,ib.e,
        KOKKOS_LAMBDA(int b,int k,int j,int i,Real& minimum) {
          const Real value=z(b,nr::Index(nr::Z4cComponent::chi),k,j,i);
          if(value<minimum) minimum=value;
        },Kokkos::Min<Real>(minimum_chi));
    parthenon::par_reduce(DEFAULT_LOOP_PATTERN,"EMS failed-state minimum spatial determinant",DevExecSpace(),
        0,data->NumBlocks()-1,kb.s,kb.e,jb.s,jb.e,ib.s,ib.e,
        KOKKOS_LAMBDA(int b,int k,int j,int i,Real& minimum) {
          Real metric[6];for(int n=0;n<6;++n) metric[n]=a(b,n,k,j,i);
          const Real value=nr::rhs::SpatialDeterminant(metric);
          if(value<minimum) minimum=value;
        },Kokkos::Min<Real>(minimum_determinant));
    std::ostringstream reason;
    reason.precision(17);
    reason<<"invalid 3D EMS state, geometry, stress energy or Gauss constraint; "
          <<"minimum conformal variable="<<minimum_chi
          <<"; minimum spatial metric determinant="<<minimum_determinant;
    PARTHENON_FAIL(reason.str());
  }
  return TaskStatus::complete;
}

TaskStatus Prepare(MeshData<Real>* data,Real time) {
  const int order=data->GetParentPointer()->packages.Get(label)->Param<Parameters>("parameters").order;
  if(order==2) return PrepareOrder<2>(data,time);
  if(order==4) return PrepareOrder<4>(data,time);
  return PrepareOrder<6>(data,time);
}

template<int Order> TaskStatus RHSOrder(MeshData<Real>* data) {
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Parameters>("parameters");
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto z=data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto a=data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto flux=data->PackVariables(std::vector<std::string>{"ems.flux"});
  const auto rhs=data->PackVariables(std::vector<std::string>{"scalar.rhs"});
  const auto ib=data->GetBoundsI(IndexDomain::interior),jb=data->GetBoundsJ(IndexDomain::interior),
             kb=data->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(DEFAULT_LOOP_PATTERN,"3D coupled Maxwell scalar RHS",DevExecSpace(),
      0,data->NumBlocks()-1,kb.s,kb.e,jb.s,jb.e,ib.s,ib.e,
      KOKKOS_LAMBDA(int b,int k,int j,int i) {
        const auto g=GetGeometry(a,z,b,k,j,i);
        Real h[3],grad[3],gradp[3],da[3],dg[3][6],d[3],c[3];
        Spacing(u,b,k,j,i,h);
        for(int axis=0;axis<3;++axis) {
          grad[axis]=nr::fd::First<Order>(axis,h[axis],Value<decltype(u)>{u,b,0,k,j,i});
          gradp[axis]=nr::fd::First<Order>(axis,h[axis],Value<decltype(u)>{u,b,1,k,j,i});
          da[axis]=nr::fd::First<Order>(axis,h[axis],Value<decltype(z)>{z,b,18,k,j,i});
          for(int n=0;n<6;++n)
            dg[axis][n]=nr::fd::First<Order>(axis,h[axis],Value<decltype(a)>{a,b,n,k,j,i});
          d[axis]=u(b,2+axis,k,j,i); c[axis]=u(b,5+axis,k,j,i);
        }
        Real lap=0,grad_lapse=0,adv_phi=0,adv_p=0;
        for(int first=0;first<3;++first) {
          adv_phi+=g.shift[first]*grad[first]; adv_p+=g.shift[first]*gradp[first];
          for(int second=0;second<3;++second) {
            const Real dd=first==second ? nr::fd::Second<Order>(first,h[first],
                Value<decltype(u)>{u,b,0,k,j,i}) : nr::fd::MixedSecond<Order>(
                first,second,h[first],h[second],Value<decltype(u)>{u,b,0,k,j,i});
            lap+=g.Inv(first,second)*dd;
            grad_lapse+=g.Inv(first,second)*da[first]*grad[second];
            for(int upper=0;upper<3;++upper) for(int l=0;l<3;++l) {
              const Real connection=0.5*g.Inv(upper,l)*(
                  dg[first][nr::SpatialSymmetricComponent(second,l)]+
                  dg[second][nr::SpatialSymmetricComponent(first,l)]-
                  dg[l][nr::SpatialSymmetricComponent(first,second)]);
              lap-=g.Inv(first,second)*connection*grad[upper];
            }
          }
        }
        const Real phi=u(b,0,k,j,i),momentum=u(b,1,k,j,i);
        const auto em=PhysicalFields(g,p.model.F(phi),d,c);
        rhs(b,0,k,j,i)=adv_phi+g.lapse*momentum;
        rhs(b,1,k,j,i)=adv_p+g.lapse*(g.trace_k*momentum+lap-
                         ScalarForce(p.model,phi,em))+grad_lapse;
        for(int n=0;n<6;++n) {
          Real value=0;
          for(int axis=0;axis<3;++axis)
            value+=nr::fd::First<Order>(axis,h[axis],Value<decltype(flux)>{flux,b,3*n+axis,k,j,i});
          rhs(b,2+n,k,j,i)=value;
        }
        if(p.dissipation!=0) for(int n=0;n<8;++n) {
          Real ko=0;
          for(int axis=0;axis<3;++axis)
            ko+=nr::fd::KreissOliger<Order>(axis,h[axis],Value<decltype(u)>{u,b,n,k,j,i});
          rhs(b,n,k,j,i)+=p.dissipation*(Order==4 ? 1.0 : -1.0)*ko/(1<<(Order+2));
        }
      });
  return TaskStatus::complete;
}

TaskStatus RHS(MeshData<Real>* data) {
  const int order=data->GetParentPointer()->packages.Get(label)->Param<Parameters>("parameters").order;
  if(order==2) return RHSOrder<2>(data);
  if(order==4) return RHSOrder<4>(data);
  return RHSOrder<6>(data);
}

TaskStatus Update(MeshData<Real>* data,int stage,Real dt) {
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto rhs=data->PackVariables(std::vector<std::string>{"scalar.rhs"});
  const auto initial=data->PackVariables(std::vector<std::string>{"scalar.initial"});
  const auto sum=data->PackVariables(std::vector<std::string>{"scalar.rk_sum"});
  const auto ib=data->GetBoundsI(IndexDomain::interior),jb=data->GetBoundsJ(IndexDomain::interior),
             kb=data->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(DEFAULT_LOOP_PATTERN,"3D EMS classical RK4 update",DevExecSpace(),
      0,data->NumBlocks()-1,0,7,kb.s,kb.e,jb.s,jb.e,ib.s,ib.e,
      KOKKOS_LAMBDA(int b,int n,int k,int j,int i) {
        if(stage==1) initial(b,n,k,j,i)=u(b,n,k,j,i);
        const Real r=rhs(b,n,k,j,i);
        const Real s=nr::classical_rk4::Accumulate(stage,r,sum(b,n,k,j,i));
        sum(b,n,k,j,i)=s;
        u(b,n,k,j,i)=nr::classical_rk4::Update(stage,initial(b,n,k,j,i),r,s,dt);
      });
  return TaskStatus::complete;
}

void InitializeFields(Mesh* mesh,ParameterInput*,MeshData<Real>* data) {
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Parameters>("parameters");
  const auto nr_package=data->GetParentPointer()->packages.Get("numerical_relativity");
  const Real chi_power=nr_package->Param<nr::Z4cOptions>("z4c_options").chi_psi_power;
  const Real inner_mask=nr_package->Param<Real>("tracked_constraint_inner_radius");
  const Real outer_mask=nr_package->Param<Real>("constraint_mask_outer_radius");
  const auto u=data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto z=data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto mask=data->PackVariables(std::vector<std::string>{"nr.constraint_mask"});
  const auto ib=data->GetBoundsI(IndexDomain::entire),jb=data->GetBoundsJ(IndexDomain::entire),
             kb=data->GetBoundsK(IndexDomain::entire);
  // Project Maxwell polarization against the centered derivative's Fourier
  // symbol, so oblique waves satisfy the discrete Gauss constraints initially.
  Kokkos::Array<Real,3> curl_wave{p.wave[0],p.wave[1],p.wave[2]};
  if(p.initial==2) {
    const auto& size=mesh->mesh_size;
    const Real dx[3]={(size.xmax(X1DIR)-size.xmin(X1DIR))/size.nx(X1DIR),
                     (size.xmax(X2DIR)-size.xmin(X2DIR))/size.nx(X2DIR),
                     (size.xmax(X3DIR)-size.xmin(X3DIR))/size.nx(X3DIR)};
    for(int n=0;n<3;++n) {
      const Real angle=p.wave[n]*dx[n];
      PARTHENON_REQUIRE(fabs(angle)<acos(-1.0),"Maxwell wave must lie below the grid Nyquist frequency");
      curl_wave[n]=(p.order==2 ? sin(angle) : (p.order==4 ?
          (8*sin(angle)-sin(2*angle))/6 :
          (45*sin(angle)-9*sin(2*angle)+sin(3*angle))/30))/dx[n];
    }
  }
  Real norm=0,curl_norm=0,polnorm=0,dot=0;
  for(int i=0;i<3;++i) {
    norm+=p.wave[i]*p.wave[i]; curl_norm+=curl_wave[i]*curl_wave[i];
    dot+=curl_wave[i]*p.polarization[i];
  }
  Real pol[3];
  for(int i=0;i<3;++i) { pol[i]=p.polarization[i]-curl_wave[i]*dot/curl_norm; polnorm+=pol[i]*pol[i]; }
  PARTHENON_REQUIRE(curl_norm>0 && polnorm>0,"Maxwell wave polarization is unresolved on this grid");
  norm=sqrt(norm); curl_norm=sqrt(curl_norm); polnorm=sqrt(polnorm);
  Kokkos::Array<Real,3> polarization{pol[0]/polnorm,pol[1]/polnorm,pol[2]/polnorm};
  parthenon::par_for(DEFAULT_LOOP_PATTERN,"3D EMS initial fields",DevExecSpace(),
      0,data->NumBlocks()-1,kb.s,kb.e,jb.s,jb.e,ib.s,ib.e,
      KOKKOS_LAMBDA(int b,int k,int j,int i) {
        const auto& coords=u.GetCoords(b);
        const Real x[3]={coords.Xc<X1DIR>(k,j,i),coords.Xc<X2DIR>(k,j,i),coords.Xc<X3DIR>(k,j,i)};
        const Real radius_squared=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
        // The scalar problem starts from Minkowski's all-active mask. Apply the
        // configured NR diagnostic shell when replacing those initial fields.
        mask(b,0,k,j,i)=radius_squared>inner_mask*inner_mask &&
            (outer_mask==0 || radius_squared<=outer_mask*outer_mask) ? 1.0 : 0.0;
        for(int n=0;n<8;++n) u(b,n,k,j,i)=0;
        Real phase=0; for(int n=0;n<3;++n) phase+=p.wave[n]*x[n];
        if(p.initial==1) {
          u(b,0,k,j,i)=p.amplitude*cos(phase);
          u(b,1,k,j,i)=p.amplitude*sqrt(norm*norm+p.model.polynomial.mass_squared)*sin(phase);
        } else if(p.initial==2) {
          for(int n=0;n<3;++n) {
            u(b,2+n,k,j,i)=p.amplitude*polarization[n]*cos(phase);
            for(int first=0;first<3;++first) for(int second=0;second<3;++second)
              u(b,5+n,k,j,i)+=p.amplitude*Epsilon(n,first,second)*curl_wave[first]/curl_norm*
                                        polarization[second]*cos(phase);
          }
        } else if(p.initial==3) {
          const Real r=sqrt(radius_squared);
          if (!(r>0)) Kokkos::abort("RN puncture must not coincide with a cell center");
          const Real psi=sqrt(1+p.mass/r+(p.mass*p.mass-p.charge*p.charge)/(4*r*r));
          z(b,0,k,j,i)=pow(psi,chi_power);
          z(b,18,k,j,i)=1/(psi*psi);
          for(int n=0;n<3;++n) u(b,2+n,k,j,i)=p.charge*x[n]/(r*r*r);
        }
      });
}

Real Timestep(MeshData<Real>* data) {
  const auto p=data->GetParentPointer()->packages.Get(label)->Param<Parameters>("parameters");
  const auto a=data->PackVariables(std::vector<std::string>{"nr.adm"});
  const auto z=data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto ib=data->GetBoundsI(IndexDomain::interior),jb=data->GetBoundsJ(IndexDomain::interior),
             kb=data->GetBoundsK(IndexDomain::interior);
  Real maximum=0;
  parthenon::par_reduce(DEFAULT_LOOP_PATTERN,"3D EMS causal timestep",DevExecSpace(),
      0,data->NumBlocks()-1,kb.s,kb.e,jb.s,jb.e,ib.s,ib.e,
      KOKKOS_LAMBDA(int b,int k,int j,int i,Real& maxrate) {
        const auto g=GetGeometry(a,z,b,k,j,i);
        Real h[3],rate=0; Spacing(a,b,k,j,i,h);
        for(int n=0;n<3;++n) rate+=(fabs(g.shift[n])+g.lapse*sqrt(g.Inv(n,n)))*h[n];
        if(rate>maxrate) maxrate=rate;
      },Kokkos::Max<Real>(maximum));
  return p.cfl/maximum;
}
} // namespace

std::shared_ptr<StateDescriptor> InitializeCartesianEMS(plugin::InitContext& context) {
  auto& pin=context.input;
  const std::string block="plugin/scalar_field";
  PARTHENON_REQUIRE(pin.GetString("parthenon/job","problem_id")=="scalar_field",
                    "Cartesian EMS requires problem_id=scalar_field");
  PARTHENON_REQUIRE(pin.GetString("numerical_relativity","matter_source")=="external",
                    "Cartesian EMS requires numerical_relativity/matter_source=external");
  const auto nr_package=context.dependencies.Get("numerical_relativity");
  PARTHENON_REQUIRE(nr_package->Param<std::string>("rk4_variant")=="classical",
                    "Cartesian EMS requires classical RK4");
  Parameters p;
  p.order=nr_package->Param<int>("finite_difference_order");
  const std::set<std::string> keys{"backend","coordinates","fields","ell","gb_eta","em_linearized",
      "potential","mass","quartic","sextic","decay_constant",
      "em_coupling","em_alpha","initial_data","amplitude","wave_x","wave_y","wave_z",
      "polarization_x","polarization_y","polarization_z","bh_mass","bh_charge","cfl","dissipation"};
  for(const auto& key:pin.GetParameterNames(block))
    PARTHENON_REQUIRE(keys.count(key)!=0,"unknown Cartesian EMS parameter: "+key);
  PARTHENON_REQUIRE(pin.GetOrAddInteger(block,"fields",1)==1 &&
      pin.GetOrAddInteger(block,"ell",0)==0 && pin.GetOrAddReal(block,"gb_eta",0)==0 &&
      !pin.GetOrAddBoolean(block,"em_linearized",false),
      "coupled EMS requires one neutral real field, ell=0, gb_eta=0 and nonlinear Maxwell coupling");
  const auto potential=pin.GetOrAddString(block,"potential","polynomial");
  PARTHENON_REQUIRE(potential=="polynomial" || potential=="axion","unknown EMS potential");
  p.model.use_axion=potential=="axion";
  const Real mass=pin.GetOrAddReal(block,"mass",0);
  p.model.polynomial={mass*mass,pin.GetOrAddReal(block,"quartic",0),pin.GetOrAddReal(block,"sextic",0)};
  p.model.axion={mass*mass,pin.GetOrAddReal(block,"decay_constant",1)};
  const auto coupling=pin.GetOrAddString(block,"em_coupling","exponential");
  PARTHENON_REQUIRE(coupling=="exponential" || coupling=="quadratic","unknown EMS coupling");
  p.model.coupling=coupling=="exponential" ? 0 : 1;
  p.model.alpha=pin.GetOrAddReal(block,"em_alpha",5);
  const auto initial=pin.GetOrAddString(block,"initial_data","reissner_nordstrom");
  PARTHENON_REQUIRE(initial=="minkowski" || initial=="zero" || initial=="scalar_wave" || initial=="maxwell_wave" ||
                    initial=="reissner_nordstrom","unknown Cartesian EMS initial data");
  p.initial=initial=="minkowski" ? 0 : (initial=="scalar_wave" ? 1 : (initial=="maxwell_wave" ? 2 : 3));
  p.amplitude=pin.GetOrAddReal(block,"amplitude",1.e-6);
  p.mass=pin.GetOrAddReal(block,"bh_mass",1); p.charge=pin.GetOrAddReal(block,"bh_charge",0.9);
  p.cfl=pin.GetOrAddReal(block,"cfl",0.25); p.dissipation=pin.GetOrAddReal(block,"dissipation",0);
  const char* axis[3]={"x","y","z"};
  for(int n=0;n<3;++n) {
    p.wave[n]=pin.GetOrAddReal(block,std::string("wave_")+axis[n],1);
    p.polarization[n]=pin.GetOrAddReal(block,std::string("polarization_")+axis[n],n==0 ? 1 : (n==1 ? -1 : 0));
  }
  PARTHENON_REQUIRE(mass>=0 && p.model.alpha>=0 && p.model.polynomial.quartic>=0 &&
      p.model.polynomial.sextic>=0 && p.model.axion.decay_constant>0 && p.cfl>0 && p.cfl<=0.5 &&
      p.dissipation>=0 && p.dissipation<=1,"invalid EMS model or stability parameters");
  PARTHENON_REQUIRE(!p.model.use_axion ||
      (p.model.polynomial.quartic==0 && p.model.polynomial.sextic==0),"axion potential excludes polynomial coefficients");
  PARTHENON_REQUIRE(p.initial!=1 || (!p.model.use_axion && p.model.polynomial.quartic==0 &&
      p.model.polynomial.sextic==0),"analytic scalar-wave momentum requires a linear scalar potential");
  PARTHENON_REQUIRE(p.initial!=3 || (p.mass>0 && fabs(p.charge)<p.mass),"RN data require M>abs(Q)");
  Real norm=0,polnorm=0,dot=0;
  for(int n=0;n<3;++n) { norm+=p.wave[n]*p.wave[n]; dot+=p.wave[n]*p.polarization[n]; }
  PARTHENON_REQUIRE(norm>0,"nonzero wave vector required");
  for(int n=0;n<3;++n) { const Real value=p.polarization[n]-p.wave[n]*dot/norm; polnorm+=value*value; }
  PARTHENON_REQUIRE(polnorm>0,"Maxwell polarization must not be parallel to wave vector");
  for(Real value:{mass,p.model.alpha,p.model.polynomial.quartic,p.model.polynomial.sextic,
      p.model.axion.decay_constant,p.amplitude,p.mass,p.charge,p.cfl,p.dissipation,
      p.wave[0],p.wave[1],p.wave[2],p.polarization[0],p.polarization[1],p.polarization[2]})
    PARTHENON_REQUIRE(std::isfinite(value),"EMS parameters must be finite");
  auto package=std::make_shared<StateDescriptor>(label);
  package->AddParam<std::string>("coordinates","cartesian");
  package->AddParam<Parameters>("parameters",p);
  package->AddParam<plugin::SynchronizedMatter>(plugin::synchronized_matter_key,
      {{"scalar.state"},Prepare,RHS,Update});
  package->AddField("scalar.state",Metadata({Metadata::Cell,Metadata::Independent,Metadata::Restart,
      Metadata::FillGhost},std::vector<int>{8},std::vector<std::string>{"phi","P","D_x","D_y","D_z","Bdens_x","Bdens_y","Bdens_z"}));
  for(const char* name:{"scalar.rhs","scalar.initial","scalar.rk_sum"})
    package->AddField(name,Metadata({Metadata::Cell,Metadata::Derived,Metadata::OneCopy},std::vector<int>{8}));
  package->AddField("ems.flux",Metadata({Metadata::Cell,Metadata::Derived,Metadata::OneCopy},std::vector<int>{18}));
  package->AddField("ems.gauss",Metadata({Metadata::Cell,Metadata::Derived,Metadata::OneCopy},std::vector<int>{2}));
  package->AddField("scalar.phi",Metadata({Metadata::Cell,Metadata::Derived,Metadata::OneCopy},std::vector<int>{1}));
  package->PostProblemGeneratorMesh=InitializeFields;
  package->FillDerivedMesh=[](MeshData<Real>* data) { Prepare(data,0); };
  package->EstimateTimestepMesh=Timestep;
  return package;
}
} // namespace pangu::plugins::scalar_field
