// Initial data only. Native AsterX and Z4c evolution are unchanged.
// Host initialization follows TOVSolverX; all evolution remains on the GPU.
#include <loop.hxx>
#include <loop_device.hxx>
#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace TableBNSData {
using namespace Loop;
struct Table {
  int nr=0,nl=0,ns=0;
  double dr=0,ds=0,kappa=0,separation=0,surface=0;
  std::vector<double> moments,source,coefficients;
  double Source(double r) const {
    if(r>=surface) return 0;
    double q=r/ds;int l=std::min(std::max(int(std::floor(q))-1,0),ns-4);
    double t=q-l;
    return std::max(0.,-(t-1)*(t-2)*(t-3)/6*source[l]
      +t*(t-2)*(t-3)/2*source[l+1]-t*(t-1)*(t-3)/2*source[l+2]
      +t*(t-1)*(t-2)/6*source[l+3]);
  }
  double Psi(double x,double y,double z) const {
    double r=std::sqrt(x*x+y*y+z*z),mu=r>0?z/r:0,q=r/dr;
    int l=std::min(std::max(int(std::floor(q))-1,0),nr-4);double t=q-l;
    double w[4]={-(t-1)*(t-2)*(t-3)/6,t*(t-2)*(t-3)/2,
                 -t*(t-1)*(t-3)/2,t*(t-1)*(t-2)/6};
    double sum=1,prev=1,cur=mu,inv=r>0?1/r:0;
    for(int ell=0;ell<=2*(nl-1);++ell) {
      double poly=ell==0?prev:cur;
      if(ell%2==0) {
        int j=ell/2;double radial=0;
        if(q<=nr-1) for(int a=0;a<4;++a) radial+=w[a]*coefficients[(l+a)*nl+j];
        else radial=moments[j]*inv;
        sum+=radial*poly;
      }
      if(ell>=1){double next=((2*ell+1)*mu*cur-ell*prev)/(ell+1);prev=cur;cur=next;}
      if(r>0)inv/=r;
    }
    return sum;
  }
};
static Table table;
extern "C" void TableBNS_Read(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TableBNS_Read;
  DECLARE_CCTK_PARAMETERS;
  std::ifstream file(table_file);std::string schema;std::getline(file,schema);
  if(!file || schema!="PANGU_BNS_TIME_SYMMETRIC_V1") CCTK_ERROR("Invalid BNS table schema");
  Table d;double residual;
  file>>d.nr>>d.nl>>d.ns>>d.dr>>d.ds>>d.kappa>>d.separation>>d.surface>>residual;
  if(!file || d.nr<4 || d.nr>200000 || d.ns<4 || d.ns>200000 || d.nl<1 || d.nl>65)
    CCTK_ERROR("Invalid BNS table dimensions");
  for(double v:{d.dr,d.ds,d.kappa,d.separation,d.surface})
    if(!std::isfinite(v) || v<=0) CCTK_ERROR("Invalid BNS table scale");
  if(!std::isfinite(residual) || residual<0 || residual>=1e-9 ||
     d.separation<=2*d.surface || (d.ns-1)*d.ds<d.surface ||
     (d.nr-1)*d.dr<=d.separation/2+d.surface || std::abs(d.kappa-100)>1e-10)
    CCTK_ERROR("Unconverged, overlapping, truncated, or incompatible BNS initial data");
  auto read=[&](){double v;file>>v;if(!file || !std::isfinite(v)) CCTK_ERROR("Invalid BNS table value");return v;};
  d.moments.resize(d.nl);d.source.resize(d.ns);d.coefficients.resize(size_t(d.nr)*d.nl);
  for(auto &v:d.moments)v=read();
  for(auto &v:d.source){v=read();if(v<0)CCTK_ERROR("Negative BNS source");}
  for(auto &v:d.coefficients)v=read();
  std::string extra;if(file>>extra)CCTK_ERROR("Unexpected trailing BNS data");
  if(d.moments[0]<=0 || d.source[0]<=0 || std::abs(d.Source(0)/d.source[0]-1)>1e-12)
    CCTK_ERROR("Invalid BNS central source or ADM moment");
  table=std::move(d);
  CCTK_VINFO("BNS_TABLE source0=%.17g psi_center=%.17g ADM_mass=%.17g separation=%.17g residual=%.17g",
    table.Source(0),table.Psi(0,0,table.separation/2),2*table.moments[0],table.separation,residual);
}
extern "C" void TableBNS_ADM(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TableBNS_ADM;
  DECLARE_CCTK_PARAMETERS;
  const Table *d=&table;
  grid.loop_all<0,0,0>(grid.nghostzones,[=] CCTK_HOST(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
    double psi=d->Psi(p.x,p.y,p.z);
    if(!std::isfinite(psi) || psi<1)CCTK_ERROR("Invalid BNS conformal factor");
    double psi4=std::pow(psi,4);
    gxx(p.I)=gyy(p.I)=gzz(p.I)=psi4;
    gxy(p.I)=gxz(p.I)=gyz(p.I)=0;
    kxx(p.I)=kxy(p.I)=kxz(p.I)=kyy(p.I)=kyz(p.I)=kzz(p.I)=0;
    alp(p.I)=1/(psi*psi);
  });
}
extern "C" void TableBNS_Hydro(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TableBNS_Hydro;
  DECLARE_CCTK_PARAMETERS;
  const Table *d=&table;
  grid.loop_all<1,1,1>(grid.nghostzones,[=] CCTK_HOST(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
    double psi=d->Psi(p.x,p.y,p.z),z1=p.z-d->separation/2,z2=p.z+d->separation/2;
    double source=d->Source(std::sqrt(p.x*p.x+p.y*p.y+z1*z1))+
                  d->Source(std::sqrt(p.x*p.x+p.y*p.y+z2*z2));
    double energy=source/std::pow(psi,8);
    double rest=2*energy/(1+std::sqrt(1+4*d->kappa*energy));
    rho(p.I)=std::max(rest,double(rho_floor));
    press(p.I)=std::max(d->kappa*rest*rest,double(pressure_floor));
    eps(p.I)=press(p.I)/rho(p.I); // Gamma=2
    velx(p.I)=vely(p.I)=velz(p.I)=0;
  });
}
extern "C" void TableBNS_Magnetic(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TableBNS_Magnetic;
  DECLARE_CCTK_PARAMETERS;
  grid.loop_all_device<1,0,0>(grid.nghostzones,[=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {Avec_x(p.I)=0;});
  grid.loop_all_device<0,1,0>(grid.nghostzones,[=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {Avec_y(p.I)=0;});
  grid.loop_all_device<0,0,1>(grid.nghostzones,[=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {Avec_z(p.I)=0;});
}
} // namespace TableBNSData
