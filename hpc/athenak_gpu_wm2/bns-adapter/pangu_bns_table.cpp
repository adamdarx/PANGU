// Initial-data adapter only. Table interpolation follows PANGU compact_objects.
// Native AthenaK GRMHD/Z4c evolution and finite differences are unchanged.
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#define PARTHENON_REQUIRE(c,m) do { if (!(c)) throw std::runtime_error(m); } while (false)
namespace {
struct BNSData {
  Kokkos::View<Real **> coefficients;
  Kokkos::View<Real *> moments, source;
  Real dr, source_dr, kappa, separation, surface;
  int nr, nl, ns;

  KOKKOS_INLINE_FUNCTION Real Source(const Real distance) const {
    if (distance >= surface) return 0;
    const Real coordinate = distance / source_dr;
    const int left = static_cast<int>(fmin(fmax(floor(coordinate)-1, Real(0)), Real(ns-4)));
    const Real t = coordinate-left;
    const Real result = -(t-1)*(t-2)*(t-3)/6*source(left)
        +t*(t-2)*(t-3)/2*source(left+1)
        -t*(t-1)*(t-3)/2*source(left+2)
        +t*(t-1)*(t-2)/6*source(left+3);
    return fmax(result, Real(0));
  }

  KOKKOS_INLINE_FUNCTION Real Psi(const Real x, const Real y, const Real z) const {
    const Real radius = sqrt(x*x+y*y+z*z);
    const Real mu = radius > 0 ? z/radius : 0;
    const Real coordinate = radius/dr;
    const int left = static_cast<int>(fmin(fmax(floor(coordinate)-1, Real(0)), Real(nr-4)));
    const Real t = coordinate-left;
    const Real w[4] = {-(t-1)*(t-2)*(t-3)/6, t*(t-2)*(t-3)/2,
                       -t*(t-1)*(t-3)/2, t*(t-1)*(t-2)/6};
    Real sum = 1, previous = 1, current = mu;
    Real inverse_power = radius > 0 ? 1/radius : 0;
    for (int ell = 0; ell <= 2*(nl-1); ++ell) {
      const Real polynomial = ell == 0 ? previous : current;
      if (ell % 2 == 0) {
        const int j = ell/2;
        Real radial = 0;
        if (coordinate <= nr-1) {
          for (int a = 0; a < 4; ++a) radial += w[a]*coefficients(left+a, j);
        } else radial = moments(j)*inverse_power;
        sum += radial*polynomial;
      }
      if (ell >= 1) {
        const Real next = ((2*ell+1)*mu*current-ell*previous)/(ell+1);
        previous = current;
        current = next;
      }
      if (radius > 0) inverse_power /= radius;
    }
    return sum;
  }
};

BNSData ReadData(const std::string &path) {
  std::ifstream file(path);
  PARTHENON_REQUIRE(file.good(), "cannot open BNS initial-data table: " + path);
  std::string schema;
  std::getline(file, schema);
  PARTHENON_REQUIRE(schema == "PANGU_BNS_TIME_SYMMETRIC_V1", "invalid BNS table schema");
  BNSData d{};
  Real residual;
  file >> d.nr >> d.nl >> d.ns >> d.dr >> d.source_dr >> d.kappa
       >> d.separation >> d.surface >> residual;
  PARTHENON_REQUIRE(file.good() && d.nr >= 4 && d.nr <= 200000 &&
                        d.nl >= 1 && d.nl <= 65 && d.ns >= 4 && d.ns <= 200000,
                    "invalid BNS table dimensions");
  for (Real value : {d.dr, d.source_dr, d.kappa, d.separation, d.surface})
    PARTHENON_REQUIRE(std::isfinite(value) && value > 0, "invalid BNS table parameter");
  PARTHENON_REQUIRE(std::isfinite(residual) && residual >= 0 && residual < 1e-9 &&
                        d.separation > 2*d.surface &&
                        (d.ns-1)*d.source_dr >= d.surface &&
                        (d.nr-1)*d.dr > d.separation/2+d.surface,
                    "unconverged, overlapping or truncated BNS initial data");
  d.moments = Kokkos::View<Real *>("BNS exterior moments", d.nl);
  d.source = Kokkos::View<Real *>("BNS TOV conformal energy", d.ns);
  d.coefficients = Kokkos::View<Real **>("BNS psi multipoles", d.nr, d.nl);
  auto hm = Kokkos::create_mirror_view(d.moments);
  auto hs = Kokkos::create_mirror_view(d.source);
  auto hc = Kokkos::create_mirror_view(d.coefficients);
  auto read = [&]() {
    Real value;
    file >> value;
    PARTHENON_REQUIRE(file && std::isfinite(value), "nonfinite or missing BNS table value");
    return value;
  };
  for (int j = 0; j < d.nl; ++j) hm(j) = read();
  PARTHENON_REQUIRE(hm(0) > 0, "BNS ADM mass must be positive");
  for (int i = 0; i < d.ns; ++i) {
    hs(i) = read();
    PARTHENON_REQUIRE(hs(i) >= 0, "negative BNS conformal energy");
  }
  for (int i = 0; i < d.nr; ++i)
    for (int j = 0; j < d.nl; ++j) hc(i, j) = read();
  std::string extra;
  PARTHENON_REQUIRE(!(file >> extra), "unexpected trailing BNS table data");
  Kokkos::deep_copy(d.moments, hm);
  Kokkos::deep_copy(d.source, hs);
  Kokkos::deep_copy(d.coefficients, hc);
  return d;
}


} // namespace
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  auto *p = pmy_mesh_->pmb_pack;
  PARTHENON_REQUIRE(p->pz4c && p->pdyngr && !p->padm->is_dynamic && !pmy_mesh_->adaptive,
                    "BNS table adapter requires evolved Z4c and a static mesh");
  PARTHENON_REQUIRE(std::abs(pin->GetReal("mhd","gamma")-2)<1e-10,
                    "BNS table requires Gamma2");
  if (restart) return;
  const auto d=ReadData(pin->GetString("problem","initial_data_file"));
  DvceArray1D<Real> samples("BNS source GPU probe",3);
  par_for("BNS source GPU reference",DevExeSpace(),0,0,KOKKOS_LAMBDA(int q) {
    samples(0)=d.Source(Real(0));samples(1)=d.source(0);
    samples(2)=d.Psi(Real(0),Real(0),d.separation/2);
  });
  auto host_samples=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),samples);
  std::cout << "BNS_SOURCE_PROBE rank=" << global_variable::my_rank
            << " interpolated=" << host_samples(0) << " tabulated=" << host_samples(1)
            << " psi_center=" << host_samples(2) << std::endl;
  PARTHENON_REQUIRE(host_samples(0)>0 && std::abs(host_samples(0)/host_samples(1)-1)<1e-12,
                    "BNS GPU source interpolation does not match table center");
  const Real dfloor=pin->GetReal("mhd","dfloor");
  const Real pfloor=pin->GetReal("problem","pressure_floor");
  auto &ind=pmy_mesh_->mb_indcs;
  const int n1=ind.nx1+2*ind.ng,n2=ind.nx2+2*ind.ng,n3=ind.nx3+2*ind.ng;
  const int is=ind.is,js=ind.js,ks=ind.ks,nx=ind.nx1,ny=ind.nx2,nz=ind.nx3;
  auto &size=p->pmb->mb_size;auto &adm=p->padm->adm;auto &w=p->pmhd->w0;
  Kokkos::deep_copy(p->pmhd->b0.x1f,0.);Kokkos::deep_copy(p->pmhd->b0.x2f,0.);
  Kokkos::deep_copy(p->pmhd->b0.x3f,0.);Kokkos::deep_copy(p->pmhd->bcc0,0.);
  par_for("BNS table initial data",DevExeSpace(),0,p->nmb_thispack-1,0,n3-1,0,n2-1,0,n1-1,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    const Real x=CellCenterX(i-is,nx,size.d_view(m).x1min,size.d_view(m).x1max);
    const Real y=CellCenterX(j-js,ny,size.d_view(m).x2min,size.d_view(m).x2max);
    const Real z=CellCenterX(k-ks,nz,size.d_view(m).x3min,size.d_view(m).x3max);
    const Real psi=d.Psi(x,y,z);
    if (!(psi>=1) || !isfinite(psi)) Kokkos::abort("Invalid BNS psi");
    const Real ebar=d.Source(sqrt(x*x+y*y+pow(z-d.separation/2,2)))
                   +d.Source(sqrt(x*x+y*y+pow(z+d.separation/2,2)));
    const Real energy=ebar/pow(psi,8),rho=2*energy/(1+sqrt(1+4*d.kappa*energy));
    w(m,IDN,k,j,i)=fmax(rho,dfloor);w(m,IPR,k,j,i)=fmax(d.kappa*rho*rho,pfloor);
    w(m,IVX,k,j,i)=w(m,IVY,k,j,i)=w(m,IVZ,k,j,i)=0;
    const Real psi4=pow(psi,4);
    adm.psi4(m,k,j,i)=psi4;adm.alpha(m,k,j,i)=1/(psi*psi);
    for(int a=0;a<3;++a) {
      adm.beta_u(m,a,k,j,i)=0;
      for(int b=a;b<3;++b) {
        adm.g_dd(m,a,b,k,j,i)=a==b?psi4:0;
        adm.vK_dd(m,a,b,k,j,i)=0;
      }
    }
  });
  Kokkos::fence();
  const int cells=p->nmb_thispack*n1*n2*n3;
  int bad=0;Real maximum_rho=0;
  Kokkos::parallel_reduce("Check BNS initial ADM",Kokkos::RangePolicy<DevExeSpace>(0,cells),
    KOKKOS_LAMBDA(int q,int &count) {
      const int i=q%n1;q/=n1;const int j=q%n2;q/=n2;const int k=q%n3;const int m=q/n3;
      bool invalid=!(adm.psi4(m,k,j,i)>=1) || !(adm.alpha(m,k,j,i)>0);
      invalid=invalid || !isfinite(w(m,IDN,k,j,i)) || w(m,IDN,k,j,i)<dfloor;
      for(int a=0;a<3;++a)for(int b=a;b<3;++b)
        invalid=invalid || !isfinite(adm.g_dd(m,a,b,k,j,i)) || !isfinite(adm.vK_dd(m,a,b,k,j,i));
      if(invalid)++count;
    },bad);
  Kokkos::parallel_reduce("Check BNS initial rho",Kokkos::RangePolicy<DevExeSpace>(0,cells),
    KOKKOS_LAMBDA(int q,Real &value) {
      const int i=q%n1;q/=n1;const int j=q%n2;q/=n2;const int k=q%n3;const int m=q/n3;
      value=fmax(value,w(m,IDN,k,j,i));
    },Kokkos::Max<Real>(maximum_rho));
  std::cout << "BNS_INIT_FILLED rank=" << global_variable::my_rank
            << " bad=" << bad << " rho_max=" << maximum_rho << std::endl;
  PARTHENON_REQUIRE(bad==0,"BNS initial ADM/primitive validation failed");
  p->pdyngr->PrimToConInit(0,n1-1,0,n2-1,0,n3-1);
  Kokkos::fence();
  std::cout << "BNS_INIT_PRIM_TO_CON_COMPLETE rank=" << global_variable::my_rank << std::endl;

  switch(ind.ng) {
    case 2:p->pz4c->ADMToZ4c<2>(p,pin);p->pz4c->ADMConstraints<2>(p);break;
    case 3:p->pz4c->ADMToZ4c<3>(p,pin);p->pz4c->ADMConstraints<3>(p);break;
    case 4:p->pz4c->ADMToZ4c<4>(p,pin);p->pz4c->ADMConstraints<4>(p);break;
    default:throw std::runtime_error("Unsupported ghost count");
  }
}
