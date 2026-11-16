#pragma once

#include "z4c/evolution/z4c_rhs.h"

namespace pangu::plugins::scalar_field::spherical {
using parthenon::Real;

// Cartesian tensor components on the positive x axis. Rotations determine all
// transverse derivatives analytically, leaving only one radial grid to evolve.
// T_ij=t delta_ij+(s-t)n_i n_j and V^i=v n^i. No angular modes are present.
KOKKOS_INLINE_FUNCTION void ScalarJet(Real first, Real second, Real r,
                                    Real d[3], Real dd[6]) {
  for (int a=0;a<3;++a) d[a]=0;
  for (int a=0;a<6;++a) dd[a]=0;
  d[0]=first; dd[0]=second; dd[3]=dd[5]=first/r;
}

KOKKOS_INLINE_FUNCTION void VectorJet(Real value, Real first, Real second, Real r,
                                    Real d[3][3], Real dd[6][3]) {
  for (int a=0;a<3;++a) for (int b=0;b<3;++b) d[a][b]=0;
  for (int a=0;a<6;++a) for (int b=0;b<3;++b) dd[a][b]=0;
  d[0][0]=first; d[1][1]=d[2][2]=value/r;
  dd[0][0]=second;
  const Real mixed=(first-value/r)/r;
  dd[3][0]=dd[5][0]=dd[1][1]=dd[2][2]=mixed;
}

KOKKOS_INLINE_FUNCTION void TensorJet(Real s, Real ds, Real dds,
                                    Real t, Real dt, Real ddt, Real r,
                                    Real d[3][6], Real dd[6][6]) {
  for (int a=0;a<3;++a) for (int b=0;b<6;++b) d[a][b]=0;
  for (int a=0;a<6;++a) for (int b=0;b<6;++b) dd[a][b]=0;
  const Real q=(s-t)/r, w=q/r, mixed=(ds-dt-q)/r;
  d[0][0]=ds; d[0][3]=d[0][5]=dt;
  d[1][1]=d[2][2]=q;
  dd[0][0]=dds; dd[0][3]=dd[0][5]=ddt;
  dd[3][0]=dd[5][0]=ds/r-2*w;
  dd[3][3]=dd[5][5]=dt/r+2*w;
  dd[3][5]=dd[5][3]=dt/r;
  dd[1][1]=dd[2][2]=mixed;
  dd[4][4]=w;
}

template<class Pack> struct RadialValue {
  Pack pack;
  int component,k,j,lo,hi,parity=1;
  KOKKOS_INLINE_FUNCTION Real operator()(int n) const {
    if(n<lo) return parity*pack(0,component,k,j,2*lo-1-n);
    if(n<=hi) return pack(0,component,k,j,n);
    // Degree-five continuation supports fourth-order second derivatives at
    // the outermost cells. Their time derivatives use Sommerfeld data.
    Real result=0;
    for(int a=0;a<6;++a) {
      Real weight=1;
      for(int b=0;b<6;++b) if(a!=b) weight*=Real(n-(hi-b))/Real(b-a);
      result+=weight*pack(0,component,k,j,hi-a);
    }
    return result;
  }
};

template<class Value> KOKKOS_INLINE_FUNCTION
void RadialJet(const Value& q,int i,Real hx,Real jac,Real jac2,Real& d,Real& dd) {
  const Real dx=(q(i-2)-8*q(i-1)+8*q(i+1)-q(i+2))/(12*hx);
  const Real dxx=(-q(i+2)+16*q(i+1)-30*q(i)+16*q(i-1)-q(i-2))/(12*hx*hx);
  d=dx/jac; dd=(dxx-jac2*dx/jac)/(jac*jac);
}

template<class Value> struct ShiftedValue {
  Value q; int i;
  KOKKOS_INLINE_FUNCTION Real operator()(int,int,int di) const {return q(i+di);}
};

template<class Pack> KOKKOS_INLINE_FUNCTION
void LoadPoint(const Pack& z,int k,int j,int i,int lo,int hi,Real hx,
               Real r,Real jac,Real jac2,nr::rhs::PointState& s,
               nr::rhs::PointDerivatives& d) {
  Real first[22]{}, second[22]{}, adv[22]{};
  const Real beta=z(0,19,k,j,i);
  for(int n=0;n<22;++n) {
    const int parity=(n==14 || n==19) ? -1:1;
    const RadialValue<Pack> q{z,n,k,j,lo,hi,parity};
    RadialJet(q,i,hx,jac,jac2,first[n],second[n]);
    adv[n]=nr::fd::AdvectiveFirst<4>(0,1/hx,beta/jac,ShiftedValue<decltype(q)>{q,i});
  }
  s.chi=z(0,0,k,j,i); s.khat=z(0,7,k,j,i); s.theta=z(0,17,k,j,i);
  s.alpha=z(0,18,k,j,i); s.beta[0]=beta; s.gamma[0]=z(0,14,k,j,i);
  for(int n=0;n<6;++n) {s.metric[n]=z(0,1+n,k,j,i);s.a[n]=z(0,8+n,k,j,i);}
  ScalarJet(first[0],second[0],r,d.dchi,d.ddchi);
  ScalarJet(first[18],second[18],r,d.dalpha,d.ddalpha);
  d.dkhat[0]=first[7]; d.dtheta[0]=first[17];
  VectorJet(beta,first[19],second[19],r,d.dbeta,d.ddbeta);
  Real unused[6][3]{};
  VectorJet(s.gamma[0],first[14],second[14],r,d.dgamma,unused);
  TensorJet(s.metric[0],first[1],second[1],s.metric[3],first[4],second[4],r,
            d.dmetric,d.ddmetric);
  d.lie_chi=adv[0];d.lie_khat=adv[7];d.lie_theta=adv[17];d.lie_alpha=adv[18];
  d.lie_gamma[0]=adv[14];d.lie_beta[0]=adv[19];
  for(int n=0;n<6;++n) {d.lie_metric[n]=adv[1+n];d.lie_a[n]=adv[8+n];}
}
} // namespace pangu::plugins::scalar_field::spherical
