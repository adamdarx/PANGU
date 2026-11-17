"""Static spherical TOV quadrature: normalize native dipole to equal magnetic energy.

No time evolution. For A=(-yF,xF,0), angular mean(curl A)^2 is
4F²+(8/3)rFF'+(2/3)r²F'². E_B=.5 integral(curl A)^2/psi² d³x.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp,quad
from scipy.optimize import brentq

def calculate(tol):
 k=100.;rhoc=.007993;pc=k*rhoc**2;ec=rhoc+pc;hc=np.log1p(2*k*rhoc);start=1e-6
 def rhs(r,y):
  h,m,L=y;rho=np.expm1(max(h,0))/(2*k);p=k*rho*rho;e=rho+p;f=1-2*m/r;q=np.sqrt(f)
  return [-(m+4*np.pi*r**3*p)/(r*r*f),4*np.pi*r*r*e,2*m/(r*r*q*(1+q))]
 def surface(r,y):return y[0]
 surface.terminal=True;surface.direction=-1
 sol=solve_ivp(rhs,(start,20),[hc-2*np.pi*(ec+3*pc)*start**2/3,4*np.pi*ec*start**3/3,0],events=surface,dense_output=True,method='DOP853',rtol=tol,atol=tol*.01)
 assert sol.success and len(sol.t_events[0])==1
 R=sol.t_events[0][0];_,M,Ls=sol.sol(R);Ri=(R-M+np.sqrt(R*(R-2*M)))/2;offset=np.log(Ri/R)-Ls;cut=1e-6
 def physical(r):
  h,m,L=sol.sol(r);rho=np.expm1(max(h,0))/(2*k);p=k*rho*rho;e=rho+p;q=np.sqrt(1-2*m/r);ri=r*np.exp(L+offset);dp=-(e+p)*(m+4*np.pi*r**3*p)/(r*r*q*q);return rho,p,dp,q,ri
 def energy(native):
  depletion=.99 if native else 1.
  edge=brentq(lambda r:depletion*physical(r)[1]-cut,start,R)
  def integrand(r):
   rho,p,dp,q,ri=physical(r);dr_dri=r*q/ri;F=depletion*p-cut;dF=depletion*dp*dr_dri
   if not native:
    taper=1-rho/rhoc;dF=dF*taper-F*dp/(2*k*rho*rhoc)*dr_dri;F*=taper
   mean=4*F*F+8/3*ri*F*dF+2/3*ri*ri*dF*dF
   return 2*np.pi*ri**4/(r*r*q)*mean
  return quad(integrand,start,edge,epsabs=tol*.01,epsrel=tol*10,limit=200)[0]
 original,native=energy(False),energy(True)
 return dict(ADM_mass=float(M),areal_radius=float(R),PANGU_continuum_seed_energy=original,ET_native_Ab1_energy=native,ET_Ab_equal_energy=float(np.sqrt(original/native)))
if __name__=='__main__':
 coarse=calculate(1e-10);fine=calculate(1e-12)
 result=dict(status='static_continuum_seed_energy_calibration',fine=fine,tolerance_refinement=coarse,relative_Ab_refinement_difference=fine['ET_Ab_equal_energy']/coarse['ET_Ab_equal_energy']-1,scope='Same total initial magnetic energy using different native seed profiles; static quadrature consistency is not a discretized GPU validation.')
 Path(__file__).with_name('SEED_ENERGY_CALIBRATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
