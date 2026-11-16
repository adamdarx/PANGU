"""Independent adaptive enthalpy-form TOV reference; no PANGU solver imports."""
from pathlib import Path
import argparse
import json
import math

import numpy as np
from scipy.integrate import solve_ivp


def solve(rhoc,kappa=100.,gamma=2.,rtol=1e-11):
    if rhoc<=0 or kappa<=0 or gamma<=1:raise ValueError('Invalid polytrope')
    hc=math.log1p(gamma*kappa*rhoc**(gamma-1)/(gamma-1))
    pc=kappa*rhoc**gamma;ec=rhoc+pc/(gamma-1)
    r0=1e-6
    initial=[hc-2*math.pi*(ec/3+pc)*r0*r0,4*math.pi*ec*r0**3/3,4*math.pi*rhoc*r0**3/3]
    def rhs(r,y):
        h,m,mb=y
        rho=(max(math.expm1(max(h,0)),0)*(gamma-1)/(gamma*kappa))**(1/(gamma-1))
        pressure=kappa*rho**gamma;energy=rho+pressure/(gamma-1)
        compactness=1-2*m/r
        if compactness<=0:raise ValueError('Static integration entered r<=2m')
        return [-(m+4*math.pi*r**3*pressure)/(r*r*compactness),4*math.pi*r*r*energy,
                4*math.pi*r*r*rho/math.sqrt(compactness)]
    def surface(r,y):return y[0]
    surface.terminal=True;surface.direction=-1
    result=solve_ivp(rhs,[r0,100],initial,method='DOP853',rtol=rtol,atol=rtol*1e-3,
                     events=surface,max_step=.05)
    if not result.success or len(result.t_events[0])!=1:raise ValueError('No resolved TOV surface')
    R=float(result.t_events[0][0]);_,M,Mb=map(float,result.y_events[0][0])
    Riso=.5*(R-M+math.sqrt(R*(R-2*M)))
    return dict(central_density=rhoc,kappa=kappa,gamma=gamma,gravitational_mass=M,
                baryon_mass=Mb,areal_radius=R,isotropic_surface_radius=Riso,
                surface_lapse=math.sqrt(1-2*M/R),relative_tolerance=rtol,
                scope='Unperturbed, unmagnetized spherical progenitor only. Velocity/magnetic seed changes and dynamical mass budgets need separate accounting.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rhoc',type=float,required=True)
    p.add_argument('--kappa',type=float,default=100);p.add_argument('--gamma',type=float,default=2)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    loose=solve(a.rhoc,a.kappa,a.gamma,1e-10);fine=solve(a.rhoc,a.kappa,a.gamma,1e-12)
    fine['tolerance_refinement_relative_differences']={k:abs(fine[k]/loose[k]-1) for k in ['gravitational_mass','baryon_mass','areal_radius']}
    if max(fine['tolerance_refinement_relative_differences'].values())>1e-7:
        raise ValueError('Independent TOV tolerance refinement failed')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(fine,indent=2)+'\n')
    print(json.dumps(fine,indent=2))


if __name__=='__main__':main()
