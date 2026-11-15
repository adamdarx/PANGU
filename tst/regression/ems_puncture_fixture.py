"""Independent spherical, time-symmetric scalar/charged-puncture test fixtures.

This solves the isotropic ADM Hamiltonian initial constraint. It is neither a
PG evolution nor a general nonspherical elliptic initial-data solver.
"""
from dataclasses import dataclass
import math
from pathlib import Path

import h5py
import numpy as np


@dataclass
class Profile:
    mass: float=1.
    charge: float=.9
    alpha: float=5.
    amplitude: float=.05
    center: float=.55
    width: float=.25
    points: int=1024

    def __post_init__(self):
        assert self.mass>abs(self.charge) and self.width>0 and self.center>self.width
        self.inner=(self.mass*self.mass-self.charge*self.charge)**.5/2
        self.left=self.center-self.width;self.right=self.center+self.width
        low=abs(self.charge)/(2*self.inner)+1.e-9
        high=max(self.mass/(2*self.inner)*1.1,low+.1)
        def residual(b):
            f,fp,_=self.integrate(b)
            return fp-math.sqrt(1+self.charge*self.charge/(4*f*f))
        assert residual(low)<0,'Scalar pulse has left the weak, positive puncture branch'
        for _ in range(20):
            if residual(high)>0:break
            high*=1.2
        else:raise RuntimeError('Could not bracket the asymptotically flat initial condition')
        for _ in range(48):
            middle=(low+high)/2
            if residual(middle)>0:high=middle
            else:low=middle
        self.b=(low+high)/2
        f,fp,table=self.integrate(self.b,True)
        self.values=np.asarray(table)
        self.adm_mass=2*f*fp-2*self.right
        self.outer_coefficient=(self.adm_mass*self.adm_mass-self.charge*self.charge)/4
        self.inner_coefficient=self.b*self.b-self.charge*self.charge/(4*self.inner*self.inner)
        self.outer_residual=fp-math.sqrt(1+self.charge*self.charge/(4*f*f))
        assert self.inner_coefficient>0 and self.adm_mass>abs(self.charge)

    def phi_and_gradient(self,r):
        s=(r-self.center)/self.width
        if abs(s)>=1:return 0.,0.
        base=1-s*s
        return self.amplitude*base**8,-16*self.amplitude*s/self.width*base**7

    def acceleration(self,r,f):
        phi,gradient=self.phi_and_gradient(r)
        # F=r*psi. For V=P=Kij=B=0 the Hamiltonian constraint is
        # F''=-F*(phi')^2/4-Q^2/(4*F^3*f(phi)), f(phi)=exp(alpha*phi^2).
        return -.25*f*gradient*gradient-self.charge*self.charge/(4*f**3*math.exp(self.alpha*phi*phi))

    def step(self,r,y,h):
        f,fp=y
        a=(fp,self.acceleration(r,f))
        b=(fp+h*a[1]/2,self.acceleration(r+h/2,f+h*a[0]/2))
        c=(fp+h*b[1]/2,self.acceleration(r+h/2,f+h*b[0]/2))
        d=(fp+h*c[1],self.acceleration(r+h,f+h*c[0]))
        return (f+h*(a[0]+2*b[0]+2*c[0]+d[0])/6,
                fp+h*(a[1]+2*b[1]+2*c[1]+d[1])/6)

    def integrate(self,b,keep=False):
        coefficient=b*b-self.charge*self.charge/(4*self.inner*self.inner)
        f=math.sqrt(coefficient*self.left**2+2*self.inner*b*self.left+self.inner*self.inner)
        y=(f,(coefficient*self.left+self.inner*b)/f)
        h=(self.right-self.left)/self.points;table=[y[0]] if keep else None
        for n in range(self.points):
            r=self.left+n*h
            full=self.step(r,y,h);half=self.step(r,y,h/2);fine=self.step(r+h/2,half,h/2)
            y=tuple(fine[k]+(fine[k]-full[k])/15 for k in [0,1])
            assert y[0]>0 and all(math.isfinite(value) for value in y)
            if keep:table.append(y[0])
        return *y,table

    def evaluate(self,r):
        r=np.asarray(r)
        f=np.empty_like(r);inside=r<=self.left;outside=r>=self.right
        f[inside]=np.sqrt(self.inner_coefficient*r[inside]**2+2*self.inner*self.b*r[inside]+self.inner**2)
        f[outside]=np.sqrt(r[outside]**2+self.adm_mass*r[outside]+self.outer_coefficient)
        middle=~(inside|outside);q=(r[middle]-self.left)/(self.right-self.left)*self.points
        base=np.clip(np.floor(q).astype(int)-2,0,self.points-5);t=q-base
        value=np.zeros_like(q)
        for j in range(6):
            weight=np.ones_like(q)
            for k in range(6):
                if k!=j:weight*=(t-k)/(j-k)
            value+=weight*self.values[base+j]
        f[middle]=value
        s=(r-self.center)/self.width;phi=self.amplitude*np.maximum(0.,1-s*s)**8
        return f,phi


def inject(seed,output,profile,power=None,solve_constraint=True):
    """Write a labeled synthetic test restart; retain the native input model."""
    import shutil
    shutil.copyfile(seed,output)
    with h5py.File(output,'r+') as target:
        assert float(target['Info'].attrs['Time'])==0
        raw=target['Input'].attrs['File']
        if isinstance(raw,bytes):raw=raw.decode()
        parameters={};section=None
        for line in raw.splitlines():
            line=line.split('#',1)[0].strip()
            if line.startswith('<'):section=line.strip('<>');parameters.setdefault(section,{})
            elif '=' in line:
                key,value=line.split('=',1);parameters[section][key.strip()]=value.strip()
        model=parameters['plugin/scalar_field']
        assert model['em_coupling']=='exponential'
        assert float(model['bh_mass'])==profile.mass
        assert float(model['bh_charge'])==profile.charge and float(model['em_alpha'])==profile.alpha
        assert all(float(model.get(key,0))==0 for key in ['mass','quartic','sextic'])
        recorded_power=float(parameters['numerical_relativity'].get('chi_psi_power',-4))
        if power is not None:
            assert recorded_power==power,'Fixture power differs from the native input'
        power=recorded_power
        for block in range(target['scalar.state'].shape[0]):
            x,y,z=[target['VolumeLocations/'+axis][block] for axis in 'xyz']
            xyz=np.broadcast_arrays(x[None,None,:],y[None,:,None],z[:,None,None])
            radius=np.sqrt(sum(a*a for a in xyz));assert radius.min()>0
            f,phi=profile.evaluate(radius)
            if not solve_constraint:f=np.sqrt(radius**2+profile.mass*radius+profile.inner**2)
            state=target['scalar.state'][block];state[:]=0;state[0]=phi
            for axis in range(3):state[2+axis]=profile.charge*xyz[axis]/radius**3
            geometry=target['nr.z4c'][block];geometry[:]=0
            geometry[0]=(f/radius)**power;geometry[18]=(radius/f)**2
            for component in [1,4,6]:geometry[component]=1
            target['scalar.state'][block]=state;target['nr.z4c'][block]=geometry


def self_check():
    empty=Profile(amplitude=0)
    radii=np.linspace(.001,2.,2001);f,_=empty.evaluate(radii)
    rn=np.sqrt(radii*radii+empty.mass*radii+empty.inner**2)
    assert np.max(np.abs(f-rn))<2.e-12
    coarse=Profile();fine=Profile(points=2048)
    fc,_=coarse.evaluate(radii);ff,_=fine.evaluate(radii)
    error=float(np.max(np.abs(fc-ff)))
    assert error<2.e-12
    assert abs(coarse.outer_residual)<1.e-12
    return dict(empty_RN_maximum_F_error=float(np.max(np.abs(f-rn))),
                radial_refinement_maximum_F_difference=error,adm_mass=coarse.adm_mass,
                seed_mass=coarse.mass,outer_boundary_residual=coarse.outer_residual)


if __name__=='__main__':
    import json
    print(json.dumps(self_check(),indent=2))
