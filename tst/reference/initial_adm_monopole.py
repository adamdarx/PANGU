"""ADM monopole check for a conformally flat, time-symmetric initial exterior.

Fits 2*r*(psi-1) with even axial multipoles in predeclared radial windows.
All fits are reported; the spread is a diagnostic, not a rigorous error bound.
This method is inappropriate for general evolved geometry or nonvacuum shells.
"""
from pathlib import Path
import argparse
import hashlib
import json

import h5py
import numpy as np
from numpy.polynomial.legendre import legval


def fit_monopole(radius,costheta,psi,weights):
    data=np.column_stack([radius,costheta,2*radius*(psi-1),weights]);fits=[]
    if not np.isfinite(data).all() or np.any(data[:,3]<=0):
        raise ValueError('Need finite samples and positive coordinate-volume weights')
    for lo,hi in [(30,45),(45,60),(60,90)]:
        r,c,y,w=data[(data[:,0]>lo)&(data[:,0]<hi)].T
        if len(r)<100:raise ValueError('Insufficient samples in a declared radial window')
        for lmax in [4,6,8]:
            for constant_offset in [False,True]:
                columns=[np.ones(len(r))]
                for ell in range(2,lmax+1,2):
                    coeff=np.zeros(ell+1);coeff[-1]=1
                    columns.append((30/r)**ell*legval(c,coeff))
                if constant_offset:columns.append(r/30)
                A=np.column_stack(columns);sw=np.sqrt(w/w.mean())
                coeff,_,rank,_=np.linalg.lstsq(A*sw[:,None],y*sw,rcond=None)
                if rank!=len(columns):raise ValueError('Rank-deficient multipole fit')
                fits.append(dict(radial_interval=[lo,hi],lmax=lmax,constant_offset_fitted=constant_offset,
                    ADM_mass=float(coeff[0]),weighted_fit_rms=float(np.sqrt(np.average((A@coeff-y)**2,weights=w))),points=len(r)))
    return fits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-sha256');p.add_argument('--expected-mass',type=float)
    a=p.parse_args();before=a.input.stat()
    with a.input.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if a.expected_sha256 and digest!=a.expected_sha256:raise ValueError('Input SHA does not match archive proof')
    samples=[];geometry=0.;curvature=0.;density=0.;baryon=0.
    with h5py.File(a.input) as f:
        if float(f['Info'].attrs['Time'])!=0 or int(f['Info'].attrs['IncludesGhost'])!=0:
            raise ValueError('Require a time-zero, interior-cell-only field')
        for b in range(len(f['Levels'])):
            dx,dy,dz=[np.diff(f['Locations/'+axis][b]) for axis in 'xyz']
            dv=dx[None,None,:]*dy[None,:,None]*dz[:,None,None]
            baryon+=float(np.sum(f['mhd.cons'][b,0]*dv))
            x,y,z=[f['VolumeLocations/'+axis][b,::4] for axis in 'xyz']
            r=np.sqrt(x[None,None,:]**2+y[None,:,None]**2+z[:,None,None]**2)
            mask=(r>30)&(r<90)
            if not mask.any():continue
            raw=f['nr.z4c'][b,:,::4,::4,::4]
            geometry=max(geometry,float(np.max(np.abs(raw[1:7,mask]-np.array([1,0,0,1,0,1])[:,None]))))
            curvature=max(curvature,float(np.max(np.abs(raw[7:14,mask]))))
            density=max(density,float(np.max(f['mhd.prim'][b,0,::4,::4,::4][mask])))
            chi=raw[0,mask]
            if not np.isfinite(chi).all() or np.any(chi<=0):raise ValueError('Invalid conformal factor')
            cos=np.broadcast_to(z[:,None,None],r.shape)[mask]/r[mask]
            samples.append(np.column_stack([r[mask],cos,chi**(-.25),dv[::4,::4,::4][mask]]))
    if geometry>1e-12 or curvature>1e-12 or density>1e-12:
        raise ValueError('Exterior is not conformally flat, time symmetric and dilute as required')
    if not samples:raise ValueError('No exterior samples')
    fits=fit_monopole(*np.concatenate(samples).T)
    after=a.input.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input changed during analysis')
    report=dict(status='initial_ADM_monopole_measured',source=str(a.input.resolve()),source_sha256=digest,
        bytes=before.st_size,time=0,baryon_mass_coordinate_integral=baryon,fits=fits,
        conformal_metric_max_deviation=geometry,extrinsic_components_max_abs=curvature,exterior_density_max=density,
        mass_fit_range=[min(v['ADM_mass'] for v in fits),max(v['ADM_mass'] for v in fits)],
        limitations=__doc__+' Requires a centered reflection-symmetric axial initial model; finite-grid nonaxial and unmodeled asymptotic errors are not certified by fit spread.')
    if a.expected_mass:
        report['expected_mass']=a.expected_mass
        report['maximum_relative_fit_difference']=max(abs(v['ADM_mass']/a.expected_mass-1) for v in fits)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['fits','limitations']}))


if __name__=='__main__':main()
