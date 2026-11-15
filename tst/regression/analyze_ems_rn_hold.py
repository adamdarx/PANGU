"""Independent finite-field, charge, history and AH checks for a local RN hold.

Passing these short electrovac checks does not establish long-time scalar hair.
Constraint convergence and finite-boundary errors require separate controls.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

import h5py
import numpy as np

from scalar_ems_rn import norms, check_history


def deck(path):
    result={};section=None
    for line in path.read_text().splitlines():
        line=line.split('#',1)[0].strip()
        if line.startswith('<'):
            section=line.strip('<>');result.setdefault(section,{})
        elif '=' in line:
            key,value=line.split('=',1);result[section][key.strip()]=value.strip()
    return result


def sphere_charge(density,minimum,spacing,radius,polar=24,azimuthal=48):
    mu,weights=np.polynomial.legendre.leggauss(polar)
    angles=np.arange(azimuthal)*2*np.pi/azimuthal
    sine=np.sqrt(1-mu[:,None]**2)
    normal=np.array([(sine*np.cos(angles)[None,:]).ravel(),
                     (sine*np.sin(angles)[None,:]).ravel(),
                     np.broadcast_to(mu[:,None],(polar,azimuthal)).ravel()])
    logical=(radius*normal-np.array(minimum)[:,None])/np.array(spacing)[:,None]-.5
    indices=[];coefficients=[]
    for axis in range(3):
        nodes=np.floor(logical[axis]).astype(int)[:,None]-2+np.arange(6)[None,:]
        assert nodes.min()>=0 and nodes.max()<density.shape[3-axis],'Charge sphere stencil leaves the domain'
        weight=np.ones_like(nodes,dtype=float)
        for j in range(6):
            for other in range(6):
                if j!=other:weight[:,j]*=(logical[axis]-nodes[:,other])/(nodes[:,j]-nodes[:,other])
        indices.append(nodes);coefficients.append(weight)
    interpolated=np.zeros_like(normal)
    for k in range(6):
        for j in range(6):
            for i in range(6):
                interpolated+=density[:,indices[2][:,k],indices[1][:,j],indices[0][:,i]]*(
                    coefficients[0][:,i]*coefficients[1][:,j]*coefficients[2][:,k])[None,:]
    weights=np.repeat(weights,azimuthal)*2*np.pi/azimuthal
    return float(radius**2*np.sum(np.sum(interpolated*normal,axis=0)*weights)/(4*np.pi))


def inspect(path,inner,outer,radius):
    hamiltonian=[];gauss=[[],[]];active=0;min_variable=np.inf;min_det=np.inf
    with h5py.File(path) as output:
        size=list(map(int,output['Info'].attrs['RootGridSize']))
        domain=output['Info'].attrs['RootGridDomain'];minimum=domain[[0,3,6]]
        spacing=(domain[[1,4,7]]-minimum)/size
        electric=np.empty((3,size[2],size[1],size[0]))
        for block in range(output['scalar.state'].shape[0]):
            state=output['scalar.state'][block];adm=output['nr.adm'][block]
            constraints=output['nr.constraints'][block];g=output['ems.gauss'][block]
            z4c=output['nr.z4c'][block];stress=output['nr.tmunu'][block]
            assert all(np.isfinite(value).all() for value in [state,adm,constraints,g,z4c,stress])
            assert np.max(np.abs(state[:2]))==0,'RN developed a nonzero scalar'
            det=adm[0]*(adm[3]*adm[5]-adm[4]**2)-adm[1]*(adm[1]*adm[5]-adm[2]*adm[4])+adm[2]*(adm[1]*adm[4]-adm[2]*adm[3])
            assert np.min(adm[0])>0 and np.min(adm[0]*adm[3]-adm[1]**2)>0 and det.min()>0,'Spatial metric lost positive definiteness'
            min_variable=min(min_variable,float(z4c[0].min()));min_det=min(min_det,float(det.min()))
            assert min_variable>0
            xyz=[output['VolumeLocations/'+axis][block] for axis in 'xyz']
            x,y,z=xyz;r2=x[None,None,:]**2+y[None,:,None]**2+z[:,None,None]**2
            mask=(r2>inner*inner)&(r2<=outer*outer)
            np.testing.assert_array_equal(output['nr.constraint_mask'][block,0],mask)
            hamiltonian.extend(constraints[1][mask]);active+=int(mask.sum())
            for component in range(2):gauss[component].extend(g[component][mask])
            i,j,k=[int(round((coordinates[0]-minimum[axis])/spacing[axis]-.5)) for axis,coordinates in enumerate(xyz)]
            electric[:,k:k+len(z),j:j+len(y),i:i+len(x)]=state[2:5]
        result=dict(file=path.name,time=float(output['Info'].attrs['Time']),active_cells=active,
                    minimum_conformal_variable=min_variable,minimum_spatial_determinant=min_det,
                    hamiltonian=norms(hamiltonian),electric_gauss=norms(gauss[0]),magnetic_gauss=norms(gauss[1]))
    charge=sphere_charge(electric,minimum,spacing,radius)
    refined=sphere_charge(electric,minimum,spacing,radius,32,64)
    result.update(charge=refined,charge_quadrature_difference=abs(refined-charge))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--charge-radius',type=float,default=.6)
    parser.add_argument('--charge-tolerance',type=float,default=1.e-6)
    parser.add_argument('--area-relative-tolerance',type=float,default=1.e-4)
    args=parser.parse_args();folder=args.run.resolve();parameters=deck(folder/'input.in')
    assert 'Driver completed.' in (folder/'run.log').read_text()
    model=parameters['plugin/scalar_field'];nr=parameters['numerical_relativity']
    mass=float(model.get('bh_mass',1));charge=float(model.get('bh_charge',.9))
    inner=float(parameters['problem']['constraint_mask_radius']);outer=float(nr['constraint_mask_outer_radius'])
    frames=sorted([inspect(path,inner,outer,args.charge_radius) for path in folder.glob('*.phdf')],key=lambda r:r['time'])
    assert frames[0]['time']==0 and frames[-1]['time']==float(parameters['parthenon/time']['tlim'])
    charges=[r['charge'] for r in frames]
    assert max(abs(value-charge) for value in charges)<args.charge_tolerance
    assert max(charges)-min(charges)<args.charge_tolerance
    assert max(r['charge_quadrature_difference'] for r in frames)<args.charge_tolerance/10
    check_history(folder,dict(initial=frames[0],final=frames[-1]))
    with (folder/'diagnostics/nr_horizon.csv').open() as stream:
        rows=list(csv.DictReader(line for line in stream if not line.startswith('#')))
    assert rows
    analytic_area=4*np.pi*(mass+np.sqrt(mass*mass-charge*charge))**2
    horizons=[]
    for row in rows:
        assert int(row['found'])==1,'Rejected horizon sample retained; hold does not pass'
        residual=float(row['scaled_rms_expansion']);threshold=float(row['expansion_tolerance'])
        assert residual<=threshold*(1+1.e-10)
        assert threshold==float(nr['horizon_expansion_tolerance'])
        area=float(row['area']);relative=abs(area/analytic_area-1)
        assert relative<args.area_relative_tolerance
        time=float(row['time']);q=float(np.interp(time,[r['time'] for r in frames],charges))
        mirr=float(row['irreducible_mass'])
        horizons.append(dict(time=time,area_relative_error=relative,scaled_rms=residual,
                             charged_RN_mass=mirr+q*q/(4*mirr)))
    report=dict(status='short_RN_invariant_checks_passed',frames=frames,horizons=horizons,
                charge_range=max(charges)-min(charges),charge_surface_radius=args.charge_radius,
                input_sha256=hashlib.sha256((folder/'input.in').read_bytes()).hexdigest(),
                criteria=dict(charge_tolerance=args.charge_tolerance,area_relative_tolerance=args.area_relative_tolerance),
                scope='Finite fields, zero scalar, positive metric, fixed-sphere charge, fresh masked history and strict AH samples; constraint convergence, boundary errors and long-time scalar hair remain separate')
    (folder/'RN_HOLD_ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
