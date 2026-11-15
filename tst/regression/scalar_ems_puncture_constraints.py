"""Constraint-solved scalar-pulse BH initial data and a short 3D EMS evolution.

The radial solve constructs spherical isotropic initial data; subsequent PDEs
are the native three-dimensional scalar/Maxwell/Z4c system. No PG data are used.
"""
from pathlib import Path
import argparse
import csv
import json
import shutil

import h5py
import numpy as np

from scalar_ems_coordinates import run
from scalar_ems_rn import norms
from analyze_ems_rn_hold import sphere_charge
from ems_puncture_fixture import Profile,inject,self_check


def inspect(path):
    h=[];mom=[];charge_fields=None;phi_max=0.;energy_max=0.
    with h5py.File(path) as f:
        n=int(f['Info'].attrs['RootGridSize'][0]);spacing=np.array([2/n]*3)
        charge_fields=np.empty((3,n,n,n))
        for b in range(f['scalar.state'].shape[0]):
            u=f['scalar.state'][b];z4c=f['nr.z4c'][b];constraints=f['nr.constraints'][b]
            assert all(np.isfinite(f[name][b]).all() for name in ['scalar.state','nr.z4c','nr.adm','nr.tmunu','nr.constraints','ems.gauss'])
            assert z4c[0].min()>0,'Nonpositive conformal variable'
            x,y,z=[f['VolumeLocations/'+axis][b] for axis in 'xyz']
            r2=x[None,None,:]**2+y[None,:,None]**2+z[:,None,None]**2
            mask=(r2>.3**2)&(r2<=.6**2)
            np.testing.assert_array_equal(f['nr.constraint_mask'][b,0],mask)
            h.extend(constraints[1][mask]);mom.extend(constraints[2][mask])
            phi_max=max(phi_max,float(np.max(np.abs(u[0]))));energy_max=max(energy_max,float(np.max(f['nr.tmunu'][b,6])))
            i,j,k=[int(round((a[0]+1)/spacing[axis]-.5)) for axis,a in enumerate([x,y,z])]
            charge_fields[:,k:k+len(z),j:j+len(y),i:i+len(x)]=u[2:5]
        return dict(time=float(f['Info'].attrs['Time']),hamiltonian=norms(h),momentum=norms(mom),
                    phi_max=phi_max,energy_max=energy_max,
                    charge=sphere_charge(charge_fields,[-1]*3,spacing,.6,32,64))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pangu',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args();root=args.root.resolve();exe=args.pangu.resolve()
    work=args.workdir.resolve();work.mkdir(parents=True,exist_ok=False)
    report=dict(profile_selfcheck=self_check(),resolutions={})
    profile=Profile()
    report['initial_data']=dict(seed_RN_mass=profile.mass,charge=profile.charge,scalar_amplitude=profile.amplitude,
        pulse_support=[profile.left,profile.right],constraint_solved_ADM_mass=profile.adm_mass,
        action='[R-2(dphi)^2-exp(5phi^2)F^2]/16pi',momentum_and_extrinsic_curvature='zero')
    (work/'INITIAL_DATA.json').write_text(json.dumps(report['initial_data'],indent=2)+'\n')
    deck=root/'plugins/scalar_field/examples/ems_cartesian_rn_hold.in'
    fixtures={}
    for n in [64,80,96]:
        block=32 if n%32==0 else 16
        common=['parthenon/time/nlim=0','numerical_relativity/horizon_enabled=false','numerical_relativity/horizon_count=0',
                *[f'parthenon/mesh/nx{axis}={n}' for axis in [1,2,3]],
                *[f'parthenon/meshblock/nx{axis}={block}' for axis in [1,2,3]]]
        seed=work/f'seed_{n}';run(exe,root,seed,common,deck=deck)
        fixture=work/f'constrained_puncture_{n}.rhdf'
        inject(next(seed.glob('*.final.rhdf')),fixture,profile);fixtures[n]=fixture
        path=run(exe,root,work/f'initial_{n}',['parthenon/time/nlim=0'],checkpoint=fixture)
        measured=inspect(path)
        assert measured['phi_max']>.9*profile.amplitude and measured['energy_max']>0
        assert measured['momentum']['Linf']<1.e-13
        report['resolutions'][str(n)]=measured
    errors=[report['resolutions'][str(n)]['hamiltonian']['L2'] for n in [64,80,96]]
    orders=[float(np.log(errors[i]/errors[i+1])/np.log([80/64,96/80][i])) for i in [0,1]]
    report['hamiltonian_observed_orders']=orders
    (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    assert min(orders)>3.2,('Perturbed initial constraint did not converge at fourth order',errors,orders)
    unsolved=work/'unsolved_RN_plus_scalar.rhdf'
    inject(next((work/'seed_96').glob('*.final.rhdf')),unsolved,profile,solve_constraint=False)
    path=run(exe,root,work/'unsolved_control',['parthenon/time/nlim=0'],checkpoint=unsolved)
    report['unsolved_control']=inspect(path)
    assert report['unsolved_control']['hamiltonian']['L2']>5*errors[-1],'Control did not distinguish solved geometry from an added scalar'
    evolved=run(exe,root,work/'evolved',[
        'parthenon/time/nlim=-1','parthenon/time/tlim=0.1',
        'numerical_relativity/horizon_enabled=true','numerical_relativity/horizon_count=1'],checkpoint=fixtures[96])
    report['evolved']=inspect(evolved)
    assert report['evolved']['time']==.1
    assert abs(report['evolved']['charge']-.9)<1.e-6
    with h5py.File(evolved) as f:
        assert np.max(np.abs(f['scalar.state'][:,1]))>1.e-5,'Scalar pulse did not evolve'
    with (work/'evolved/diagnostics/nr_horizon.csv').open() as stream:
        rows=list(csv.DictReader(line for line in stream if not line.startswith('#')))
    assert rows and all(int(row['found'])==1 and float(row['scaled_rms_expansion'])<=float(row['expansion_tolerance'])*(1+1.e-10) for row in rows)
    report.update(status='initial_constraints_and_short_evolution_passed',
        horizon_scaled_rms=[float(row['scaled_rms_expansion']) for row in rows],
        scope='Spherical Hamiltonian-solved isotropic test initial data evolved with 3D EMS/Z4c to0.1M; not general nonspherical initial data, long-time scalar hair, or AMR acceptance')
    (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
