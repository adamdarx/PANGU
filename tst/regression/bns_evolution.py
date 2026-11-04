#!/usr/bin/env python3
"""GPU BNS importer, discrete constraints, short evolution and restart admission."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import h5py
import numpy as np


def edit(text, changes):
    section = ''
    used = set()
    result = []
    for line in text.splitlines():
        if line.strip().startswith('<'):
            section = line.strip()[1:-1]
        key = line.split('=',1)[0].strip()
        if (section,key) in changes:
            used.add((section,key))
            line = key+' = '+str(changes[section,key])
        result.append(line)
    if used != set(changes):
        raise ValueError('unknown input edits: '+str(set(changes)-used))
    return '\n'.join(result)+'\n'


def inspect(path):
    with h5py.File(path) as f:
        for name in f:
            if name.startswith(('mhd.','nr.')):
                assert np.isfinite(f[name][:]).all(), (str(path),name)
        prim,cons = f['mhd.prim'][:],f['mhd.cons'][:]
        z4c,tmunu,c = f['nr.z4c'][:],f['nr.tmunu'][:],f['nr.constraints'][:]
        xyz = [f['Locations/'+a][:] for a in 'xyz']
        widths = [np.diff(x,axis=1) for x in xyz]
        volume = widths[2][:,:,None,None]*widths[1][:,None,:,None]*widths[0][:,None,None,:]
        active = prim[:,0]>1.e-6
        core = prim[:,0] > .2*prim[:,0].max()
        norm = 16*np.pi*tmunu[:,6].max()
        diagnostics = dict(time=float(f['Info'].attrs['Time']),blocks=len(prim),
                           baryon_mass=float(np.sum(cons[:,0]*volume)),
                           density_max=float(prim[:,0].max()),
                           lapse_min=float(z4c[:,18].min()),
                           chi_min=float(z4c[:,0].min()),
                           core_hamiltonian_l2_scaled=float(np.sqrt(np.mean(c[:,1][core]**2))/norm),
                           material_hamiltonian_l2_scaled=float(np.sqrt(np.mean(c[:,1][active]**2))/norm),
                           momentum_max=float(abs(c[:,4:7]).max()),
                           recovery_sums=[float(np.sum(a)) for a in np.moveaxis(f['mhd.recovery'][:],1,0)],
                           material_entropy_min=float(np.min(prim[:,4][core]/prim[:,0][core]**2)))
        assert diagnostics['chi_min']>0 and diagnostics['lapse_min']>0
        assert prim[:,0].min()>0 and prim[:,4].min()>0
        assert abs(f['mhd.b_cell'][:]).max() == 0
        assert abs(f['mhd.divb'][:]).max() == 0
        return diagnostics


def importer_check(path, producer, profile, r, coeff, moments):
    with h5py.File(path) as f:
        xyz = [f['Locations/'+a][:] for a in 'xyz']
        xyz = [(v[:,:-1]+v[:,1:])/2 for v in xyz]
        shape = f['mhd.prim'][:,0].shape
        coords = [np.broadcast_to(xyz[0][:,None,None,:],shape),
                  np.broadcast_to(xyz[1][:,None,:,None],shape),
                  np.broadcast_to(xyz[2][:,:,None,None],shape)]
        indices = np.random.default_rng(713).choice(np.prod(shape),min(4096,np.prod(shape)),replace=False)
        x,y,z = [v.ravel()[indices] for v in coords]
        expected = producer.evaluate(r,coeff,moments,x,y,z)
        actual = f['nr.z4c'][:,0].ravel()[indices]**(-.25)
        psi_error = float(abs(actual-expected).max())
        assert psi_error < 3e-14,psi_error
        ebar = producer.free_source(profile,np.hypot(x,y),z,24)
        energy = ebar/expected**8
        rho = 2*energy/(1+np.sqrt(1+4*profile['kappa']*energy))
        active = rho > 1e-6
        actual_rho = f['mhd.prim'][:,0].ravel()[indices]
        rho_error = float(np.max(abs(actual_rho[active]/rho[active]-1)))
        actual_e = f['nr.tmunu'][:,6].ravel()[indices]
        energy_error = float(np.max(abs(actual_e[active]/energy[active]-1)))
        assert rho_error < 2e-12 and energy_error < 2e-12
        assert abs(f['mhd.prim'][:,1:4]).max()==0
        assert abs(f['nr.z4c'][:,7:18]).max()==0
        return dict(psi_abs_error=psi_error,rho_relative_error=rho_error,
                    stress_energy_relative_error=energy_error)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pangu',required=True,type=Path)
    p.add_argument('--table',required=True,type=Path)
    p.add_argument('--workdir',required=True,type=Path)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('bns_source',repo/'plugins/compact_objects/bns_data.py')
    m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    profile=m.tov()
    r,coeff,moments,_=m.solve(profile,step=.0125,lmax=128,angular_points=384)
    # Bind the tested file to the freshly recomputed producer, not just to a
    # claimed residual written in its header.
    a.workdir.mkdir(parents=True,exist_ok=False)
    m.write(a.workdir/'recomputed.dat',profile,24,r,coeff,moments,_)
    assert (a.workdir/'recomputed.dat').read_bytes()==a.table.read_bytes()
    base=(repo/'plugins/compact_objects/examples/bns_head_on_admission.in').read_text()
    binary=a.pangu.resolve()
    report=dict(status='running',binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                table_sha256=hashlib.sha256(a.table.read_bytes()).hexdigest(),initial=[],runs={},
                acceptance=dict(short_time_baryon_drift=2e-4,restart_primary_scaled_error=5e-12))
    def save():
        (a.workdir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,changes,restart=None):
        directory=a.workdir/name;directory.mkdir()
        changes={('plugin/compact_objects','initial_data_file'):str(a.table.resolve()),**changes}
        deck=directory/'input.in';deck.write_text(edit(base,changes))
        args=[str(binary),'-i',str(deck.resolve())]
        if restart:args+=['-r',str(restart.resolve())]
        start=time.monotonic()
        with (directory/'run.log').open('w') as log:
            proc=subprocess.run(args,cwd=directory,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        report['runs'][name]=dict(exit_code=proc.returncode,wall_seconds=time.monotonic()-start)
        assert proc.returncode==0 and 'Driver completed.' in (directory/'run.log').read_text(),name
        return directory
    save()
    try:
        for n in [32,48,64]:
            changes={('parthenon/time','tlim'):0}
            for axis in [1,2,3]:changes['parthenon/mesh',f'nx{axis}']=n
            d=run(f'initial_n{n}',changes)
            path=next(d.glob('*.00000.phdf'))
            diag=inspect(path);diag.update(resolution=n,importer=importer_check(path,m,profile,r,coeff,moments))
            assert diag['momentum_max']==0
            report['initial'].append(diag);save()
        errs=[x['material_hamiltonian_l2_scaled'] for x in report['initial']]
        assert errs[2]<errs[1]<errs[0],errs
        assert errs[-1]<.002
        direct=run('evolve_n64',{})
        initial=inspect(next(direct.glob('*.00000.phdf')))
        final=inspect(next(direct.glob('*.final.phdf')))
        assert abs(final['time']-2)<1e-12
        drift=abs(final['baryon_mass']/initial['baryon_mass']-1)
        assert drift<report['acceptance']['short_time_baryon_drift'],drift
        report['short_evolution']=dict(initial=initial,final=final,baryon_drift=drift)
        cp=sorted(direct.glob('*.00001.rhdf'))[0]
        restarted=run('restart_n64',{},cp)
        differences={}
        with h5py.File(next(direct.glob('*.final.phdf'))) as f,h5py.File(next(restarted.glob('*.final.phdf'))) as g:
            for name in ['nr.z4c','mhd.cons','mhd.b_face','mhd.b_cell']:
                left,right=f[name][:],g[name][:]
                diff=float(abs(left-right).max()/max(1.,abs(left).max()))
                differences[name]=diff
                assert diff<5e-12,(name,diff)
        report['restart_scaled_differences']=differences
        report.update(status='passed_local_uniform_admission',production_qualified=False)
        save()
    except BaseException as error:
        report.update(status='failed',error=repr(error));save();raise
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
