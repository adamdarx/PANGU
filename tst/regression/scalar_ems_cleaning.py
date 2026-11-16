"""Damped electric/magnetic longitudinal waves, shift transport and10-field restart."""
from pathlib import Path
import argparse
import json
import math
import shutil

import h5py
import numpy as np

from scalar_ems_coordinates import run,fields


def inject_longitudinal(source,target,shift):
    shutil.copyfile(source,target)
    with h5py.File(target,'r+') as f:
        assert f['scalar.state'].shape[1]==10
        for b in range(len(f['Levels'])):
            u=f['scalar.state'][b];u[:]=0;u[0]=.25
            phase=2*np.pi*f['VolumeLocations/x'][b][None,None,:]
            u[2]=1.e-6*np.sin(phase);u[5]=.5e-6*np.cos(phase)
            f['scalar.state'][b]=u
            z=f['nr.z4c'][b];z[19]=shift;f['nr.z4c'][b]=z


def error(path,shift):
    amplitude=1.e-6;k=2*np.pi;kappa=1.;omega=math.sqrt(k*k-kappa*kappa/4)
    with h5py.File(path) as f:
        t=float(f['Info'].attrs['Time']);decay=math.exp(-kappa*t/2)
        field=decay*(math.cos(omega*t)+kappa*math.sin(omega*t)/(2*omega))
        cleaning=-decay*k*math.sin(omega*t)/omega;errors=[]
        for b in range(len(f['Levels'])):
            u=f['scalar.state'][b];phase=k*(f['VolumeLocations/x'][b][None,None,:]+shift*t)
            assert np.isfinite(u).all() and u.shape[0]==10
            for c,expected in [(2,amplitude*field*np.sin(phase)),(5,.5*amplitude*field*np.cos(phase)),(8,amplitude*cleaning*np.cos(phase)),(9,-.5*amplitude*cleaning*np.sin(phase))]:
                errors.append((u[c]-expected).ravel()/amplitude)
            assert np.max(np.abs(u[0]-.25))<1.e-10
        return float(np.sqrt(np.mean(np.concatenate(errors)**2)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pangu',type=Path,required=True);parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args();exe=args.pangu.resolve();root=args.root.resolve();work=args.workdir.resolve();work.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scope='Analytic longitudinal electric/magnetic constraint waves with nonunit f, constant shift, transverse-wave equivalence and1-to2-MPI restart; strong-field acceptance remains separate',samples=[])
    def save():(work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    save();checkpoints={};errors=[]
    for n,shift in [(16,0.),(32,0.),(32,.2)]:
        name=str(n)+('-shift' if shift else '')
        common=[*[f'parthenon/mesh/nx{a}={n}' for a in [1,2,3]],'parthenon/time/tlim=10',
            'numerical_relativity/lapse_gauge=frozen','numerical_relativity/shift_gauge=frozen',
            'plugin/scalar_field/maxwell_cleaning=true','plugin/scalar_field/cleaning_damping=1',
            'plugin/scalar_field/initial_data=minkowski','plugin/scalar_field/em_alpha=5',
            'parthenon/output1/dt=10','parthenon/output3/dt=10','parthenon/time/ncycle_out=20']
        seed=work/('seed-'+name);run(exe,root,seed,common+['parthenon/time/nlim=0'])
        checkpoint=work/('longitudinal-'+name+'.rhdf');inject_longitudinal(next(seed.glob('*.final.rhdf')),checkpoint,shift)
        checkpoints[name]=checkpoint
        final=run(exe,root,work/('wave-'+name),['parthenon/time/nlim=-1','parthenon/time/tlim=.2'],checkpoint=checkpoint)
        value=error(final,shift);report['samples'].append(dict(n=n,shift=shift,relative_RMS=value));save()
        assert value<.005,(name,value)
        if not shift:errors.append(value)
    report['spatial_order']=math.log2(errors[0]/errors[1]);assert report['spatial_order']>3.5;save()

    transverse=[]
    for enabled in ['false','true']:
        output=fields(run(exe,root,work/('transverse-'+enabled),['plugin/scalar_field/initial_data=maxwell_wave','plugin/scalar_field/maxwell_cleaning='+enabled,'parthenon/time/tlim=.025']))
        transverse.append(output['scalar.state'])
    difference=float(np.max(np.abs(transverse[0]-transverse[1][:,:8])))
    assert difference<5.e-12,difference
    assert np.max(np.abs(transverse[1][:,8:]))<1.e-12
    report['transverse_on_off_max_difference']=difference;save()

    direct=run(exe,root,work/'mpi-direct',['parthenon/time/nlim=16'],ranks=2,checkpoint=checkpoints['16'])
    first=work/'mpi-first';run(exe,root,first,['parthenon/time/nlim=8'],checkpoint=checkpoints['16'])
    continued=run(exe,root,work/'mpi-continued',['parthenon/time/nlim=16'],ranks=2,checkpoint=next(first.glob('*.final.rhdf')))
    with h5py.File(direct) as a,h5py.File(continued) as b:
        assert int(a['Info'].attrs['NCycle'])==int(b['Info'].attrs['NCycle'])==16
        assert abs(float(a['Info'].attrs['Time'])-float(b['Info'].attrs['Time']))<1.e-14
        differences={key:float(np.max(np.abs(a[key][...]-b[key][...]))) for key in ['scalar.state','nr.z4c']}
        assert max(differences.values())<5.e-12,differences
    report.update(status='passed',restart_maximum_differences=differences);save();print(json.dumps(report,indent=2))


if __name__=='__main__':main()
