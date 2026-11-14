"""CUDA 3D wave, geometric-feedback, Gauss and restart acceptance checks."""
from pathlib import Path
import argparse
import json
import os
import subprocess

import h5py
import numpy as np


def run(executable, root, folder, changes, ranks=1, checkpoint=None, deck=None):
    folder.mkdir()
    command = ['mpirun', '-np', str(ranks), str(executable)]
    if checkpoint:
        command += ['-r', str(checkpoint)]
    else:
        command += ['-i', str(deck or root/'plugins/scalar_field/examples/ems_cartesian_vacuum.in')]
    command += changes
    with (folder/'run.log').open('wb') as output:
        subprocess.run(command, cwd=folder, stdout=output, stderr=subprocess.STDOUT, check=True)
    if 'Driver completed.' not in (folder/'run.log').read_text():
        raise RuntimeError('Driver completion missing: '+str(folder))
    return next(folder.glob('*.final.phdf'))


def fields(path):
    with h5py.File(path) as f:
        result = {name: f[name][...] for name in ['scalar.state','ems.gauss','nr.z4c','nr.tmunu','nr.constraints']}
        if not all(np.isfinite(value).all() for value in result.values()):
            raise RuntimeError('Nonfinite output: '+str(path))
        result['time'] = float(f['Info'].attrs['Time'])
        result['cycle'] = int(f['Info'].attrs['NCycle'])
        result['xyz'] = [f['VolumeLocations/'+axis][...] for axis in 'xyz']
    return result


def wave_error(output, model):
    state=output['scalar.state']; x,y,z=output['xyz']; k=np.array([2,4,6])*np.pi
    frequency=float(np.linalg.norm(k)); amplitude=1.e-6
    errors=[]
    pol=np.array([1.,-1.,0.]); pol-=k*np.dot(k,pol)/np.dot(k,k); pol/=np.linalg.norm(pol)
    magnetic=np.cross(k/frequency,pol)
    for block in range(state.shape[0]):
        phase=(k[0]*x[block][None,None,:]+k[1]*y[block][None,:,None]+
               k[2]*z[block][:,None,None]-frequency*output['time'])
        if model=='scalar_wave':
            errors.append((state[block,0]-amplitude*np.cos(phase)).ravel())
            errors.append((state[block,1]/frequency-amplitude*np.sin(phase)).ravel())
        else:
            for axis in range(3):
                errors.append((state[block,2+axis]-amplitude*pol[axis]*np.cos(phase)).ravel())
                errors.append((state[block,5+axis]-amplitude*magnetic[axis]*np.cos(phase)).ravel())
    return float(np.sqrt(np.mean(np.concatenate(errors)**2))/amplitude)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pangu',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args(); exe=args.pangu.resolve(); root=args.root.resolve()
    work=args.workdir.resolve(); work.mkdir(parents=True,exist_ok=False)
    report={}
    pg=run(exe,root,work/'pg-rn',[
        'plugin/scalar_field/initial_data=reissner_nordstrom',
        'parthenon/mesh/nx1=512','parthenon/meshblock/nx1=512',
        'parthenon/mesh/x1max=12','parthenon/time/tlim=0.01'],
        deck=root/'plugins/scalar_field/examples/ems_coupled_rn.in')
    with h5py.File(pg) as f:
        assert 'nr.z4c' not in f,'PG mode installed an unrelated NR evolution'
        state=f['scalar.state'][...]
        assert np.isfinite(state).all() and np.max(np.abs(state[:,:2]))==0
        radius=f['VolumeLocations/x'][...]
        error=float(np.max(np.abs(state[:,2]-
                       (1-0.9**2/(2*radius[:,None,None,:])))))
        assert error<1.e-11,('PG RN mass preservation',error)
    report['pg']={'coordinates':'pg','RN_mass_maximum_error':error,'status':'passed'}
    vacuum=fields(run(exe,root,work/'vacuum',[]))
    assert np.max(np.abs(vacuum['scalar.state']))==0
    assert np.max(np.abs(vacuum['nr.tmunu']))==0
    assert np.max(np.abs(vacuum['nr.constraints']))<1.e-14
    report['vacuum']='passed'
    for model in ['scalar_wave','maxwell_wave']:
        values=[]
        for n in [16,32]:
            path=run(exe,root,work/(model+'-'+str(n)),[
                'plugin/scalar_field/initial_data='+model,
                *['parthenon/mesh/nx'+str(axis)+'='+str(n) for axis in [1,2,3]]])
            result=fields(path); error=wave_error(result,model); values.append(error)
            assert np.max(result['nr.tmunu'][:,6])>0,'matter energy missing'
            assert np.max(np.abs(result['nr.z4c'][:,7]))>0,'matter did not source geometry'
            if model=='maxwell_wave':
                assert np.max(np.abs(result['ems.gauss']))<1.e-11
        order=float(np.log2(values[0]/values[1])); assert order>3.2,(model,values,order)
        report[model]={'relative_rms_errors':values,'observed_order':order,'geometric_backreaction':True}
    common=['plugin/scalar_field/initial_data=maxwell_wave','parthenon/time/tlim=10']
    direct=fields(run(exe,root,work/'direct',common+['parthenon/time/nlim=8'],ranks=2))
    run(exe,root,work/'first',common+['parthenon/time/nlim=4'])
    restart=next((work/'first').glob('*.final.rhdf'))
    continued=fields(run(exe,root,work/'restart',['parthenon/time/nlim=8'],ranks=2,checkpoint=restart))
    assert direct['time']==continued['time'] and direct['cycle']==continued['cycle']==8
    differences={}
    for name in ['scalar.state','nr.z4c']:
        difference=float(np.max(np.abs(direct[name]-continued[name])))
        assert difference<5.e-12,(name,difference)
        differences[name]=difference
    report['restart']={'status':'passed','maximum_absolute_differences':differences,'rank_change':'1 to 2'}
    report['scope']='3D CUDA PDE/feedback and restart verification; not long-time scalarized-BH physical acceptance'
    (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
