"""Analytic wave convergence, changing AMR topology and repartitioned restart."""
from pathlib import Path
import argparse
import json
import math
import subprocess

import h5py
import numpy as np

from scalar_ems_coordinates import run


def inspect(path, wave=(2, 4, 6), model='scalar_wave'):
    wave=np.asarray(wave)*np.pi;frequency=float(np.linalg.norm(wave));amplitude=1.e-6
    polarization=np.array([1.,-1.,0.])
    polarization-=wave*np.dot(wave,polarization)/np.dot(wave,wave)
    polarization/=np.linalg.norm(polarization)
    magnetic=np.cross(wave/frequency,polarization)
    total=0.;error=0.;gauss2=0.;maximum_gauss=0.;minimum_chi=math.inf
    maximum_h=0.;mask_disagreements=0;maximum_det_error=0.
    with h5py.File(path) as f:
        time=float(f['Info'].attrs['Time']);blocks=int(f['Info'].attrs['NumMeshBlocks'])
        for b in range(blocks):
            values={n:f[n][b] for n in ['scalar.state','nr.z4c','nr.adm','nr.constraints','nr.tmunu','ems.gauss','nr.constraint_mask']}
            assert all(np.isfinite(v).all() for v in values.values()),str(path)
            x,y,z=[f['VolumeLocations/'+a][b] for a in 'xyz']
            dx,dy,dz=[float(f['Locations/'+a][b,1]-f['Locations/'+a][b,0]) for a in 'xyz']
            volume=dx*dy*dz
            xyz=np.broadcast_arrays(x[None,None,:],y[None,:,None],z[:,None,None])
            phase=sum(wave[a]*xyz[a] for a in range(3))-frequency*time
            state=values['scalar.state']
            if model=='scalar_wave':
                delta=np.stack([state[0]-amplitude*np.cos(phase),
                                state[1]/frequency-amplitude*np.sin(phase)])
            else:
                reference=np.array([*polarization,*magnetic])[:,None,None,None]*amplitude*np.cos(phase)
                delta=state[2:8]-reference
            error+=float(np.sum(delta*delta))*volume/delta.shape[0]
            total+=state[0].size*volume
            gauss=values['ems.gauss']
            gauss2+=float(np.sum(gauss*gauss))*volume/2
            maximum_gauss=max(maximum_gauss,float(np.max(np.abs(gauss))))
            minimum_chi=min(minimum_chi,float(values['nr.z4c'][0].min()))
            g=values['nr.z4c'][1:7]
            det=g[0]*(g[3]*g[5]-g[4]*g[4])-g[1]*(g[1]*g[5]-g[2]*g[4])+g[2]*(g[1]*g[4]-g[2]*g[3])
            maximum_det_error=max(maximum_det_error,float(np.max(np.abs(det-1))))
            active=values['nr.constraint_mask'][0]
            radius2=sum(a*a for a in xyz)
            expected=(radius2>.18**2)&(radius2<=.65**2)
            mask_disagreements+=int(np.count_nonzero(active!=expected))
            if expected.any():maximum_h=max(maximum_h,float(np.max(np.abs(values['nr.constraints'][1][expected]))))
        levels=sorted(set(int(v) for v in f['Levels'][...]))
        result=dict(time=time,cycle=int(f['Info'].attrs['NCycle']),blocks=blocks,levels=levels,
            relative_volume_rms_error=math.sqrt(error/total)/amplitude,
            relative_gauss_rms=math.sqrt(gauss2/total)/(amplitude*frequency),
            maximum_gauss=maximum_gauss,minimum_chi=minimum_chi,
            conformal_determinant_linf=maximum_det_error,mask_disagreements=mask_disagreements,
            masked_hamiltonian_linf=maximum_h)
    assert minimum_chi>0 and maximum_det_error<1.e-10,result
    assert mask_disagreements==0,result
    return result


def changes(n, block=None):
    size=block or n//4
    return [*[f'parthenon/mesh/nx{a}={n}' for a in [1,2,3]],
            *[f'parthenon/meshblock/nx{a}={size}' for a in [1,2,3]],
            'numerical_relativity/chi_psi_power=-4',
            'problem/constraint_mask_radius=0.18',
            'numerical_relativity/constraint_mask_outer_radius=0.65']


def compare(left,right):
    names=['scalar.state','nr.z4c','nr.constraint_mask','nr.constraints','ems.gauss']
    def load(path):
        with h5py.File(path) as f:
            keys=[(int(f['Levels'][b]),*[float(f['Locations/'+a][b,0]) for a in 'xyz'])
                  for b in range(len(f['Levels']))]
            order=sorted(range(len(keys)),key=keys.__getitem__)
            minimum_spacing=min(float(np.min(np.diff(f['Locations/'+axis][...],axis=1))) for axis in 'xyz')
            return ([keys[b] for b in order],{n:f[n][...][order] for n in names},
                    float(f['Info'].attrs['Time']),int(f['Info'].attrs['NCycle']),minimum_spacing)
    a,fa,ta,ca,ha=load(left);b,fb,tb,cb,hb=load(right)
    epsilon=np.finfo(float).eps
    assert a==b and abs(ta-tb)<=32*epsilon*max(1.,abs(ta),abs(tb)) and ca==cb,('Restart topology/time differs',ta,tb,ca,cb)
    differences={n:float(np.max(np.abs(fa[n]-fb[n]))) for n in names}
    primary=['scalar.state','nr.z4c','nr.constraint_mask','ems.gauss']
    assert max(differences[n] for n in primary)<5.e-12,differences
    # A second derivative of an O(epsilon) metric difference is O(epsilon/h²).
    # Keep that conditioning bound separate from the primary-field criterion.
    diagnostic_bound=256*epsilon/min(ha,hb)**2
    assert differences['nr.constraints']<diagnostic_bound,(differences,diagnostic_bound)
    differences['time_absolute_difference']=abs(ta-tb)
    differences['constraint_roundoff_bound']=diagnostic_bound
    return differences


def topology_changes(folder):
    topology={}
    # Logical coordinates are local to each Parthenon tree. Use physical
    # boxes so an anisotropic multi-tree mesh cannot alias distinct blocks.
    for path in folder.glob('*.phdf'):
        with h5py.File(path) as f:
            blocks=set()
            for b in range(int(f['Info'].attrs['NumMeshBlocks'])):
                faces=[f['Locations/'+a][b] for a in 'xyz']
                blocks.add(tuple([float(a[0]) for a in faces]+[float(a[-1]) for a in faces]))
            assert len(blocks)==int(f['Info'].attrs['NumMeshBlocks'])
            topology[float(f['Info'].attrs['Time'])]=blocks
    def children(parent):
        middle=[(parent[a]+parent[3+a])/2 for a in range(3)]
        return {tuple([parent[a] if bit[a]==0 else middle[a] for a in range(3)]+
                      [middle[a] if bit[a]==0 else parent[3+a] for a in range(3)])
                for bit in [(i,j,k) for i in [0,1] for j in [0,1] for k in [0,1]]}
    refinements=0;derefinements=0;steps=sorted(topology)
    for c,d in zip(steps,steps[1:]):
        old=topology[c];new=topology[d]
        refinements+=sum(children(parent)<=new for parent in old-new)
        derefinements+=sum(children(parent)<=old for parent in new-old)
    assert refinements>0 and derefinements>0,(refinements,derefinements)
    return dict(steps=len(steps),refined_families=refinements,derefined_families=derefinements,
                block_counts=sorted({len(v) for v in topology.values()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pangu',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--stage',choices=['static','adaptive','all'],default='all')
    args=parser.parse_args();work=args.workdir.resolve();work.mkdir(parents=True,exist_ok=False)
    exe=args.pangu.resolve();root=args.root.resolve();report=dict(status='running',conformal_power=-4)
    def save(): (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    if args.stage in ['static','all']:
        report['static']={}
        try:
            run(exe,root,work/'rejected-small-block',changes(16,4)+['parthenon/mesh/refinement=static'])
            raise AssertionError('AMR accepted a block narrower than its transfer stencil')
        except subprocess.CalledProcessError:
            assert 'interior cells per block axis' in (work/'rejected-small-block/run.log').read_text()
        report['undersized_block_rejected']=True;save()
        for model in ['scalar_wave','maxwell_wave']:
            rows=[]
            sizes=[24,32,40]
            for n in sizes:
                path=run(exe,root,work/(model+'-'+str(n)),changes(n)+[
                    'plugin/scalar_field/initial_data='+model,
                    'parthenon/mesh/refinement=static','parthenon/mesh/numlevel=2',
                    'parthenon/static_refinement0/level=1',
                    'parthenon/static_refinement0/x1min=0.25','parthenon/static_refinement0/x1max=0.75',
                    *[f'parthenon/static_refinement0/x{a}{edge}={value}' for a in [2,3] for edge,value in [('min',0),('max',1)]]])
                row=inspect(path,model=model);assert len(row['levels'])==2,row;rows.append(row)
                report['static'][model]=dict(resolutions=sizes,samples=rows);save()
            for metric in ['relative_volume_rms_error']+(['relative_gauss_rms'] if model=='maxwell_wave' else []):
                orders=[math.log(rows[i][metric]/rows[i+1][metric])/math.log(sizes[i+1]/sizes[i]) for i in [0,1]]
                report['static'][model][metric+'_orders']=orders;save()
                assert min(orders)>3.2,(model,metric,orders,rows)
    if args.stage in ['adaptive','all']:
        report['adaptive']={}
        for model,component in [('scalar_wave',0),('maxwell_wave',3)]:
            common=changes(12,6)+['parthenon/mesh/nx1=24','parthenon/meshblock/nx1=6',
                'parthenon/mesh/refinement=adaptive','parthenon/mesh/numlevel=2',
                'parthenon/mesh/derefine_count=1','parthenon/loadbalancing/interval=1',
                'numerical_relativity/amr_method=chi','numerical_relativity/amr_chi_min=0.5',
                'plugin/scalar_field/initial_data='+model,
                'plugin/scalar_field/wave_x=6.283185307179586','plugin/scalar_field/wave_y=0','plugin/scalar_field/wave_z=0',
                'parthenon/refinement0/method=magnitude','parthenon/refinement0/field=scalar.state',
                'parthenon/refinement0/vector_i='+str(component),'parthenon/refinement0/max_level=1',
                'parthenon/refinement0/refine_tol=0.00000095','parthenon/refinement0/derefine_tol=0.0000008',
                'parthenon/time/tlim=10','parthenon/output1/dt=0.02','parthenon/output3/dt=1']
            direct_folder=work/(model+'-direct')
            direct=run(exe,root,direct_folder,common+['parthenon/time/nlim=160'],ranks=2)
            row=inspect(direct,wave=(2,0,0),model=model)
            movement=topology_changes(direct_folder)
            first_folder=work/(model+'-first')
            run(exe,root,first_folder,common+['parthenon/time/nlim=80'])
            checkpoint=next(first_folder.glob('*.final.rhdf'))
            continued_folder=work/(model+'-restart')
            continued=run(exe,root,continued_folder,['parthenon/time/nlim=160'],ranks=2,checkpoint=checkpoint)
            differences=compare(direct,continued)
            report['adaptive'][model]=dict(final=row,topology=movement,restart_maximum_differences=differences,
                                          restart_rank_change='1 to 2');save()
    report.update(status='passed',scope='Cartesian FD4 AMR analytic wave convergence, geometric masks, actual refinement/derefinement and 1-to-2-rank restart. Strong-field and long-time gates are separate.')
    save();print(json.dumps(report,indent=2))


if __name__=='__main__': main()
