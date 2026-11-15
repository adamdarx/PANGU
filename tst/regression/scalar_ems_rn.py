"""Local electrovac RN constraints, history masks and repartitioned restart.

The fixed coordinate shell excludes the unresolved puncture and boundaries.
Passing this short check does not validate a resolved horizon or scalar hair.
"""
from pathlib import Path
import argparse
import json
import re

import h5py
import numpy as np

from scalar_ems_coordinates import run


def norms(values):
    values=np.asarray(values)
    return dict(L1=float(np.mean(np.abs(values))),
                L2=float(np.sqrt(np.mean(values*values))),
                Linf=float(np.max(np.abs(values))))


def inspect(path,initial=False):
    collected={name:[] for name in ['hamiltonian','electric_gauss','momentum']}
    active=0
    with h5py.File(path) as output:
        for name in ['scalar.state','ems.gauss','nr.z4c','nr.adm','nr.tmunu','nr.constraints','nr.constraint_mask']:
            assert np.isfinite(output[name][...]).all(),('nonfinite',path,name)
        state=output['scalar.state'][...]
        assert np.max(np.abs(state[:,:2]))==0,'Unperturbed RN generated a scalar'
        for block in range(len(state)):
            x,y,z=[output['VolumeLocations/'+axis][block] for axis in 'xyz']
            xyz=np.broadcast_arrays(x[None,None,:],y[None,:,None],z[:,None,None])
            radius=np.sqrt(sum(axis*axis for axis in xyz))
            mask=(radius>1.5)&(radius<=3.)
            assert np.array_equal(output['nr.constraint_mask'][block,0],mask),(
                'Configured diagnostic mask was not applied',path,block)
            active+=int(mask.sum())
            for name,field,component in [('hamiltonian','nr.constraints',1),
                                         ('electric_gauss','ems.gauss',0),
                                         ('momentum','nr.constraints',2)]:
                collected[name].extend(output[field][block,component][mask])
            if initial:
                # Closed-form isotropic electrovac data, independently evaluated
                # on the output coordinates rather than read from NR internals.
                psi=np.sqrt(1+1/radius+(1-0.9**2)/(4*radius**2))
                for axis in range(3):
                    np.testing.assert_allclose(state[block,2+axis],0.9*xyz[axis]/radius**3,rtol=5.e-14,atol=5.e-14)
                np.testing.assert_allclose(output['nr.adm'][block,12],psi**4,rtol=5.e-14,atol=5.e-14)
                rho=0.9**2/(8*np.pi*psi**8*radius**4)
                np.testing.assert_allclose(output['nr.tmunu'][block,6],rho,rtol=5.e-13,atol=5.e-14)
                np.testing.assert_array_equal(state[block,5:],0)
                # Independent FD4 divergence of the analytic electric density.
                divergence=np.zeros_like(radius)
                for axis,coordinates in enumerate([x,y,z]):
                    spacing=float(coordinates[1]-coordinates[0])
                    for offset,weight in [(-2,1),(-1,-8),(1,8),(2,-1)]:
                        shifted=list(xyz)
                        shifted[axis]=xyz[axis]+offset*spacing
                        shifted_radius=np.sqrt(sum(value*value for value in shifted))
                        divergence+=weight*0.9*shifted[axis]/shifted_radius**3/(12*spacing)
                np.testing.assert_allclose(output['ems.gauss'][block,0][mask],
                                           (divergence/psi**6)[mask],rtol=2.e-10,atol=5.e-14)
        assert 0<active<state.shape[0]*np.prod(state.shape[2:]),'Empty or unmasked diagnostic shell'
        time=float(output['Info'].attrs['Time'])
    return dict(time=time,active_cells=active,**{name:norms(value) for name,value in collected.items()})


def check_history(folder,report):
    path=next(folder.glob('*.hst'))
    columns={name:int(index)-1 for index,name in re.findall(r'\[(\d+)\]=([^\s]+)',path.read_text().splitlines()[1])}
    data=np.atleast_2d(np.loadtxt(path))
    for sample in [report['initial'],report['final']]:
        rows=data[np.isclose(data[:,columns['time']],sample['time'],atol=1.e-12,rtol=0)]
        assert len(rows)==1,('Missing or duplicate history endpoint',sample['time'])
        row=rows[0]
        assert row[columns['nr_constraint_mask_cells_1']]==sample['active_cells']
        for norm,name in [('L1','nr_constraint_norm_2'),('L2','nr_constraint_norm_3'),('Linf','nr_constraint_max_1')]:
            np.testing.assert_allclose(row[columns[name]],sample['hamiltonian'][norm],rtol=2.e-11,atol=2.e-13,
                                       err_msg='History disagrees with fresh masked Hamiltonian field')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pangu',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--chi-psi-power',type=int,choices=[-4,-2],default=-4)
    args=parser.parse_args()
    root=args.root.resolve();exe=args.pangu.resolve();work=args.workdir.resolve()
    work.mkdir(parents=True,exist_ok=False)
    deck=root/'plugins/scalar_field/examples/ems_cartesian_rn.in'
    geometry=['numerical_relativity/chi_psi_power='+str(args.chi_psi_power)]
    report=dict(status='running',conformal_power=args.chi_psi_power,resolutions={},mask=dict(inner=1.5,outer=3.,
                norm='coordinate-volume L1/L2/Linf in a fixed isotropic-coordinate shell'))
    for n in [32,48,64]:
        folder=work/('n'+str(n))
        final=run(exe,root,folder,geometry+[f'parthenon/mesh/nx{axis}={n}' for axis in [1,2,3]],deck=deck)
        initial=next(folder.glob('*.00000.phdf'))
        item=dict(initial=inspect(initial,initial=True),final=inspect(final))
        assert item['initial']['time']==0 and item['final']['time']==0.25
        check_history(folder,item)
        report['resolutions'][str(n)]=item
    report['observed_orders']={}
    for phase in ['initial','final']:
        report['observed_orders'][phase]={}
        for field in ['hamiltonian','electric_gauss']:
            errors=[report['resolutions'][str(n)][phase][field]['L2'] for n in [32,48,64]]
            orders=[float(np.log(errors[i]/errors[i+1])/np.log([48/32,64/48][i])) for i in range(2)]
            report['observed_orders'][phase][field]=orders
            assert min(orders)>3.2,('RN exterior constraints did not converge',phase,field,errors,orders)
    # Check MPI history reductions as well as both evolved fields and the
    # independent, checkpointed mask when repartitioning this curved geometry.
    common=geometry+['parthenon/time/tlim=10']
    direct=run(exe,root,work/'direct',common+['parthenon/time/nlim=8'],ranks=2,deck=deck)
    direct_report=dict(initial=inspect(next((work/'direct').glob('*.00000.phdf')),True),final=inspect(direct))
    check_history(work/'direct',direct_report)
    run(exe,root,work/'first',common+['parthenon/time/nlim=4'],deck=deck)
    restart=next((work/'first').glob('*.final.rhdf'))
    continued=run(exe,root,work/'restart',['parthenon/time/nlim=8'],ranks=2,checkpoint=restart)
    differences={}
    with h5py.File(direct) as a,h5py.File(continued) as b:
        assert a['Info'].attrs['Time']==b['Info'].attrs['Time']
        assert a['Info'].attrs['NCycle']==b['Info'].attrs['NCycle']==8
        for name in ['scalar.state','nr.z4c','nr.constraint_mask','nr.constraints']:
            difference=float(np.max(np.abs(a[name][...]-b[name][...])))
            assert difference<5.e-12,(name,difference)
            differences[name]=difference
    inspect(continued)
    report.update(status='passed',restart=dict(rank_change='1 to 2',maximum_absolute_differences=differences),
                  scope='Short electrovac RN exterior constraints and restart; unresolved horizon; no scalarized-BH long-time acceptance')
    (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
