"""Nonlinear EMS coupling checked against an independent Bianchi I ADM ODE.

A labeled synthetic restart fixture supplies spatially homogeneous,
constraint-satisfying initial data. It is not a black-hole initial-data solver.
"""
from pathlib import Path
import argparse
import json
import shutil

import h5py
import numpy as np

from scalar_ems_coordinates import run

PHI=0.2
P=0.1
D=0.4
MASS=0.3
QUARTIC=0.4
SEXTIC=0.2
ALPHA=5.


def potential(phi):
    return MASS**2*phi**2/2+QUARTIC*phi**4/2+SEXTIC*phi**6/6


def initial():
    rho=P**2/2+potential(PHI)+D**2/(2*np.exp(ALPHA*PHI**2))
    expansion=np.sqrt(2*rho/3)
    return np.array([PHI,P,1.,1.,1.,-expansion,-expansion,-expansion])


def rhs(state):
    phi,p=state[:2];metric=state[2:5];extrinsic=state[5:]
    trace=float(np.sum(extrinsic/metric))
    coupling=np.exp(ALPHA*phi*phi)
    electric=np.array([D,0.,0.])/(np.sqrt(np.prod(metric))*coupling)
    electric_lower=metric*electric
    electric_squared=float(np.sum(electric_lower*electric))
    gradient=MASS**2*phi+2*QUARTIC*phi**3+SEXTIC*phi**5
    # Homogeneous ADM Einstein equations, alpha=1 and beta=0. The geometric
    # variables here are physical diagonal gamma_ii and K_ii, not Z4c fields.
    return np.concatenate(([p,trace*p-gradient+ALPHA*phi*coupling*electric_squared],
        -2*extrinsic,trace*extrinsic-2*extrinsic**2/metric-
        2*potential(phi)*metric-coupling*electric_squared*metric+2*coupling*electric_lower**2))


def reference(time,steps):
    state=initial();dt=time/steps
    for _ in range(steps):
        a=rhs(state);b=rhs(state+dt*a/2);c=rhs(state+dt*b/2);d=rhs(state+dt*c)
        state+=dt*(a+2*b+2*c+d)/6
    return state


def inspect(path):
    with h5py.File(path) as f:
        scalar=f['scalar.state'][...];adm=f['nr.adm'][...]
        assert np.isfinite(scalar).all() and np.isfinite(adm).all()
        state=np.array([np.mean(scalar[:,0]),np.mean(scalar[:,1]),
                        *[np.mean(adm[:,n]) for n in [0,3,5,6,9,11]]])
        np.testing.assert_allclose(scalar[:,2],D,atol=1.e-13,rtol=0)
        np.testing.assert_allclose(scalar[:,3:],0,atol=1.e-13,rtol=0)
        for data in [scalar,adm]:
            for component in range(data.shape[1]):
                assert np.ptp(data[:,component])<1.e-11,('Lost homogeneity',component)
        return dict(state=state.tolist(),time=float(f['Info'].attrs['Time']),
                    hamiltonian_max=float(np.max(np.abs(f['nr.constraints'][:,1]))),
                    gauss_max=float(np.max(np.abs(f['ems.gauss'][...]))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pangu',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--chi-psi-power',type=int,choices=[-4,-2],default=-4)
    args=parser.parse_args();exe=args.pangu.resolve();root=args.root.resolve()
    work=args.workdir.resolve();work.mkdir(parents=True,exist_ok=False)
    model=['plugin/scalar_field/initial_data=minkowski','plugin/scalar_field/em_alpha='+str(ALPHA),
           'numerical_relativity/chi_psi_power='+str(args.chi_psi_power),
           'plugin/scalar_field/mass='+str(MASS),'plugin/scalar_field/quartic='+str(QUARTIC),
           'plugin/scalar_field/sextic='+str(SEXTIC),'numerical_relativity/lapse_gauge=frozen',
           'numerical_relativity/shift_gauge=frozen','parthenon/output1/dt=2',
           'parthenon/output2/dt=2','parthenon/output3/dt=2']
    # No spatial structure is present. A small grid keeps the three timestep
    # errors above roundoff instead of making the finest result noise-limited.
    model += [f'parthenon/mesh/nx{axis}=8' for axis in [1,2,3]]
    run(exe,root,work/'seed',model+['parthenon/time/nlim=0'])
    fixture=work/'analytic-bianchi-initial.rhdf'
    shutil.copyfile(next((work/'seed').glob('*.final.rhdf')),fixture)
    state=initial()
    with h5py.File(fixture,'r+') as f:
        u=f['scalar.state'][...];u[:]=0;u[:,0]=PHI;u[:,1]=P;u[:,2]=D;f['scalar.state'][...]=u
        z=f['nr.z4c'][...];z[:]=0;z[:,0]=1
        for n in [1,4,6]:z[:,n]=1
        z[:,7]=np.sum(state[5:]);z[:,18]=1;f['nr.z4c'][...]=z
        f['nr.constraint_mask'][...]=1
    (work/'FIXTURE.json').write_text(json.dumps(dict(kind='analytic homogeneous EMS initial data',
        method='Independent fields placed in an explicitly synthetic restart fixture',
        scalar_phi=PHI,scalar_P=P,electric_density_x=D,initial_ADM_state=state.tolist(),
        constraint='K^2-KijKij=16pi rho; div(D)=div(B)=0; homogeneous momentum=0'),indent=2)+'\n')
    final_time=2.
    exact=reference(final_time,8192);fine=reference(final_time,16384)
    reference_error=float(np.max(np.abs(exact-fine)))
    assert reference_error<1.e-11,('ODE reference unresolved',reference_error)
    report=dict(reference_state=fine.tolist(),reference_refinement_error=reference_error,
                conformal_power=args.chi_psi_power,cases=[])
    for cfl in [.45,.225,.1125]:
        folder=work/('cfl-'+str(cfl))
        output=run(exe,root,folder,['parthenon/time/tlim=2','parthenon/time/nlim=-1',
            'numerical_relativity/cfl='+str(cfl),'plugin/scalar_field/cfl='+str(cfl)],checkpoint=fixture)
        # Restart continues the seed output counter; select by physical time.
        initial_outputs=[]
        for path in folder.glob('*.phdf'):
            with h5py.File(path) as f:
                if float(f['Info'].attrs['Time'])==0:initial_outputs.append(path)
        assert len(initial_outputs)==1,('Missing initial restart output',folder)
        start=inspect(initial_outputs[0])
        assert start['hamiltonian_max']<1.e-13,'Synthetic initial condition violates Hamiltonian constraint'
        measured=inspect(output)
        assert measured['time']==final_time
        error=float(np.max(np.abs(np.array(measured['state'])-fine)))
        report['cases'].append(dict(cfl=cfl,maximum_ADM_error=error,**measured))
    errors=[c['maximum_ADM_error'] for c in report['cases']]
    orders=[float(np.log2(errors[i]/errors[i+1])) for i in range(2)]
    report['observed_orders']=orders
    (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    assert min(orders)>3.2,('Nonlinear EMS temporal convergence failed',errors,orders)
    assert errors[-1]<1.e-8
    assert abs(fine[0]-PHI)>.01 and abs(fine[2]-fine[3])>.01,'Fixture did not exercise nonlinear exchange and anisotropic geometry'
    report.update(status='passed',scope='Homogeneous nonlinear scalar-Maxwell-Einstein evolution versus independent ADM ODE; not black-hole or spatial-derivative acceptance')
    (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
