"""Check native initial data satisfy the canonical chi=psi^-4 ADM contract."""
from pathlib import Path
import argparse
import json
import subprocess

import h5py
import numpy as np

from scalar_ems_coordinates import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pangu',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args();root=args.root.resolve();work=args.workdir.resolve()
    work.mkdir(parents=True,exist_ok=False);report={}
    for problem,options,bounds in [
        ('nr_schwarzschild','mass = 1',(2,4)),
        ('nr_boosted_puncture','punc_ADM_mass = 1\npunc_velocity_x1 = 0.2',(-2,2)),
        ('nr_puncture','puncture_count = 1\npuncture_0_mass = 1\npuncture_spectral_points = 12',(-2,2)),
        ('nr_polarized_gowdy','gowdy_t0 = 1',(0,1))]:
        text=f'''<parthenon/job>
problem_id = {problem}
<pangu>
strict_parameters = true
compatibility_mode = native
<geometry>
background = minkowski
expected_metric = z4c
expected_mode = sync
<parthenon/mesh>
refinement = none
numlevel = 1
nghost = 4
'''
        for axis in [1,2,3]:
            text+=f'nx{axis} = 8\nx{axis}min = {bounds[0]}\nx{axis}max = {bounds[1]}\nix{axis}_bc = outflow\nox{axis}_bc = outflow\n'
        text+='''<parthenon/meshblock>
nx1 = 8
nx2 = 8
nx3 = 8
<parthenon/time>
integrator = rk4
nlim = 0
tlim = 0.1
<numerical_relativity>
matter_source = vacuum
rk4_variant = classical
finite_difference_order = 4
<problem>
'''+options+'''
<parthenon/output1>
file_type = hdf5
dt = 1
variables = nr.z4c,nr.adm
single_precision_output = false
ghost_zones = false
'''
        source=work/(problem+'.in');source.write_text(text)
        path=run(args.pangu.resolve(),root,work/problem,
                 ['numerical_relativity/chi_psi_power=-4'],deck=source)
        with h5py.File(path) as f:
            adm=f['nr.adm'][...];z=f['nr.z4c'][...]
            assert float(f['Info'].attrs['Time'])==0
        assert np.isfinite(adm).all() and np.isfinite(z).all()
        chi=z[:,0];g=z[:,1:7];a=z[:,8:14];trace_k=z[:,7]+2*z[:,17]
        assert chi.min()>0
        metric=g/chi[:,None]
        extrinsic=(a+g*trace_k[:,None]/3)/chi[:,None]
        np.testing.assert_allclose(adm[:,:6],metric,rtol=5.e-12,atol=5.e-12)
        np.testing.assert_allclose(adm[:,6:12],extrinsic,rtol=5.e-12,atol=5.e-12)
        determinant=g[:,0]*(g[:,3]*g[:,5]-g[:,4]**2)-g[:,1]*(g[:,1]*g[:,5]-g[:,2]*g[:,4])+g[:,2]*(g[:,1]*g[:,4]-g[:,2]*g[:,3])
        np.testing.assert_allclose(determinant,1,rtol=5.e-12,atol=5.e-12)
        report[problem]=dict(conformal_power=-4,
            maximum_ADM_metric_difference=float(np.max(np.abs(adm[:,:6]-metric))),
            maximum_ADM_extrinsic_difference=float(np.max(np.abs(adm[:,6:12]-extrinsic))),
            maximum_conformal_determinant_error=float(np.max(np.abs(determinant-1))))
    rejected=work/'unsupported_power';rejected.mkdir()
    with (rejected/'run.log').open('wb') as log:
        result=subprocess.run(['mpirun','-np','1',str(args.pangu.resolve()),
            '-i',str(work/'nr_schwarzschild.in'),'numerical_relativity/chi_psi_power=-3'],
            cwd=rejected,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode!=0
    assert 'chi_psi_power must be -4' in (rejected/'run.log').read_text()
    report.update(status='passed',unsupported_conformal_power='rejected',
        scope='Canonical psi^-4 native initial ADM contract and parameter rejection; not evolution acceptance')
    (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
