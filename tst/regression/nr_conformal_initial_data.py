"""Check physical initial data are invariant under chi=psi^-4 or W=psi^-2."""
from pathlib import Path
import argparse
import json

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
        source=work/(problem+'.in');source.write_text(text);values=[]
        for power in [-4,-2]:
            path=run(args.pangu.resolve(),root,work/(problem+'_'+str(power)),
                     ['numerical_relativity/chi_psi_power='+str(power)],deck=source)
            with h5py.File(path) as f:
                values.append({name:f[name][...] for name in ['nr.adm','nr.z4c']})
                assert float(f['Info'].attrs['Time'])==0
        for item in values:
            assert all(np.isfinite(array).all() for array in item.values())
        np.testing.assert_allclose(values[0]['nr.adm'],values[1]['nr.adm'],rtol=5.e-12,atol=5.e-12)
        np.testing.assert_allclose(values[0]['nr.z4c'][:,1:],values[1]['nr.z4c'][:,1:],rtol=5.e-12,atol=5.e-12)
        np.testing.assert_allclose(np.sqrt(values[0]['nr.z4c'][:,0]),values[1]['nr.z4c'][:,0],rtol=5.e-13,atol=5.e-13)
        report[problem]=dict(maximum_ADM_absolute_difference=float(np.max(np.abs(values[0]['nr.adm']-values[1]['nr.adm']))))
    report.update(status='passed',scope='Initial physical ADM and common conformal fields only; not evolution acceptance')
    (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
