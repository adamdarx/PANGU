#!/usr/bin/env python3
"""Compare continuous and checkpointed coupled RK4 at curved AMR interfaces."""
from pathlib import Path
import argparse
import hashlib
import json
import resource
import subprocess
import time

import h5py
import numpy as np

from compact_object_plasma import edit, inspect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pangu',type=Path,required=True)
    p.add_argument('--workdir',type=Path,required=True)
    a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    binary=a.pangu.resolve();root=a.workdir.resolve();root.mkdir(exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    text=(repo/'plugins/compact_objects/examples/binary_plasma_admission.in').read_text()
    text=text[:text.index('<parthenon/static_refinement0>')]+text[text.index('<parthenon/output1>'):]
    changes={('parthenon/time','tlim'):.2,('parthenon/time','ncycle_out'):1,
             ('parthenon/mesh','refinement'):'adaptive',('parthenon/mesh','numlevel'):3,
             ('numerical_relativity','rk4_variant'):'athenak',
             ('numerical_relativity','slow_start_lapse'):'true',
             ('numerical_relativity','tracker_enabled'):'true',
             ('numerical_relativity','tracker_count'):2,
             ('numerical_relativity','amr_method'):'tracker',
             ('problem','puncture_0_center_x'):3.257,('problem','puncture_1_center_x'):-3.257,
             ('problem','puncture_0_mass'):.483,('problem','puncture_1_mass'):.483,
             ('problem','puncture_0_momentum_x'):0,('problem','puncture_1_momentum_x'):0,
             ('problem','puncture_0_momentum_y'):-.133,('problem','puncture_1_momentum_y'):.133,
             ('parthenon/output1','dt'):.1,('parthenon/output2','dt'):.1,
             ('parthenon/output1','single_precision_output'):'false'}
    for i,c in [(0,3.257),(1,-3.257)]:
        for key,value in [('center_x1',c),('center_x2',0),('center_x3',0),('mass',.505),('refinement_radius',.5),('refinement_level',2)]:
            changes['numerical_relativity',f'tracker_{i}_{key}']=value
    for axis in [1,2,3]:
        changes['parthenon/mesh',f'nx{axis}']=16
        changes['parthenon/meshblock',f'nx{axis}']=8
        changes['parthenon/mesh',f'x{axis}min']=-8
        changes['parthenon/mesh',f'x{axis}max']=8
    text=edit(text,changes)
    for index,level,center,width in [(0,1,0,4),(1,2,3.257,.5),(2,2,-3.257,.5)]:
        text+=f'\n<parthenon/static_refinement{index}>\nlevel = {level}\n'
        for axis in [1,2,3]:
            c=center if axis==1 else 0
            text+=f'x{axis}min = {c-width}\nx{axis}max = {c+width}\n'
    deck=root/'input.in';deck.write_text(text)
    result=dict(status='running',binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                acceptance_max_scaled_primary_difference=5e-10,
                scope='Small whole-binary curved-spacetime AMR regression, not production or boundary-error acceptance.')
    def save():(root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for name in ['continuous','restart']:
            folder=root/name;folder.mkdir();command=[str(binary),'-i',str(deck)]
            if name=='restart':
                candidates=[x for x in (root/'continuous').glob('*.rhdf') if '.final.' not in x.name and '.00000.' not in x.name]
                assert len(candidates)==1,candidates
                command+=['-r',str(candidates[0])]
            start=time.monotonic()
            with (folder/'run.log').open('w') as f:rc=subprocess.run(command,cwd=folder,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
            assert rc==0 and 'Driver completed.' in (folder/'run.log').read_text(),folder
            result[name]=dict(wall_seconds=time.monotonic()-start,final=inspect(next(folder.glob('*.final.phdf'))))
            save()
        left=next((root/'continuous').glob('*.final.phdf'));right=next((root/'restart').glob('*.final.phdf'))
        delta={}
        with h5py.File(left) as x,h5py.File(right) as y:
            assert np.array_equal(x['LogicalLocations'][:],y['LogicalLocations'][:])
            assert np.array_equal(x['Levels'][:],y['Levels'][:])
            for name in ['nr.z4c','mhd.cons','mhd.b_face']:
                a=x[name][:];b=y[name][:];delta[name]=float(abs(a-b).max()/max(1.,float(abs(a).max())))
        result['restart_difference']=delta;save()
        assert max(delta.values())<5e-10,delta
        result['status']='passed'
    except BaseException as e:result.update(status='failed',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()
