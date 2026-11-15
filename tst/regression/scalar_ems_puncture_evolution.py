"""Short coupled-puncture Hamiltonian/momentum convergence at matched time."""
from pathlib import Path
import argparse
import json

from scalar_ems_coordinates import run
from scalar_ems_puncture_constraints import inspect

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pangu',type=Path,required=True)
    parser.add_argument('--initial-run',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args();work=args.workdir.resolve();work.mkdir(parents=True,exist_ok=False)
    initial=args.initial_run.resolve();previous=json.loads((initial/'RESULT.json').read_text())
    assert previous['status']=='initial_constraints_and_short_evolution_passed'
    report=dict(resolutions={'96':previous['evolved']},initial_run=str(initial))
    for n in [64,80]:
        path=run(args.pangu.resolve(),args.root.resolve(),work/f'n{n}',[
            'parthenon/time/tlim=0.1','parthenon/time/nlim=-1',
            'numerical_relativity/horizon_enabled=false','numerical_relativity/horizon_count=0'],
            checkpoint=initial/f'constrained_puncture_{n}.rhdf')
        item=inspect(path);assert item['time']==.1 and abs(item['charge']-.9)<1.e-6
        report['resolutions'][str(n)]=item
    report['observed_orders']={}
    for field in ['hamiltonian','momentum']:
        errors=[report['resolutions'][str(n)][field]['L2'] for n in [64,80,96]]
        orders=[float(np.log(errors[i]/errors[i+1])/np.log([80/64,96/80][i])) for i in [0,1]]
        report['observed_orders'][field]=orders
        (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
        assert min(orders)>3.2,('Evolved coupled constraints did not converge',field,errors,orders)
    report.update(status='short_coupled_constraint_convergence_passed',
        scope='Matched0.1M 3D coupled scalar-puncture constraints in fixed0.3<r<=0.6 coordinate shell; not long-time scalar hair or boundary/AMR acceptance')
    (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
