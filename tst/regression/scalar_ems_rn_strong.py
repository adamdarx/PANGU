"""Matched-time chi=psi^-4 charged-puncture constraint convergence.

Reuse an independently accepted N96 RN hold, then run fresh N64/N80 data with
the same numerical parameters. Coarse grids omit AH searches; this does not
replace the strict N96 horizon checks or establish long-time acceptance.
"""
from pathlib import Path
import argparse
import json

import h5py
import numpy as np

from analyze_ems_rn_hold import deck
from scalar_ems_coordinates import run
from scalar_ems_puncture_constraints import inspect
from scalar_ems_rn import norms


def measure(path):
    result=inspect(path);electric=[]
    with h5py.File(path) as f:
        for b in range(f['ems.gauss'].shape[0]):
            mask=f['nr.constraint_mask'][b,0].astype(bool)
            electric.extend(f['ems.gauss'][b,0][mask])
    result['electric_gauss']=norms(electric)
    assert result['phi_max']==0 and abs(result['charge']-.9)<1.e-6
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pangu',type=Path,required=True)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args();reference=args.reference_run.resolve()
    previous=json.loads((reference/'RN_HOLD_ANALYSIS.json').read_text())
    assert previous['status']=='short_RN_invariant_checks_passed'
    assert previous['chi_psi_power']==-4 and not previous['geometric_floor_enabled']
    parameters=deck(reference/'input.in')
    assert int(parameters['parthenon/mesh']['nx1'])==96
    work=args.workdir.resolve();work.mkdir(parents=True,exist_ok=False)
    report=dict(chi_psi_power=-4,reference_run=str(reference),resolutions={},
        minimum_observed_order=3.2,observed_orders={},
        scope='Uniform N64/N80/N96 constraints at matched time0.5M in0.3<r<=0.6; coarse AH, long-time and boundary-error acceptance excluded')
    for n in [96,64,80]:
        if n==96:
            folder=reference;path=next(folder.glob('*.final.phdf'))
        else:
            folder=work/('n'+str(n));block=32 if n%32==0 else 16
            path=run(args.pangu.resolve(),args.root.resolve(),folder,[
                'numerical_relativity/horizon_enabled=false','numerical_relativity/horizon_count=0',
                'parthenon/time/nlim=-1',
                *[f'parthenon/mesh/nx{a}={n}' for a in [1,2,3]],
                *[f'parthenon/meshblock/nx{a}={block}' for a in [1,2,3]]],deck=reference/'input.in')
        initial=next(folder.glob('*.00000.phdf'))
        report['resolutions'][str(n)]=dict(initial=measure(initial),final=measure(path))
        assert report['resolutions'][str(n)]['initial']['time']==0
        assert report['resolutions'][str(n)]['final']['time']==.5
        (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    for phase,fields in [('initial',['hamiltonian','electric_gauss']),
                         ('final',['hamiltonian','momentum','electric_gauss'])]:
        for field in fields:
            errors=[report['resolutions'][str(n)][phase][field]['L2'] for n in [64,80,96]]
            assert all(np.isfinite(value) and value>0 for value in errors)
            orders=[float(np.log(errors[i]/errors[i+1])/np.log([80/64,96/80][i])) for i in [0,1]]
            report['observed_orders'][phase+'_'+field]=orders
    (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    assert min(min(row) for row in report['observed_orders'].values())>report['minimum_observed_order'],report['observed_orders']
    report['status']='short_chi4_RN_constraint_convergence_passed'
    (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
