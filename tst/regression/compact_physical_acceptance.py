"""Evaluate evidence-backed physical accuracy budgets for BBH, TOV and BNS.

Missing evidence is pending, never a zero error or a passing measurement.
Budgets are project accuracy requirements, not universal literature constants.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
from scipy.optimize import brentq


def richardson(h, values):
    """Three unequal spacings; scalar or aligned trajectory L2 differences."""
    h=np.asarray(h,dtype=float);v=np.asarray(values)
    if not np.issubdtype(v.dtype,np.number):raise ValueError('Non-numeric values')
    v=v.astype(np.complex128 if np.iscomplexobj(v) else np.float64)
    if h.shape!=(3,) or not np.all(h>0) or not np.all(np.diff(h)<0):
        raise ValueError('Need three strictly decreasing effective spacings')
    if v.shape[0]!=3 or not np.isfinite(v).all():
        raise ValueError('Need three finite scalar values or aligned trajectories')
    a,b=v[0]-v[1],v[1]-v[2]
    da=float(np.linalg.norm(np.atleast_1d(a)));db=float(np.linalg.norm(np.atleast_1d(b)))
    if min(da,db)<=1e-14*max(1,float(np.max(np.abs(v)))):
        raise ValueError('Resolution differences unresolved; no order claim')
    if float(np.vdot(np.ravel(a),np.ravel(b)).real)<=0:
        raise ValueError('Nonmonotone resolution differences; no Richardson estimate')
    ratio=da/db
    def equation(p):
        return (math.expm1(p*math.log(h[0]/h[2]))-math.expm1(p*math.log(h[1]/h[2])))/math.expm1(p*math.log(h[1]/h[2]))-ratio
    if equation(.1)*equation(10)>=0:
        raise ValueError('No observed positive order in[0.1,10]')
    p=brentq(equation,.1,10)
    error=db/math.expm1(p*math.log(h[1]/h[2]))
    scale=float(np.linalg.norm(np.atleast_1d(v[2])))
    return dict(observed_order=float(p),finest_absolute_error=error,
                finest_relative_error=error/max(scale,1e-300))


def evaluate(document, protocol, base):
    case=document['case'];rules={**protocol['common'],**protocol['cases'][case]}
    metrics=document.get('metrics',{});checks={}
    if case=='bbh':
        family=document.get('family')
        if family not in ['head_on','quasicircular','aligned_spin']:
            checks['binary_family']=dict(status='pending',reason='Specify head_on, quasicircular or aligned_spin')
        if family!='head_on':
            rules['reference_phase_max_rad']=dict(max=protocol['waveform_phase_additional_requirement']['max_radians'],meaning='Non-head-on dominant mode phase, amplitude >1% peak; fixed early alignment and trusted interval.')
    for name,rule in rules.items():
        row=metrics.get(name)
        if row is None:
            checks[name]=dict(status='pending',reason='Missing measurement');continue
        value=row.get('value')
        if not isinstance(value,(int,float)) or not math.isfinite(value):
            checks[name]=dict(status='fail',reason='Nonfinite or invalid measurement');continue
        sources=row.get('evidence',[])
        if not sources:
            checks[name]=dict(status='pending',reason='No source evidence');continue
        reason=None
        for source in sources:
            path=Path(source['path']);path=path if path.is_absolute() else base/path
            if not path.is_file():reason='Missing source: '+str(path);break
            if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:
                reason='Source hash changed: '+str(path);break
        if reason:
            checks[name]=dict(status='pending',reason=reason);continue
        passed=rule.get('min',0)<=value<=rule.get('max',math.inf)
        checks[name]=dict(status='pass' if passed else 'fail',value=value,rule=rule,evidence=sources)
    failures=[n for n,x in checks.items() if x['status']=='fail']
    pending=[n for n,x in checks.items() if x['status']=='pending']
    status='failed' if failures else 'pending' if pending else 'passed'
    return dict(schema='pangu.compact-physical-acceptance.v1',case=case,status=status,
                physical_acceptance=status=='passed',protocol_id=protocol['id'],checks=checks,
                failed=failures,pending=pending,notes=document.get('notes',[]),
                reported_metrics=metrics,
                tolerance_origin=protocol['tolerance_origin'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'reference/compact_physical_acceptance_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text());document=json.loads(args.measurements.read_text())
    result=evaluate(document,protocol,args.measurements.resolve().parent)
    result['protocol_sha256']=hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(case=result['case'],status=result['status'],failed=result['failed'],pending=result['pending'])))
    return 0 if result['status']=='passed' else 2 if result['status']=='pending' else 1


if __name__=='__main__':raise SystemExit(main())
