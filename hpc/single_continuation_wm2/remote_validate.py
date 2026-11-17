"""Read-only closed-snapshot validation; executed on the data node."""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import time

import h5py
import numpy as np

ROOT=Path('/lustre/home/2601110206/CODEX_WORKPLACE')
RUNS=[
    dict(group='single',case=case,job=job,
         directory='SinglePuncture/20261030-spin-accuracy-rerun/runs/'+case+'/'+job+'/attempt-0/pangu/output',
         log='SinglePuncture/20261030-spin-accuracy-rerun/runs/'+case+'/'+job+'/attempt-0/pangu/run.log',
         cadence=25.,target=500.)
    for case,job in [('d200_spinning_n96','17313451'),('d200_stationary_n96','17313453'),
                     ('d200_spinning_n128','17313452'),('d200_stationary_n192','17260150'),
                     ('d200_spinning_n192','17260147'),('d200_stationary_n128','17260149')]
]+[dict(group='bns',case='bns_n96_romio',job='17254476',
        directory='Matter/20261112-bns-dt5/runs/n96/17254476/formal',
        log='Matter/20261112-bns-dt5/runs/n96/17254476/formal/run.log',cadence=5.,target=200.)]


RUNS=[r for r in RUNS if r['group']=='single']

def progress(run):
    p=ROOT/run['log']
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-131072));text=f.read().decode(errors='replace')
    matches=re.findall(r'\btime=\s*([0-9.eE+-]+)',text)
    assert matches,'No advancing run log: '+str(p)
    return float(matches[-1])


def discover():
    entries=[];deferred=[]
    for run in RUNS:
        if not (ROOT/run['directory']).is_dir() or not (ROOT/run['log']).is_file():continue
        latest=progress(run)
        for path in sorted((ROOT/run['directory']).iterdir()):
            match=re.search(r'\.(\d{5})\.(phdf|rhdf)$',path.name)
            if not match:continue
            before=path.stat()
            if path.is_symlink() or before.st_uid!=71767:continue
            try:
                with h5py.File(path,'r') as f:stamp=float(f['Info'].attrs['Time'])
                assert math.isfinite(stamp) and 0<=stamp<=run['target']+.01
                assert latest>stamp+.01,'Writer has not advanced beyond this output'
                assert time.time()-before.st_mtime>60,'Output is too recent'
                if match.group(2)=='phdf':
                    assert abs(stamp-int(match.group(1))*run['cadence'])<=.01,'Unexpected field cadence'
                after=path.stat()
                assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                category='SinglePuncture' if run['group']=='single' else 'Matter'
                rel=str(path.relative_to(ROOT/category))
                entries.append(dict(group=run['group'],case=run['case'],job=run['job'],path=rel,
                    absolute_source=str(path),bytes=before.st_size,allocated_bytes=before.st_blocks*512,
                    mtime_ns=before.st_mtime_ns,time=stamp,latest_logged_time=latest,
                    kind='field' if match.group(2)=='phdf' else 'checkpoint',remove_after_verified=False,
                    closure_basis='Numbered immutable output; synchronous writer advanced past its time; stat stable and older than60s',
                    target_time=run['target'],run_complete=False))
            except Exception as error:deferred.append(dict(path=str(path),reason=repr(error)))
    return dict(entries=entries,deferred=deferred,remote_read_only=True,source_originals_deleted=False)


def validate(entry):
    path=Path(entry['absolute_source']);before=path.stat()
    assert not path.is_symlink() and before.st_uid==71767
    assert (before.st_size,before.st_mtime_ns)==(entry['bytes'],entry['mtime_ns'])
    result=dict(status='validating',path=str(path),time=entry['time'],datasets={},
                scope='Closed snapshot structure, physical-cell finiteness, geometry and cadence; not completed-run physical acceptance')
    with h5py.File(path,'r') as f:
        assert float(f['Info'].attrs['Time'])==entry['time']
        raw=f['Input'].attrs.get('File','')
        if isinstance(raw,bytes):raw=raw.decode()
        power=re.search(r'^\s*chi_psi_power\s*=\s*([-+0-9.eE]+)',str(raw),re.M)
        if power:assert float(power.group(1))==-4,'Unexpected evolved conformal representation'
        result['declared_chi_psi_power']=float(power.group(1)) if power else None
        ghost=int(f['Info'].attrs.get('NGhost',0)) if int(f['Info'].attrs.get('IncludesGhost',0)) else 0
        crop=(slice(ghost,-ghost),)*3 if ghost else (slice(None),)*3
        names=[]
        f.visititems(lambda name,obj:names.append(name) if isinstance(obj,h5py.Dataset) and np.issubdtype(obj.dtype,np.floating) else None)
        assert 'nr.z4c' in f
        result['checked_metric_representation']='physical_ADM' if 'nr.adm' in f else 'conformal_metric_with_positive_chi'
        if entry['kind']=='field':
            assert all(k in f for k in ['nr.adm','nr.constraints','nr.constraint_mask'])
        if entry['group']=='bns':assert 'mhd.cons' in f
        for name in names:
            dataset=f[name];count=0
            if not dataset.shape:
                assert np.isfinite(dataset[()]);count=1
            else:
                slab=max(1,33554432//max(1,int(np.prod(dataset.shape[1:]))*dataset.dtype.itemsize))
                for start in range(0,dataset.shape[0],slab):
                    values=dataset[start:start+slab]
                    if values.ndim>=4:values=values[(Ellipsis,)+crop]
                    assert np.isfinite(values).all(),'Nonfinite physical values in '+name
                    count+=int(values.size)
            result['datasets'][name]=dict(shape=list(dataset.shape),dtype=str(dataset.dtype),finite_physical_values=count)
        minimum_chi=np.inf;minimum_det=np.inf;mask_count=0;h2=0.;momentum2=0.;rho_min=np.inf
        for b in range(f['nr.z4c'].shape[0]):
            state=f['nr.z4c'][b][(Ellipsis,)+crop];minimum_chi=min(minimum_chi,float(state[0].min()))
            assert state[0].min()>0,'Nonpositive chi in physical cells'
            metric=f['nr.adm'][b][(Ellipsis,)+crop][:6] if 'nr.adm' in f else state[1:7]
            determinant=metric[0]*(metric[3]*metric[5]-metric[4]**2)-metric[1]*(metric[1]*metric[5]-metric[2]*metric[4])+metric[2]*(metric[1]*metric[4]-metric[2]*metric[3])
            assert np.isfinite(determinant).all() and metric[0].min()>0 and (metric[0]*metric[3]-metric[1]**2).min()>0 and determinant.min()>0,'Spatial metric is not positive definite'
            minimum_det=min(minimum_det,float(determinant.min()))
            if 'nr.constraints' in f:
                mask=f['nr.constraint_mask'][b,0][crop].astype(bool)
                c=f['nr.constraints'][b][(Ellipsis,)+crop]
                mask_count+=int(mask.sum());h2+=float(np.sum(c[1][mask]**2));momentum2+=float(np.sum(c[2][mask]**2))
            if 'mhd.prim' in f:
                rho=f['mhd.prim'][b,0][crop];rho_min=min(rho_min,float(rho.min()))
                assert rho.min()>=0,'Negative primitive density'
        result.update(minimum_chi=minimum_chi,minimum_spatial_determinant=minimum_det,
            constraint_mask_cells=mask_count,hamiltonian_L2=math.sqrt(h2/mask_count) if mask_count else None,
            momentum_L2=math.sqrt(momentum2/mask_count) if mask_count else None,
            minimum_primitive_density=rho_min if math.isfinite(rho_min) else None)
        if entry['kind']=='field':assert mask_count>0,'No active constraint diagnostic cells'
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''):digest.update(chunk)
    after=path.stat();assert (after.st_size,after.st_mtime_ns)==(before.st_size,before.st_mtime_ns)
    result.update(status='snapshot_checks_passed',source_sha256=digest.hexdigest(),bytes=after.st_size,mtime_ns=after.st_mtime_ns,
                  checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),ghost_cells_excluded=bool(ghost))
    return result
