"""Allocated-only mesh, memory, timestep and exterior-finite admission."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import os
import re
import statistics
import subprocess
import time

import h5py
import numpy as np

ROOT=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-bbh-convergence')
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
staged=json.loads((ROOT/'STAGED.json').read_text());exe=Path(staged['binary'])
assert hashlib.sha256(exe.read_bytes()).hexdigest()==staged['binary_sha256']
plan=json.loads((ROOT/'candidates/PLAN.json').read_text())
job=os.environ['SLURM_JOB_ID'];work=ROOT/'admission'/job;work.mkdir(parents=True,exist_ok=False)
report=dict(status='resource_admission_running',job_id=job,node=os.uname().nodename,
            binary_sha256=staged['binary_sha256'],source_plan_sha256=staged['plan_sha256'],
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=[],physical_acceptance=False)
receipt=ROOT/('ADMISSION-'+job+'.json')
def save():
    temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(receipt)
def command(arguments):
    return ['mpirun','--mca','io','romio341','-np','4','--map-by','ppr:4:node','--bind-to','none',
            'bash',str(ROOT/'rank.sh'),str(exe),*arguments]
def launch(cmd,folder,name,timeout):
    (folder/(name+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
    start=time.monotonic()
    with (folder/(name+'.log')).open('w') as stream:
        result=subprocess.run(cmd,cwd=folder,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    text=(folder/(name+'.log')).read_text(errors='replace')
    assert result.returncode==0,(name,result.returncode,text[-3000:])
    return text,time.monotonic()-start
save()
try:
    for case in plan['cases']:
        deck=ROOT/'candidates'/case['input']
        assert hashlib.sha256(deck.read_bytes()).hexdigest()==case['input_sha256']
        folder=work/('n'+str(case['n']));folder.mkdir()
        item=dict(n=case['n'],block=case['block'],status='running',case_folder=str(folder));report['cases'].append(item);save()
        quota=subprocess.check_output(['lfs','quota','-u','2601110206','/lustre'],text=True)
        match=re.search(r'/lustre\s+(\d+)\s+\d+\s+(\d+)',quota);assert match,quota
        free=(int(match.group(2))-int(match.group(1)))/1024**2
        assert free>150,('Protect active production quota',free)
        mesh,_=launch(command(['-i',str(deck),'-m','4']),folder,'mesh',90)
        blocks=int(re.search(r'Total number of MeshBlocks = (\d+)',mesh).group(1))
        level=int(re.search(r'Number of physical refinement levels = (\d+)',mesh).group(1))
        assert level==case['requested_finest_level']
        item.update(static_mesh_blocks=blocks,max_physical_level=level,initial_quota_free_GiB=free)
        gpu_file=(folder/'gpu-memory.csv').open('w')
        monitor=subprocess.Popen(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
            '--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits','--loop=1'],stdout=gpu_file,stderr=subprocess.STDOUT)
        try:
            overrides=['parthenon/time/nlim=32','parthenon/time/ncycle_out_mesh=1',
                'parthenon/output2/dt=1e30','parthenon/output2/variables=nr.constraints,nr.constraint_mask',
                'parthenon/output2/single_precision_output=true',
                'parthenon/output3/dt=-1','parthenon/output3/dn=-1']
            text,wall=launch(command(['-i',str(deck),*overrides]),folder,'evolution',480)
        finally:
            monitor.terminate();monitor.wait(timeout=10);gpu_file.close()
        assert 'Driver completed.' in text
        cycles=re.findall(r'\bcycle=\s*(\d+)\s+time=\s*([\deE+.\-]+)\s+dt=\s*([\deE+.\-]+)',text)
        assert cycles and int(cycles[-1][0])==32,(cycles[-3:],text[-1500:])
        # Match the standalone seconds field. The rate field ends in the same suffix.
        step_times=[float(v) for v in re.findall(r'(?<!\S)wsec_step=\s*([\deE+.\-]+)',text)]
        assert len(step_times)>=16
        fields=[]
        for path in sorted(folder.glob('*.phdf')):
            exterior=0
            with h5py.File(path) as f:
                for b in range(len(f['Levels'])):
                    x,y,z=[f['VolumeLocations/'+a][b] for a in 'xyz']
                    r2=x[None,None,:]**2+y[None,:,None]**2+z[:,None,None]**2
                    mask=(r2>64)&(r2<900)&(f['nr.constraint_mask'][b,0]>0)
                    if mask.any():assert np.isfinite(f['nr.constraints'][b,:][:,mask]).all()
                    exterior+=int(mask.sum())
                fields.append(dict(path=path.name,bytes=path.stat().st_size,
                    time=float(f['Info'].attrs['Time']),cycle=int(f['Info'].attrs['NCycle']),
                    blocks=int(f['Info'].attrs['NumMeshBlocks']),exterior_finite_cells=exterior))
        assert fields and all(f['exterior_finite_cells']>0 for f in fields)
        assert max(f['cycle'] for f in fields)==32
        memory={}
        with (folder/'gpu-memory.csv').open() as stream:
            for row in csv.reader(stream):
                if len(row)!=5:continue
                index=row[1].strip();used,total=float(row[2]),float(row[3])
                value=memory.setdefault(index,dict(total_MiB=total,peak_sampled_MiB=0))
                value['peak_sampled_MiB']=max(value['peak_sampled_MiB'],used)
        assert len(memory)==4,memory
        comfortable=all(v['peak_sampled_MiB']<.80*v['total_MiB'] and v['total_MiB']-v['peak_sampled_MiB']>8192 for v in memory.values())
        dt=float(cycles[-1][2]);median=statistics.median(step_times[-16:]);mean=statistics.mean(step_times[-16:])
        item.update(status='resource_admission_passed' if comfortable else 'resource_reserve_insufficient',
             evolution_cycles=32,final_time=float(cycles[-1][1]),dt=dt,wall_seconds=wall,
             median_last16_step_seconds=median,mean_last16_step_seconds=mean,
             rough300M_median_step_hours=median*300/dt/3600,
             rough300M_mean_step_hours=mean*300/dt/3600,gpu_memory=memory,fields=fields,
             memory_measurement='1-second samples; not allocator high-water. Constraints-only field I/O; full production output buffers and later AMR need an additional reserve.',
             timing_scope='Early32 steps only; initial, AMR, diagnostic cadence and future IO/merger costs need separate accounting.')
        save()
        if not comfortable:raise RuntimeError('Measured memory reserve insufficient; do not submit formal evolution')
    report['status']='resource_admission_passed_formal_not_submitted'
except Exception as error:
    report.update(status='resource_admission_failed',error=repr(error));save();raise
report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(json.dumps(report,indent=2))
