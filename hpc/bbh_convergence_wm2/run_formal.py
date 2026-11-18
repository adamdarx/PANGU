"""GPU-only full-output admission and queued quasi-circular production."""
from pathlib import Path
import os,json,hashlib,subprocess,time,re,signal,sys
import numpy as np,h5py
R=Path(__file__).resolve().parent;mode=sys.argv[1];ns=[32,48,64] if mode=='io' else [int(sys.argv[2])]
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
staged=json.loads((R/'STAGED.json').read_text());exe=Path(staged['binary']);assert hashlib.sha256(exe.read_bytes()).hexdigest()==staged['binary_sha256']
plan=json.loads((R/'candidates/PLAN.json').read_text());job=os.environ['SLURM_JOB_ID'];receipt=R/('FULL_IO_ADMISSION.json' if mode=='io' else 'FORMAL_N%d.json'%ns[0]);report=dict(status='running',job_id=job,mode=mode,cases=[],physical_acceptance=False)
def save():
 t=receipt.with_suffix('.'+job+'.tmp');t.write_text(json.dumps(report,indent=2));t.replace(receipt)
def free():
 v=subprocess.check_output(['lfs','quota','-u','2601110206','/lustre']).decode();m=re.search(r'/lustre\s+(\d+)\s+\d+\s+(\d+)',v);assert m;return (int(m[2])-int(m[1]))/1024**2
save()
try:
 gate=json.loads((R/'ADMISSION-17369148.json').read_text());assert gate['status']=='resource_admission_passed_formal_not_submitted'
 if mode=='formal':assert json.loads((R/'FULL_IO_ADMISSION.json').read_text())['status']=='passed'
 for n in ns:
  assert free()>200,'Protect existing production quota'
  case=next(c for c in plan['cases'] if c['n']==n);deck=R/'candidates'/case['input'];assert hashlib.sha256(deck.read_bytes()).hexdigest()==case['input_sha256']
  if mode=='formal':
   measured=next(c for c in gate['cases'] if c['n']==n);assert measured['rough300M_mean_step_hours']<110,'Measured throughput exceeds five-day budget'
  work=R/'runs'/('n%d'%n)/job;work.mkdir(parents=True,exist_ok=False)
  cmd=['mpirun','-np','4','--map-by','ppr:4:node','--bind-to','none','bash',str(R/'rank.sh'),str(exe),'-i',str(deck)]
  cmd+=['parthenon/output2/hdf5_compression_level=0','parthenon/output3/hdf5_compression_level=0']
  if mode=='io':cmd+=['parthenon/time/nlim=4','parthenon/time/ncycle_out=1']
  (work/'command.json').write_text(json.dumps(cmd));start=time.monotonic();last=start;seen=False
  with (work/'run.log').open('w') as log:
   p=subprocess.Popen(cmd,cwd=str(work),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    while p.poll() is None:
     apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=process_name','--format=csv,noheader']).decode();seen=seen or exe.name in apps
     if time.monotonic()-last>60:assert free()>150,'Protect production quota';last=time.monotonic()
     if mode=='io' and time.monotonic()-start>1800:raise TimeoutError('Full IO probe limit')
     time.sleep(2)
   except BaseException:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
  assert p.returncode==0 and seen
  path=next(work.glob('*.final.phdf'))
  with h5py.File(path) as f:
   t=float(f['Info'].attrs['Time']);cy=int(f['Info'].attrs['NCycle']);assert cy==4 if mode=='io' else abs(t-case['target'])<1e-9
   for b in range(len(f['Levels'])):
    x,y,z=[f['VolumeLocations/'+a][b] for a in 'xyz'];rr=x[None,None,:]**2+y[None,:,None]**2+z[:,None,None]**2;mask=(rr>64)&(rr<900)&(f['nr.constraint_mask'][b,0]>0)
    assert np.isfinite(f['nr.constraints'][b,:][:,mask]).all()
  report['cases'].append(dict(n=n,time=t,cycle=cy,returncode=p.returncode,actual_GPU_observed=seen,work=str(work),wall_seconds=time.monotonic()-start));save()
 report['status']='passed' if mode=='io' else 'completed300M_analysis_pending';save()
except BaseException as e:report.update(status='failed',error=repr(e));save();raise
