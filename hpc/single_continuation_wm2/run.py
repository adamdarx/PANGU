"""Frozen-binary GPU checkpoint continuation; never modifies the original job."""
from pathlib import Path
import datetime,hashlib,json,os,re,signal,subprocess,time
import h5py,numpy as np
from remote_validate import validate
R=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-single-continuation')
OLD=R.parents[1]/'SinglePuncture/20261030-spin-accuracy-rerun'
SHA='0215d235a2ff3daca0d433b97da08a04f3db4953f8d425683e59259972a32500'
CASES={'stationary':('17260150','26a38fc495bfde57ba1e2a4ef208e3c711870fa4d99782694f989d7340b9ae90'),'spinning':('17260147','c3c2066cc9ff54cd0c910990e416fa773cb27c7f775d947ae06c61ca21bcf4c1')}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save(p,data):
 t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');t.replace(p)
def origin(kind):return OLD/'runs'/('d200_'+kind+'_n192')/CASES[kind][0]/'attempt-0/pangu'
def pin(kind):
 source=origin(kind);tail=(source/'run.log').read_bytes()[-131072:].decode(errors='replace');latest=float(re.findall(r'\btime=\s*([0-9.eE+-]+)',tail)[-1]);candidates=[]
 for p in (source/'output').glob('*.rhdf'):
  if not re.search(r'\.\d{5}\.rhdf$',p.name):continue
  s=p.stat()
  if s.st_uid!=71767 or p.is_symlink() or time.time()-s.st_mtime<120:continue
  with h5py.File(p) as f:t=float(f['Info'].attrs['Time']);cycle=int(f['Info'].attrs['NCycle'])
  if latest>t+.01:candidates.append((t,cycle,p,s))
 assert candidates,'No closed checkpoint';t,cycle,p,s=max(candidates,key=lambda v:v[0]);folder=R/'pinned'/kind;folder.mkdir(parents=True,exist_ok=True);target=folder/p.name
 if not target.exists():os.link(str(p),str(target))
 q=target.stat();assert (q.st_ino,q.st_size,q.st_mtime_ns)==(s.st_ino,s.st_size,s.st_mtime_ns)
 return dict(group='single',kind='checkpoint',absolute_source=str(target),original_source=str(p),bytes=q.st_size,mtime_ns=q.st_mtime_ns,time=t,cycle=cycle,original_job=CASES[kind][0])
def free():
 s=subprocess.check_output(['lfs','quota','-u','2601110206','/lustre']).decode();m=re.search(r'/lustre\s+(\d+)\s+\d+\s+(\d+)',s);assert m;return (int(m[2])-int(m[1]))/1024**2

def field(path):
 with h5py.File(path) as f:
  a=f['Info'].attrs;assert 'nr.z4c' in f
  for b in range(len(f['Levels'])):
   assert np.isfinite(f['nr.z4c'][b]).all()
   if 'nr.constraints' in f:
    mask=f['nr.constraint_mask'][b,0]>0
    if mask.any():assert np.isfinite(f['nr.constraints'][b][:,mask]).all(),'Nonfinite exterior constraints'
  return dict(path=str(path),time=float(a['Time']),cycle=int(a['NCycle']),blocks=len(f['Levels']))
def compare(checkpoint,output):
 def key(f):return [tuple(np.ravel(l).tolist()+np.ravel(x).tolist()) for l,x in zip(f['Levels'][:],f['LogicalLocations'][:])]
 with h5py.File(checkpoint) as a,h5py.File(output) as b:
  ka,kb=key(a),key(b);index={k:i for i,k in enumerate(ka)};assert len(index)==len(ka) and set(ka)==set(kb)
  ghost=int(a['Info'].attrs.get('NGhost',0)) if int(a['Info'].attrs.get('IncludesGhost',0)) else 0
  crop=(slice(ghost,-ghost),)*3 if ghost else (slice(None),)*3;error=0.
  for j,k in enumerate(kb):
   x=a['nr.z4c'][index[k]][(Ellipsis,)+crop];y=b['nr.z4c'][j];assert x.shape==y.shape
   error=max(error,float(np.max(np.abs(x-y)/(1+np.abs(x)))))
  assert error<1e-11,'Restart changed evolved state: '+str(error)
  return error

def main(mode,kind):
 assert mode in ('admission','formal') and kind in CASES
 assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES'),'GPU Slurm execution required'
 exe=R/'bin/pangu';deck=R/'inputs'/(kind+'.in');assert digest(exe)==SHA and digest(deck)==CASES[kind][1]
 work=R/'runs'/kind/os.environ['SLURM_JOB_ID'];work.mkdir(parents=True,exist_ok=False)
 report=dict(status=mode+'_running',kind=kind,job_id=os.environ['SLURM_JOB_ID'],original_job=CASES[kind][0],binary_sha256=SHA,input_sha256=CASES[kind][1],target=500,work=str(work),physical_acceptance=False,started_utc=datetime.datetime.utcnow().isoformat()+'Z')
 receipt=R/(mode.upper()+'_'+kind+'.json')
 def write():save(receipt,report);save(work/'STATUS.json',report)
 write()
 try:
  assert free()>200,'Protect existing production quota'
  if mode=='formal':
   gate=json.loads((R/('ADMISSION_'+kind+'.json')).read_text());assert gate['status']=='admission_passed' and gate['binary_sha256']==SHA and gate['kind']==kind and gate['input_sha256']==CASES[kind][1]
   existing=origin(kind)/'phase-validated.json'
   if existing.exists():
    proof=json.loads(existing.read_text())
    if proof.get('status')=='completed' and proof.get('returncode')==0 and float(proof.get('last_logged_time',-1))>=500-1e-9:
     report.update(status='skipped_original_completed500M',original_completion=proof);write();return
   rows=subprocess.check_output(['sacct','-X','-n','-P','-j',CASES[kind][0],'-o','JobIDRaw,State,ExitCode']).decode().splitlines();states=[row.split('|')[1].split()[0] for row in rows if row.split('|')[0]==CASES[kind][0]]
   assert states and states[0] in ('TIMEOUT','NODE_FAIL','PREEMPTED'),'Original not eligible for automatic continuation: '+str(states)
   report['original_terminal_state']=states[0]
   entry=pin(kind)
  else:entry=json.loads((R/'pinned'/(kind+'.json')).read_text())
  report['checkpoint']=entry;write();report['checkpoint_validation']=validate(entry);write()
  if mode=='formal':assert gate['checkpoint_validation']['status']=='snapshot_checks_passed'
  def run(label,nlim):
   folder=work/label;folder.mkdir();cmd=['mpirun','-np','4','--map-by','ppr:4:node','--bind-to','none','bash',str(R/'rank.sh'),str(exe),'-i',str(deck),'-r',entry['absolute_source'],'-d','output','parthenon/time/tlim=500','parthenon/time/nlim='+str(nlim),'parthenon/output2/hdf5_compression_level=0','parthenon/output3/hdf5_compression_level=0']
   if mode=='admission':cmd+=['parthenon/time/ncycle_out=1','parthenon/output2/dt=1e30','parthenon/output3/dt=1e30']
   save(folder/'command.json',cmd);start=time.monotonic();last=start;observed=False;peak=0;reserve=float('inf')
   with (folder/'run.log').open('w') as log:
    p=subprocess.Popen(cmd,cwd=str(folder),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
     while p.poll() is None:
      apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']).decode();observed|='pangu' in apps
      for row in subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits']).decode().splitlines():
       used,total=map(float,row.split(','));peak=max(peak,used);reserve=min(reserve,total-used)
      if time.monotonic()-last>60:assert free()>150,'Protect existing production quota';last=time.monotonic()
      if mode=='admission':assert time.monotonic()-start<2700,'GPU restart admission timeout'
      time.sleep(2)
    except BaseException:
     os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=20)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
     raise
   assert p.returncode==0 and observed
   files=list((folder/'output').glob('*.phdf'));assert files
   final=max((field(x) for x in files),key=lambda v:v['cycle']);result=dict(final=final,wall_seconds=time.monotonic()-start,actual_GPU_observed=observed,peak_MiB=peak,minimum_reserve_MiB=reserve)
   report[label]=result;write();return final
  if mode=='admission':
   zero=run('restart_same_cycle',entry['cycle']);assert zero['cycle']==entry['cycle'] and abs(zero['time']-entry['time'])<1e-10
   report['restart_state_scaled_max_difference']=compare(entry['absolute_source'],zero['path']);write()
   advanced=run('restart_step8',entry['cycle']+8);assert advanced['cycle']==entry['cycle']+8 and advanced['time']>entry['time']
   assert report['restart_step8']['minimum_reserve_MiB']>4096,'Less than4GiB operational reserve'
   report['status']='admission_passed'
  else:
   final=run('continuation',-1);assert abs(final['time']-500)<1e-9
   report.update(status='completed500M_numerical_checks_passed_physical_comparison_pending',segments=[dict(source=str(origin(kind)),end_time_inclusive=entry['time']),dict(source=str(work/'continuation'),start_time_exclusive=entry['time'],end_time_inclusive=500)],original_job_receipt_unchanged=True)
 except BaseException as e:report.update(status=mode+'_failed',error=repr(e));write();raise
 report['finished_utc']=datetime.datetime.utcnow().isoformat()+'Z';write();print(json.dumps(report,indent=2))
if __name__=='__main__':
 import sys
 if sys.argv[1]=='pin':
  for kind in CASES:save(R/'pinned'/(kind+'.json'),pin(kind))
 else:main(*sys.argv[1:])
