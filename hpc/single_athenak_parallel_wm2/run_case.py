"""Independent original AthenaK input/program, GPU admission then500M."""
from pathlib import Path
import csv,datetime,hashlib,json,os,re,signal,statistics,subprocess,sys,time
R=Path(__file__).resolve().parent;mode,key=sys.argv[1:];case=next(c for c in json.loads((R/'PLAN.json').read_text())['cases'] if c['id']==key)
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
exe=Path(case['binary']);deck=Path(case['input']);assert hashlib.sha256(exe.read_bytes()).hexdigest()==case['binary_sha256'];assert hashlib.sha256(deck.read_bytes()).hexdigest()==case['input_sha256']
job=os.environ['SLURM_JOB_ID'];work=R/'runs'/key/job;work.mkdir(parents=True,exist_ok=False);receipt=R/('%s_%s.json'%(mode.upper(),key));report=dict(status=mode+'_running',job_id=job,case=key,mode=mode,work=str(work),target=500,binary_sha256=case['binary_sha256'],input_sha256=case['input_sha256'],physical_acceptance=False)
def save():
 for p in [receipt,work/'STATUS.json']:
  q=p.with_suffix('.tmp');q.write_text(json.dumps(report,indent=2));q.replace(p)
def free():
 s=subprocess.check_output(['lfs','quota','-u','2601110206','/lustre']).decode();m=re.search(r'/lustre\s+(\d+)\s+\d+\s+(\d+)',s);assert m;return (int(m[2])-int(m[1]))/1024**2
save()
try:
 assert free()>200,'Protect existing production quota'
 if mode=='formal':
  p=json.loads((R/('ADMISSION_'+key+'.json')).read_text());assert p['status']=='admission_passed' and p['binary_sha256']==case['binary_sha256'] and p['input_sha256']==case['input_sha256']
 cmd=['mpirun','-np',str(case['ranks']),'--map-by','ppr:%d:node'%case['gpus_per_node'],'--bind-to','none','bash',str(R/'rank.sh'),'bash',str(R/'monitor_exec.sh'),str(exe),'-i',str(deck)]
 if mode=='admission':cmd+=['time/nlim=32','time/ndiag=1']
 os.environ['AA_MODE']=mode;(work/'command.json').write_text(json.dumps(cmd,indent=2));start=time.monotonic();last=start
 with (work/'run.log').open('w') as log:
  child=subprocess.Popen(cmd,cwd=str(work),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while child.poll() is None:
    if time.monotonic()-last>60:report['free_GiB']=free();save();assert report['free_GiB']>150,'Protect production quota';last=time.monotonic()
    if mode=='admission' and time.monotonic()-start>1500:raise TimeoutError('Bounded GPU admission exceeded1500seconds')
    time.sleep(2)
  except BaseException:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   raise
 report.update(returncode=child.returncode,wall_seconds=time.monotonic()-start);assert child.returncode==0
 text=(work/'run.log').read_text(errors='replace');final=re.findall(r'^time=([0-9.eE+\-]+) cycle=(\d+)',text,re.M);assert final,text[-2000:];t,cy=float(final[-1][0]),int(final[-1][1]);report.update(final_time=t,final_cycle=cy)
 if mode=='admission':
  assert cy==32
  memory={}
  for rank in range(case['ranks']):
   rows=list(csv.reader((work/('gpu-memory-%d.csv'%rank)).read_text().splitlines()));vals=[(float(r[0]),float(r[1])) for r in rows if len(r)==2];assert vals
   peak=max(v[0] for v in vals);total=vals[-1][1]
   assert exe.name in (work/('gpu-apps-%d.csv'%rank)).read_text(),'Actual GPU process missing on rank%d'%rank
   memory[str(rank)]=dict(peak_MiB=peak,total_MiB=total)
  samples=[(float(a),int(b),float(c)) for a,b,c in re.findall(r'elapsed=([\deE+.\-]+) cycle=(\d+) time=([\deE+.\-]+)',text)]
  tail=samples[-17:];assert len(tail)>=8 and tail[-1][2]>tail[0][2]
  projected=(tail[-1][0]-tail[0][0])*500/(tail[-1][2]-tail[0][2])/3600
  report.update(gpu_memory=memory,projected500M_hours=projected,projection_scope='Early32 cycles with full native initial/final I/O; margin required for later dynamics')
  save();assert all(v['peak_MiB']<.8*v['total_MiB'] and v['total_MiB']-v['peak_MiB']>8192 for v in memory.values()),'Measured memory reserve insufficient'
  assert projected<110,'Measured throughput exceeds five-day budget; increase allocation before formal run'
  report['status']='admission_passed'
 else:
  assert abs(t-500)<1e-9;report['status']='completed500M_analysis_pending'
except BaseException as e:report.update(status=mode+'_failed',error=repr(e));save();raise
report['finished_utc']=datetime.datetime.utcnow().isoformat()+'Z';save();print(json.dumps(report,indent=2))
