"""Native ET vacuum GPU admission and gated long evolution; no CPU evolution."""
from pathlib import Path
import csv,datetime,hashlib,json,os,re,signal,subprocess,sys,time
import numpy as np
R=Path(__file__).resolve().parent;root=R.parent;mode,key=sys.argv[1:];case=next(c for c in json.loads((R/'PLAN.json').read_text())['cases'] if c['id']==key)
sys.path.insert(0,str(root))
from reference_history_guard import check_norms
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
build=json.loads((root/'AH_BUILD_STATUS.json').read_text());exe=Path(build['binary']);assert hashlib.sha256(exe.read_bytes()).hexdigest()==build['sha256'];deck=R/'inputs'/case['inputs'][mode]['file'];assert hashlib.sha256(deck.read_bytes()).hexdigest()==case['inputs'][mode]['sha256']
job=os.environ['SLURM_JOB_ID'];work=R/'runs'/key/job;work.mkdir(parents=True,exist_ok=False);receipt=R/('%s_%s.json'%(mode.upper(),key));report=dict(status=mode+'_running',job_id=job,case=key,mode=mode,work=str(work),target=case['target'],binary_sha256=build['sha256'],input_sha256=case['inputs'][mode]['sha256'],physical_acceptance=False,comparison_differences=case['differences'])
def save():
 for p in [receipt,work/'STATUS.json']:
  t=p.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2,allow_nan=False));t.replace(p)
def free():
 s=subprocess.check_output(['lfs','quota','-u','2601110206','/lustre']).decode();m=re.search(r'/lustre\s+(\d+)\s+\d+\s+(\d+)',s);assert m;return (int(m[2])-int(m[1]))/1024**2
save()
try:
 assert free()>200,'Protect active production quota'
 if mode=='formal':
  proof=json.loads((R/('ADMISSION_'+key+'.json')).read_text());assert proof['status']=='admission_passed' and proof['binary_sha256']==build['sha256'] and proof['input_sha256']==case['inputs']['admission']['sha256']
 cmd=['mpirun','-np',os.environ['SLURM_NTASKS'],'--bind-to','none','bash',str(root/'native-tov-reference/rank.sh'),str(exe),'--parameter-level','strict',str(deck)];(work/'command.json').write_text(json.dumps(cmd,indent=2))
 gpu=(work/'gpu.csv').open('w');mon=subprocess.Popen(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits','--loop=1'],stdout=gpu,stderr=subprocess.STDOUT)
 seen=False;start=time.monotonic();last=start;samples=[]
 try:
  with (work/'run.log').open('w') as log:
   p=subprocess.Popen(cmd,cwd=str(work),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    while p.poll() is None:
     apps=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=process_name','--format=csv,noheader']).decode();seen=seen or exe.name in apps
     if time.monotonic()-last>60:check_norms(work);report['free_GiB']=free();save();assert report['free_GiB']>150,'Protect production quota';last=time.monotonic()
     if mode=='admission':
      if time.monotonic()-start>1600:raise TimeoutError('Bounded GPU admission exceeded1600seconds')
      q=next(work.rglob('z4c-hc.tsv'),None)
      if q:
       try:
        d=np.atleast_2d(np.loadtxt(q))
        if int(d[-1,0])>=4 and (not samples or d[-1,1]>samples[-1][1]):samples.append((time.monotonic()-start,float(d[-1,1]),int(d[-1,0])))
       except (ValueError,IndexError):pass
     time.sleep(1)
   except BaseException:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
 finally:mon.terminate();mon.wait(timeout=10);gpu.close()
 check_norms(work,full=True)
 report.update(returncode=p.returncode,actual_GPU_process_observed=seen,wall_seconds=time.monotonic()-start);assert p.returncode==0 and seen
 text=(work/'run.log').read_text(errors='replace');assert 'Done.' in text
 hc=next(work.rglob('z4c-hc.tsv'));d=np.atleast_2d(np.loadtxt(hc));assert np.isfinite(d).all();report.update(final_time=float(d[-1,1]),final_cycle=int(d[-1,0]))
 ah=[]
 for q in work.rglob('*'):
  if q.is_file() and 'BH_diagnostics' in q.name:
   try:
    data=np.atleast_2d(np.loadtxt(q))
    if data.size and data.shape[1]>=2 and np.isfinite(data[:,:2]).all():ah.append(dict(file=str(q.relative_to(work)),rows=len(data)))
   except ValueError:pass
 report['horizon_diagnostic_files']=ah
 if case['kind']=='quasicircular':
  waves=[]
  for q in work.rglob('*l2_m2*'):
   if q.is_file():
    data=np.atleast_2d(np.loadtxt(q));assert np.isfinite(data).all();waves.append(str(q.relative_to(work)))
  report['waveform_l2m2_files']=waves;assert len(waves)>=3,'Missing three extraction-radius waveforms'
 if mode=='admission':
  assert report['final_cycle']==64 and len(ah)>=case['required_initial_horizons'],'Initial horizon or64-step check failed'
  memory={}
  for row in csv.reader((work/'gpu.csv').read_text().splitlines()):
   if len(row)!=3 or row[0].strip() not in os.environ['CUDA_VISIBLE_DEVICES'].split(','):continue
   used,total=float(row[1]),float(row[2]);v=memory.setdefault(row[0].strip(),dict(peak_MiB=0,total_MiB=total));v['peak_MiB']=max(v['peak_MiB'],used)
  report['GPU_memory']=memory;save();assert memory and all(v['peak_MiB']<.8*v['total_MiB'] and v['total_MiB']-v['peak_MiB']>8192 for v in memory.values()),'Insufficient GPU memory reserve'
  if len(samples)>=2:
   rate=(samples[-1][1]-samples[0][1])/(samples[-1][0]-samples[0][0]);hours=case['target']/rate/3600;report['projected_target_hours']=hours;assert hours<110,'Measured early throughput exceeds five-day budget'
  else:report['timing_limitation']='Too few interval samples; no steady-state projection claimed'
  report.update(status='admission_passed',limitations='64 steps verify GPU, allocation, finite norm diagnostics and initial AH/wave output. Initial-data/gauge/AMR differences require independent quantitative review.')
 else:
  assert abs(report['final_time']-case['target'])<max(1e-9,case['expected_dt']*1.1);report['status']='formal_completed_analysis_pending'
except BaseException as e:report.update(status=mode+'_failed',error=repr(e));save();raise
report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(json.dumps(report,indent=2))
