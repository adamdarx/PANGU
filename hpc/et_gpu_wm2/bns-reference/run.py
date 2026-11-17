"""GPU-only BNS reference, initial-state checks and gated long evolution."""
from pathlib import Path
import csv,datetime,hashlib,json,os,re,signal,subprocess,sys,time
import numpy as np
R=Path(__file__).resolve().parent;root=R.parent;mode=sys.argv[1];n=int(sys.argv[2])
sys.path.insert(0,str(root))
from reference_history_guard import check_norms
assert mode in ['admission','formal'] and n in [64,96]
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
case=next(c for c in json.loads((R/'PLAN.json').read_text())['cases'] if c['n']==n)
build=json.loads((R/'BUILD_STATUS.json').read_text());exe=Path(build['binary'])
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(exe)==build['sha256'];assert digest(Path(case['table']))==case['table_sha256']
deck=R/case['inputs'][mode]['file'];assert digest(deck)==case['inputs'][mode]['sha256']
work=R/'runs'/('n%d'%n)/os.environ['SLURM_JOB_ID'];work.mkdir(parents=True,exist_ok=False)
receipt=R/('N%d_%s.json'%(n,mode.upper()));report=dict(status=mode+'_running',job_id=os.environ['SLURM_JOB_ID'],n=n,mode=mode,work=str(work),target=200,binary_sha256=build['sha256'],input_sha256=digest(deck),table_sha256=case['table_sha256'],physical_acceptance=False)
def save():
 for p in [receipt,work/'STATUS.json']:
  q=p.with_suffix('.tmp');q.write_text(json.dumps(report,indent=2,allow_nan=False));q.replace(p)
def free():
 s=subprocess.check_output(['lfs','quota','-u','2601110206','/lustre']).decode();m=re.search(r'/lustre\s+(\d+)\s+\d+\s+(\d+)',s);assert m;return (int(m[2])-int(m[1]))/1024**2
save()
try:
 assert free()>200,'Protect production storage'
 if mode=='formal':
  proof=json.loads((R/('N%d_ADMISSION.json'%n)).read_text());assert proof['status']=='admission_passed' and proof['binary_sha256']==build['sha256'] and proof['table_sha256']==case['table_sha256'] and proof['input_sha256']==case['inputs']['admission']['sha256']
 cmd=['mpirun','-np',os.environ['SLURM_NTASKS'],'--bind-to','none','bash',str(root/'native-tov-reference/rank.sh'),'bash',str(R/'monitor_exec.sh'),str(exe),'--parameter-level','strict',str(deck)]
 (work/'command.json').write_text(json.dumps(cmd,indent=2));start=time.monotonic();last=start;samples=[];os.environ['AA_MODE']=mode
 with (work/'run.log').open('w') as log:
  child=subprocess.Popen(cmd,cwd=str(work),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while child.poll() is None:
    now=time.monotonic()
    if now-last>60:check_norms(work);report['free_GiB']=free();save();assert report['free_GiB']>150,'Protect production storage';last=now
    if mode=='admission':
     if now-start>1500:raise TimeoutError('Bounded GPU admission exceeded1500seconds')
     p=next(work.rglob('hydrobasex-rho.tsv'),None)
     if p:
      try:
       d=np.atleast_2d(np.loadtxt(p))
       if int(d[-1,0])>=4 and (not samples or d[-1,1]>samples[-1][1]):samples.append((now-start,float(d[-1,1])))
      except (ValueError,IndexError):pass
    time.sleep(1)
  except BaseException:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   raise
 check_norms(work,full=True)
 report.update(returncode=child.returncode,wall_seconds=time.monotonic()-start);assert child.returncode==0
 text=(work/'run.log').read_text(errors='replace');assert 'Done.' in text and 'BNS_TABLE source0=' in text
 memory={}
 for rank in range(int(os.environ['SLURM_NTASKS'])):
  vals=[(float(r[0]),float(r[1])) for r in csv.reader((work/('gpu-memory-%d.csv'%rank)).read_text().splitlines()) if len(r)==2];assert vals
  memory[str(rank)]=dict(peak_MiB=max(x[0] for x in vals),total_MiB=vals[-1][1]);assert exe.name in (work/('gpu-apps-%d.csv'%rank)).read_text()
 report.update(actual_GPU_process_observed=True,GPU_memory=memory)
 def norm(name):
  p=next(work.rglob(name+'.tsv'));header=p.read_text().splitlines()[0];cols={k:int(i)-1 for i,k in re.findall(r'(\d+):([^\s\\]+)',header)};d=np.atleast_2d(np.loadtxt(p));assert np.isfinite(d).all();return d,cols
 d,c=norm('asterx-cons_vector');rho,rc=norm('hydrobasex-rho');h,hc=norm('z4c-hc');mag,mc=norm('asteranalysis-mhd_analysis');vel,vc=norm('hydrobasex-vel')
 mass=d[:,c['asterx::dens.sum']];density=rho[:,rc['hydrobasex::rho.max']];energy=mag[:,mc['asteranalysis::magnetic_energy.sum']]
 initial_velocity=max(abs(vel[0,i]) for k,i in vc.items() if k.endswith('.maxabs'))
 bvec,bc=norm('hydrobasex-bvec');initial_B=max(abs(bvec[0,i]) for k,i in bc.items() if k.endswith('.maxabs'))
 epsilon=np.finfo(float).eps;B_tolerance=64*epsilon*np.sqrt(density[0]);energy_tolerance=64*epsilon**2*mass[0]
 initial=json.loads((R/'INITIAL_REFERENCE.json').read_text())[str(n)]
 measured=dict(initial_baryon_mass=float(mass[0]),initial_rho_max=float(density[0]),initial_magnetic_energy=float(energy[0]),initial_velocity_maxabs=float(initial_velocity),relative_baryon_difference_to_PANGU=float(mass[0]/initial['baryon_mass']-1),max_relative_baryon_drift=float(np.max(abs(mass/mass[0]-1))),initial_H_L2=float(h[0,hc['z4c::hc.L2norm']]),final_time=float(d[-1,1]),final_cycle=int(d[-1,0]))
 measured.update(initial_B_maxabs=float(initial_B),magnetic_roundoff_B_tolerance=float(B_tolerance),magnetic_roundoff_energy_tolerance=float(energy_tolerance))
 report['measurements']=measured;save()
 if mode=='admission':
  assert measured['final_cycle']==32
  assert abs(measured['relative_baryon_difference_to_PANGU'])<.01,'Initial mass differs >1%; investigate grid/staggering'
  assert abs(density[0]/initial['rho_max']-1)<.05,'Initial stellar density differs >5%'
  assert initial_velocity<1e-12 and initial_B<=B_tolerance and 0<=energy[0]<=energy_tolerance,'Initial v/B exceeds the from-rest unmagnetized roundoff budget'
  assert measured['max_relative_baryon_drift']<1e-4,'Early baryon conservation failed'
  assert all(v['peak_MiB']<.8*v['total_MiB'] and v['total_MiB']-v['peak_MiB']>8192 for v in memory.values()),'Insufficient GPU memory reserve'
  assert len(samples)>=8 and samples[-1][1]>samples[0][1],'Insufficient throughput measurements'
  hours=(samples[-1][0]-samples[0][0])*200/(samples[-1][1]-samples[0][1])/3600;report['projected200M_hours']=hours;save();assert hours<110,'Insufficient measured speed for five-day200M budget'
  report.update(status='admission_passed',limitations='32-step technical/initial-state acceptance. Same initial table is not an independent initial-data solver. Long-time waveform, mass, horizon and convergence comparison pending.')
 else:
  assert abs(measured['final_time']-200)<case['dt']*1.1;report['status']='formal_completed_analysis_pending'
except BaseException as e:report.update(status=mode+'_failed',error=repr(e));save();raise
report['finished_utc']=datetime.datetime.utcnow().isoformat()+'Z';save();print(json.dumps(report,indent=2))
