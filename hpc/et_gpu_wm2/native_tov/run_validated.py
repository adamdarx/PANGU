from pathlib import Path
import csv,datetime,hashlib,json,os,re,signal,subprocess,sys,time
import numpy as np
ROOT=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu');suite=ROOT/'native-tov-reference'
sys.path.insert(0,str(ROOT))
from reference_history_guard import check_norms
n=int(os.environ['REFERENCE_N']);mode=os.environ.get('REFERENCE_MODE','validation');assert mode in ('validation','formal') and n in (64,96)
profile=os.environ.get('REFERENCE_PROFILE','original');assert profile in ('original','stability')
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
build=json.loads((ROOT/'AH_BUILD_STATUS.json').read_text());exe=Path(build['binary']);assert hashlib.sha256(exe.read_bytes()).hexdigest()==build['sha256']
work=suite/'runs'/('n%d'%n)/os.environ['SLURM_JOB_ID'];work.mkdir(parents=True,exist_ok=False)
report=dict(status=mode+'_running',job_id=os.environ['SLURM_JOB_ID'],n=n,mode=mode,binary_sha256=build['sha256'],started_utc=datetime.datetime.utcnow().isoformat()+'Z',physical_acceptance=False,work=str(work));receipt=suite/(('N%d_STABILITY_%s.json' if profile=='stability' else 'N%d_%s.json')%(n,mode.upper()))
report['profile']=profile
def save():
 p=receipt.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(receipt)
def free():
 text=subprocess.check_output(['lfs','quota','-u','2601110206','/lustre']).decode();h=re.search(r'/lustre\s+(\d+)\s+\d+\s+(\d+)',text);assert h
 return (int(h[2])-int(h[1]))/1024**2
save()
try:
 assert free()>200,'Protect existing production quota'
 deck=suite/(('native_tov_n%d_%s_stability.par' if profile=='stability' else 'native_tov_n%d_%s.par')%(n,mode));report['input_sha256']=hashlib.sha256(deck.read_bytes()).hexdigest();save()
 if mode=='formal':
  proof=json.loads((suite/(('N%d_STABILITY_VALIDATION.json' if profile=='stability' else 'N%d_VALIDATION.json')%n)).read_text());assert proof['status']=='validation_passed_formal_ready' and proof['binary_sha256']==build['sha256']
 cmd=['mpirun','-np',os.environ['SLURM_NTASKS'],'--bind-to','none','bash',str(suite/'rank.sh'),str(exe),str(deck)];(work/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 gpu=(work/'gpu.csv').open('w');monitor=subprocess.Popen(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits','--loop=1'],stdout=gpu,stderr=subprocess.STDOUT)
 observed=False;start=time.monotonic();last=start
 try:
  with (work/'run.log').open('w') as log:
   child=subprocess.Popen(cmd,cwd=str(work),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    while child.poll() is None:
     processes=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid,process_name','--format=csv,noheader']).decode()
     with (work/'gpu-processes.log').open('a') as f:f.write(processes)
     if exe.name in processes:observed=True
     if time.monotonic()-last>60:
      check_norms(work)
      assert free()>150,'Terminate this new reference only to protect production quota';last=time.monotonic()
     if mode=='validation' and time.monotonic()-start>(3600 if profile=='stability' else 1000):raise TimeoutError('GPU validation exceeded1000seconds')
     time.sleep(1)
   except BaseException:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise
 finally:monitor.terminate();monitor.wait(timeout=10);gpu.close()
 check_norms(work,full=True)
 report.update(returncode=child.returncode,actual_GPU_process_observed=observed);assert child.returncode==0 and observed,'GPU execution/exit check failed'
 assert 'Done.' in (work/'run.log').read_text(errors='replace')
 memory={}
 for row in csv.reader((work/'gpu.csv').read_text().splitlines()):
  if len(row)!=3 or row[0].strip() not in os.environ['CUDA_VISIBLE_DEVICES'].split(','):continue
  used,total=float(row[1]),float(row[2]);v=memory.setdefault(row[0].strip(),dict(peak_MiB=0,total_MiB=total));v['peak_MiB']=max(v['peak_MiB'],used)
 report['GPU_memory']=memory
 def norm(var):
  path=next(work.rglob(var+'.tsv'));header=path.read_text().splitlines()[0];cols={key:int(i)-1 for i,key in re.findall(r'(\d+):([^\s\\]+)',header)};data=np.atleast_2d(np.loadtxt(path));return data,cols
 d,c=norm('asterx-cons_vector');rho,rc=norm('hydrobasex-rho');mag,mc=norm('asteranalysis-mhd_analysis');h,hc=norm('z4c-hc')
 mass=d[:,c['asterx::dens.sum']];density=rho[:,rc['hydrobasex::rho.max']];energy=mag[:,mc['asteranalysis::magnetic_energy.sum']]
 assert np.isfinite(mass).all() and np.isfinite(density).all() and np.isfinite(energy).all() and np.isfinite(h).all()
 report['measurements']=dict(initial_baryon_mass=float(mass[0]),initial_rho_max=float(density[0]),initial_magnetic_energy=float(energy[0]),max_relative_baryon_drift=float(np.max(abs(mass/mass[0]-1))),initial_H_L2=float(h[0,hc['z4c::hc.L2norm']]),final_time=float(d[-1,1]))
 if mode=='validation':
  assert int(d[-1,0])==(n*10 if profile=='stability' else 16),'Incomplete validation trajectory'
  if profile=='stability':assert abs(float(d[-1,1])-2)<1e-9,'Must exceed earlier0.9/1.4M nonfinite onset'
  assert memory and all(v['peak_MiB']<.8*v['total_MiB'] and v['total_MiB']-v['peak_MiB']>8192 for v in memory.values()),'Insufficient GPU memory reserve'
  assert 1.4<mass[0]<1.65 and abs(density[0]/.007993-1)<.05 and 0<energy[0]<.01*1.4475941892,'Unexpected physical model scale'
  target=json.loads((suite/'SEED_ENERGY_CALIBRATION.json').read_text())['fine']['PANGU_continuum_seed_energy']
  report['measurements']['initial_magnetic_energy_relative_to_continuum_target']=float(energy[0]/target-1)
  assert abs(energy[0]/target-1)<.3,'Initial magnetic energy not consistent with selected weak-field model'
  assert report['measurements']['max_relative_baryon_drift']<1e-4,'Early conservation check failed'
  report['status']='validation_passed_formal_ready'
 else:
  assert abs(float(d[-1,1])-100)<1e-9,'Target100 not reached';report['status']='formal_completed_quantitative_analysis_pending'
except BaseException as e:report.update(status=mode+'_failed',error=repr(e));save();raise
report['finished_utc']=datetime.datetime.utcnow().isoformat()+'Z';save();print(json.dumps(report,indent=2))
