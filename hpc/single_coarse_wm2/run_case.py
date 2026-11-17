from pathlib import Path
import csv,datetime,hashlib,json,os,re,signal,subprocess,time
import h5py,numpy as np
R=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-single-coarse-convergence')
mode,kind,n=os.environ['COHORT_MODE'],os.environ['COHORT_KIND'],int(os.environ['COHORT_N']);assert mode in ('admission','formal') and kind in ('stationary','spinning') and n in (32,48,64)
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
case=next(c for c in json.loads((R/'PLAN.json').read_text())['cases'] if c['kind']==kind and c['n']==n)
meta=json.loads((R/'BINARY.json').read_text());exe=Path(meta['binary']);assert hashlib.sha256(exe.read_bytes()).hexdigest()==meta['sha256'];deck=R/'inputs'/case['input'];assert hashlib.sha256(deck.read_bytes()).hexdigest()==case['sha256'] and case['target']==500
work=R/'runs'/('%s_n%d'%(kind,n))/os.environ['SLURM_JOB_ID'];work.mkdir(parents=True,exist_ok=False)
receipt=R/('%s_%s_n%d.json'%(mode.upper(),kind,n));report=dict(status=mode+'_running',job_id=os.environ['SLURM_JOB_ID'],kind=kind,n=n,target_M=500,binary_sha256=meta['sha256'],input_sha256=case['sha256'],work=str(work),started_utc=datetime.datetime.utcnow().isoformat()+'Z',physical_acceptance=False)
def save():
 data=json.dumps(report,indent=2,allow_nan=False)+'\n'
 for target in (receipt,work/'STATUS.json'):
  p=target.with_suffix('.tmp');p.write_text(data);p.replace(target)
def free():
 text=subprocess.check_output(['lfs','quota','-u','2601110206','/lustre']).decode();m=re.search(r'/lustre\s+(\d+)\s+\d+\s+(\d+)',text);assert m;return (int(m[2])-int(m[1]))/1024**2
save()
try:
 report['initial_free_GiB']=free();assert report['initial_free_GiB']>200,'Protect existing production quota'
 if mode=='formal':
  v=json.loads((R/('ADMISSION_%s_n%d.json'%(kind,n))).read_text());assert v['status']=='admission_passed' and v['binary_sha256']==meta['sha256'] and v['input_sha256']==case['sha256'];proofs=[json.loads((R/('ADMISSION_%s_n%d.json'%(k,m))).read_text()) for k in ('stationary','spinning') for m in (32,48,64)]
  proofs=[p for p in proofs if p['status']=='admission_passed'];assert proofs
  geometries={f['geometry_sha256'] for p in proofs for f in p['fields']};assert len(geometries)==1,'Actual physical leaf layouts differ'
  marker=R/'COHORT_GEOMETRY_VERIFIED.json';temp=marker.with_name(marker.name+'.'+os.environ['SLURM_JOB_ID']+'.tmp');temp.write_text(json.dumps(dict(status='all_admitted_actual_GPU_initial_and_step16_mesh_geometries_identical',geometry_sha256=next(iter(geometries)))));temp.replace(marker)
 cmd=['mpirun','-np',os.environ['SLURM_NTASKS'],'--bind-to','none','bash',str(R/'rank.sh'),str(exe),'-i',str(deck),'-d','output']
 cmd+=['parthenon/output2/hdf5_compression_level=0','parthenon/output3/hdf5_compression_level=0']
 report['output_policy']='Uncompressed native HDF5; lossless zstd archival outside evolution'
 if mode=='admission':cmd+=['parthenon/time/nlim=16','parthenon/time/ncycle_out=1','parthenon/output2/dt=1e30','parthenon/output3/dt=1e30']
 (work/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');gpu=(work/'gpu.csv').open('w');monitor=subprocess.Popen(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits','--loop=1'],stdout=gpu,stderr=subprocess.STDOUT)
 start=time.monotonic();last=start;observed=False
 try:
  with (work/'run.log').open('w') as log:
   p=subprocess.Popen(cmd,cwd=str(work),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    while p.poll() is None:
     gpu_text=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid,process_name','--format=csv,noheader']).decode()
     if exe.name in gpu_text:observed=True
     if time.monotonic()-last>60:
      report['latest_free_GiB']=free();save();assert report['latest_free_GiB']>150,'Stop this new job only to protect production quota';last=time.monotonic()
     if mode=='admission' and time.monotonic()-start>2100:raise TimeoutError('Bounded GPU admission exceeded2100seconds')
     time.sleep(1)
   except BaseException:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
 finally:monitor.terminate();monitor.wait(timeout=10);gpu.close()
 report.update(returncode=p.returncode,actual_GPU_observed=observed,wall_seconds=time.monotonic()-start);assert p.returncode==0 and observed
 fields=sorted((work/'output').glob('*.phdf'));assert fields
 def inspect(path):
  with h5py.File(path) as f:
   info=f['Info'].attrs;geo=np.stack([f['Locations/'+a][:,s] for a in 'xyz' for s in (0,-1)],axis=1);levels=f['Levels'][:]
   keys=sorted([(int(l),*[0. if abs(float(x))<1e-9 else round(float(x),9) for x in row]) for l,row in zip(levels,geo)])
   fingerprint=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest();exterior=0
   for b in range(len(levels)):
    x,y,z=[f['VolumeLocations/'+a][b] for a in 'xyz'];rr=x[None,None,:]**2+y[None,:,None]**2+z[:,None,None]**2;mask=(rr>4)&(rr<=100)&(f['nr.constraint_mask'][b,0]>0)
    if mask.any():assert np.isfinite(f['nr.constraints'][b,:][:,mask]).all() and np.isfinite(f['nr.z4c'][b,:][:,mask]).all()
    exterior+=int(mask.sum())
   assert exterior>0
   minimum=float(np.min((geo[:,1]-geo[:,0])/int(info['MeshBlockSize'][0])))
   return dict(file=path.name,time=float(info['Time']),cycle=int(info['NCycle']),blocks=len(levels),geometry_sha256=fingerprint,min_dx=minimum,exterior_finite_cells=exterior)
 if mode=='admission':
  report['fields']=[inspect(fields[0]),inspect(fields[-1])];assert report['fields'][0]['cycle']==0 and report['fields'][-1]['cycle']==16
  assert all(abs(f['min_dx']/case['min_dx']-1)<1e-10 for f in report['fields'])
  memory={}
  for row in csv.reader((work/'gpu.csv').read_text().splitlines()):
   if len(row)!=3 or row[0].strip() not in os.environ['CUDA_VISIBLE_DEVICES'].split(','):continue
   used,total=float(row[1]),float(row[2]);v=memory.setdefault(row[0].strip(),dict(peak_MiB=0,total_MiB=total));v['peak_MiB']=max(v['peak_MiB'],used)
  report['GPU_memory']=memory;save();assert memory and all(v['peak_MiB']<.8*v['total_MiB'] and v['total_MiB']-v['peak_MiB']>8192 for v in memory.values()),'Insufficient memory reserve'
  report['status']='admission_passed'
 else:
  final=inspect(fields[-1]);assert abs(final['time']-500)<1e-9;report.update(status='completed500M_physical_analysis_pending',final=final)
except BaseException as e:report.update(status=mode+'_failed',error=repr(e));save();raise
report['finished_utc']=datetime.datetime.utcnow().isoformat()+'Z';save();print(json.dumps(report,indent=2))
