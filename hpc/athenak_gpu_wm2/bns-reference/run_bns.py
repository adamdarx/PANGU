"""Allocated CUDA admission/formal execution; never accepts a short run as physics validation."""
from pathlib import Path
import argparse,csv,datetime,hashlib,importlib.util,json,os,re,signal,subprocess,time
import numpy as np
from history_guard import check_history

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['admission','formal']);parser.add_argument('resolution',type=int,choices=[64,96]);args=parser.parse_args()
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
binary=json.loads((ROOT/'BNS_BUILD_STATUS.json').read_text());exe=Path(binary['binary'])
assert binary['CUDA_device_images_verified'] and hashlib.sha256(exe.read_bytes()).hexdigest()==binary['sha256']
case=next(c for c in json.loads((ROOT/'BNS_INPUTS.json').read_text())['cases'] if c['resolution']==args.resolution)
deck=Path(case['input']);assert hashlib.sha256(deck.read_bytes()).hexdigest()==case['sha256']
job=os.environ['SLURM_JOB_ID'];ranks=int(os.environ['SLURM_NTASKS'])
work=ROOT/'runs'/('bns_n%d'%args.resolution)/job;work.mkdir(parents=True,exist_ok=False)
report=dict(status=args.mode+'_running',mode=args.mode,resolution=args.resolution,job_id=job,
            started_utc=datetime.datetime.utcnow().isoformat()+'Z',binary_sha256=binary['sha256'],
            input_sha256=case['sha256'],work=str(work),physical_acceptance=False)
receipt=ROOT/('BNS_%s_N%d.json'%(args.mode.upper(),args.resolution))
def save():
    temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(receipt)
def quota_free():
    value=subprocess.check_output(['lfs','quota','-u','2601110206','/lustre']).decode()
    hit=re.search(r'/lustre\s+(\d+)\s+\d+\s+(\d+)',value);assert hit,value
    return (int(hit[2])-int(hit[1]))/1024**2
def load_reader():
    spec=importlib.util.spec_from_file_location('akreader',str(ROOT/'source/vis/python/bin_convert.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.read_binary
save()
try:
    assert quota_free()>200,'Protect existing production: fewer than200 GiB free before launch'
    if args.mode=='formal':
        admission=json.loads((ROOT/('BNS_ADMISSION_N%d.json'%args.resolution)).read_text())
        assert admission['status']=='admission_passed' and admission['binary_sha256']==binary['sha256'] and admission['input_sha256']==case['sha256']
        report['admission_job_id']=admission['job_id'];save()
    command=['mpirun','-np',str(ranks),'--bind-to','none','bash',str(ROOT/'rank.sh'),str(exe),'-i',str(deck)]
    if args.mode=='admission':
        command+=['time/nlim=8','time/ndiag=1','fastflow/start_time_0=1e30']
        command+=['output%d/dt=1e30'%i for i in range(2,10)]
    (work/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    gpu=(work/'gpu-memory.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits','--loop=1'],stdout=gpu,stderr=subprocess.STDOUT)
    observed=False;last_quota=time.monotonic();start=time.monotonic();last_history=start
    try:
        with (work/'run.log').open('w') as log:
            child=subprocess.Popen(command,cwd=str(work),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while child.poll() is None:
                    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']).decode()
                    if str(exe) in processes or exe.name in processes:observed=True
                    with (work/'gpu-processes.log').open('a') as stream:stream.write(processes)
                    if time.monotonic()-last_history>15:
                        check_history(work);last_history=time.monotonic()
                    if time.monotonic()-last_quota>60:
                        free=quota_free();report['latest_free_GiB']=free;save();last_quota=time.monotonic()
                        if free<150:raise RuntimeError('Storage reserve below150GiB; terminate this new reference only')
                    if args.mode=='admission' and time.monotonic()-start>900:raise TimeoutError('Bounded admission exceeded900seconds')
                    time.sleep(1)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
    finally:
        monitor.terminate();monitor.wait(timeout=10);gpu.close()
    check_history(work,full=True)
    report.update(returncode=child.returncode,wall_seconds=time.monotonic()-start,actual_gpu_process_observed=observed)
    assert child.returncode==0 and observed,'CUDA process or successful exit missing'
    text=(work/'run.log').read_text(errors='replace')
    final=re.findall(r'^time=([0-9.eE+\-]+) cycle=(\d+)',text,re.M);assert final,text[-2000:]
    report.update(final_time=float(final[-1][0]),final_cycle=int(final[-1][1]))
    assert report['final_cycle']==8 if args.mode=='admission' else abs(report['final_time']-200)<1e-10
    memory={}
    # Only allocated devices: CUDA_VISIBLE_DEVICES carries physical indices on WM2.
    visible=set(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    for row in csv.reader((work/'gpu-memory.csv').read_text().splitlines()):
        if len(row)!=5 or row[1].strip() not in visible:continue
        key=row[1].strip();used,total=float(row[2]),float(row[3]);item=memory.setdefault(key,dict(total_MiB=total,peak_sampled_MiB=0));item['peak_sampled_MiB']=max(item['peak_sampled_MiB'],used)
    report['gpu_memory']=memory;assert len(memory)==ranks,memory
    if args.mode=='admission':
        assert all(v['peak_sampled_MiB']<.8*v['total_MiB'] and v['total_MiB']-v['peak_sampled_MiB']>8192 for v in memory.values()),'Memory reserve insufficient'
        read=load_reader();reference_path=ROOT/'initial-reference'/('bns_n%d_initial.npz'%args.resolution)
        proof=json.loads(reference_path.with_suffix('.json').read_text());assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==proof['npz_sha256']
        reference=np.load(reference_path)
        wfile=next((work/'bin').glob('*.mhd_w_bcc.00000.bin'));ufile=next((work/'bin').glob('*.mhd_u.00000.bin'))
        w,u=read(str(wfile)),read(str(ufile));assert w['time']==0 and u['time']==0
        target_keys=[(int(lev),*map(int,loc)) for lev,loc in zip(reference['levels'],reference['locations'])]
        source_keys=[(int(row[3]),*map(int,row[:3])) for row in w['mb_logical']]
        assert len(set(source_keys))==len(source_keys) and set(target_keys)==set(source_keys),'Different initial leaf cells'
        index={key:i for i,key in enumerate(source_keys)};order=np.array([index[key] for key in target_keys]);geometry=np.asarray(w['mb_geometry'])[order]
        assert np.allclose(geometry,reference['geometry'],rtol=0,atol=1e-12),'Different physical block bounds'
        assert np.array_equal(w['mb_logical'],u['mb_logical'])
        volume=np.prod((geometry[:,1::2]-geometry[:,::2])/(args.resolution//4),axis=1)[:,None,None,None]
        arho=np.asarray(w['mb_data']['dens'],dtype=float)[order];adens=np.asarray(u['mb_data']['dens'],dtype=float)[order]
        rho_l1=float(np.sum(abs(arho-reference['rho'])*volume)/np.sum(abs(reference['rho'])*volume))
        mass_bias=float(np.sum(adens*volume)/proof['baryon_mass']-1)
        ab=np.stack([np.asarray(w['mb_data']['bcc%d'%i],dtype=float)[order] for i in (1,2,3)],axis=1)
        magnetic_l1=float(np.max(abs(ab-reference['bfield']))) # BNS reference has exactly zero magnetic field
        report['initial_match']=dict(blocks=len(order),rho_relative_volume_L1=rho_l1,baryon_relative_bias=mass_bias,
                                    magnetic_absolute_Linf=magnetic_l1,pangu_source_sha256=proof['source_phdf_sha256'])
        save();assert rho_l1<1e-5 and abs(mass_bias)<1e-5 and magnetic_l1<1e-4,'Initial-state match failed; do not submit formal comparison'
        for path in (work/'bin').glob('*.bin'):
            snapshot=read(str(path));assert all(np.isfinite(a).all() for a in snapshot['mb_data'].values()),'Nonfinite early fields'
        report['status']='admission_passed'
        report['limitations']='Eight early steps; not long-time accuracy or horizon acceptance. AH search deferred during this resource probe; AH objects and full field/restart I/O remain allocated.'
    else:report['status']='formal_evolution_completed_analysis_pending'
except BaseException as error:
    report.update(status=args.mode+'_failed',error=repr(error));save();raise
report['finished_utc']=datetime.datetime.utcnow().isoformat()+'Z';save();print(json.dumps(report,indent=2))
