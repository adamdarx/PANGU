"""LOGIN watcher submits allocated high-QOS probes, then low-QOS formal runs after success."""
from pathlib import Path
import datetime,json,socket,subprocess,time
ROOT=Path(__file__).resolve().parent;assert socket.gethostname().startswith('wm2-login')
record=dict(status='waiting_for_CUDA_build',created_utc=datetime.datetime.utcnow().isoformat()+'Z',resolutions=[64,96],admissions={},formal={})
receipt=ROOT/'TOV_SUBMISSIONS.json'
assert not receipt.exists(),'Refuse duplicate submit controller; inspect existing receipt first'
def save():
    tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(receipt)
def submit(mode,n):
    ranks=1 if n==64 else 2
    command=['sbatch','--parsable','--partition=GPU40G','--qos='+('high' if mode=='admission' else 'low'),
             '--nodes=1','--ntasks='+str(ranks),'--gres=gpu:'+str(ranks),'--cpus-per-task=4',
             '--time='+('00:20:00' if mode=='admission' else '5-00:00:00'),'--no-requeue',
             '--job-name=AK_TOV_N%d_%s'%(n,'ADMIT' if mode=='admission' else 'T100'),
             '--output='+str(ROOT/'logs'/'tov.%j.out'),'--error='+str(ROOT/'logs'/'tov.%j.err'),
             str(ROOT/'tov.slurm'),mode,str(n)]
    return subprocess.check_output(command).decode().strip().split(';')[0]
save()
while True:
    state=json.loads((ROOT/'TOV_BUILD_STATUS.json').read_text())
    if state['status'].endswith('build_failed'):record.update(status='build_failed_no_jobs_submitted');save();raise SystemExit(1)
    if state['status']=='TOV_sm80_GPU_binary_frozen_input_validation_pending':break
    time.sleep(20)
for n in (64,96):
    assert (ROOT/'initial-reference'/('pangu_n%d_initial.npz'%n)).exists()
    record['admissions'][str(n)]=submit('admission',n);save()
record['status']='high_GPU_admissions_submitted';save()
while len(record['formal'])<2:
    for n in (64,96):
        key=str(n);path=ROOT/('TOV_ADMISSION_N%d.json'%n)
        if key in record['formal'] or not path.exists():continue
        state=json.loads(path.read_text());assert state['job_id']==record['admissions'][key]
        if state['status']=='admission_failed':record.update(status='admission_failed_no_unqualified_formal_submission',failure_resolution=n);save();raise SystemExit(1)
        if state['status']=='admission_passed':record['formal'][key]=submit('formal',n);save()
    time.sleep(20)
record['status']='formal_N64_N96_T100_submitted';record['finished_utc']=datetime.datetime.utcnow().isoformat()+'Z';save()
