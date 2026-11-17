"""LOGIN-only: wait for all six real GPU mesh checks, then submit six500M jobs."""
from pathlib import Path
import json,socket,subprocess,time,datetime
assert socket.gethostname().startswith('wm2-login')
R=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-single-coarse-convergence');p=R/'SUBMISSIONS.json';v=json.loads(p.read_text());assert len(v['admission'])==6 and not v['formal']
def save():
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
while True:
 proofs=[]
 for key,job in v['admission'].items():
  f=R/('ADMISSION_'+key+'.json')
  if not f.exists():continue
  data=json.loads(f.read_text());
  if data['job_id']!=job:continue
  if data['status']=='admission_failed':
   log=(Path(data['work'])/'run.log').read_text(errors='replace').lower()
   memory_failure='memory reserve' in data.get('error','').lower() or any(t in log for t in ('out of memory','bad_alloc','memoryallocation','memory allocation','failed to allocate'))
   if memory_failure and key not in v.get('GPU80_retry',{}):
    kind,num=key.rsplit('_n',1)
    cmd=['sbatch','--parsable','--partition=GPU80G','--qos=high','--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=4','--time=00:20:00','--no-requeue','--job-name=SP_COARSE_'+key+'_ADMIT80','--output='+str(R/'logs/%j.out'),'--error='+str(R/'logs/%j.err'),str(R/'case.slurm'),'admission',kind,num]
    newjob=subprocess.check_output(cmd).decode().strip().split(';')[0]
    v.setdefault('GPU80_retry',{})[key]=dict(previous_job=job,retry_job=newjob,reason='Measured GPU40G memory admission failed; same input/program, one80G GPU')
    v['admission'][key]=newjob;v['status']='awaiting_GPU80_retry';v.pop('error',None);v.pop('failed_case',None);save();continue
   v.update(status='admission_failed_formal_not_submitted',failed_case=key,error=data.get('error'));save();raise SystemExit(1)
  if data['status']=='admission_passed':proofs.append(data)
 if len(proofs)==6:break
 time.sleep(20)
assert all(x['target_M']==500 for x in proofs)
initial={x['fields'][0]['geometry_sha256'] for x in proofs};final={x['fields'][-1]['geometry_sha256'] for x in proofs}
assert len(initial)==len(final)==1 and initial==final,'Actual physical leaf layouts differ: do not run a false convergence series'
(R/'COHORT_GEOMETRY_VERIFIED.json').write_text(json.dumps(dict(status='all_six_actual_GPU_initial_and_step16_mesh_geometries_identical',geometry_sha256=next(iter(initial)),cases=[dict(kind=x['kind'],n=x['n'],min_dx=x['fields'][0]['min_dx']) for x in proofs],scope='Actual16-step validation; later AMR geometry must still be checked in the500M data.'),indent=2)+'\n')
# Local archive ownership was established before this controller was launched.
assert (R/'ARCHIVE_OWNER.json').exists()
for data in proofs:
 key='%s_n%d'%(data['kind'],data['n'])
 cmd=['sbatch','--parsable','--partition='+('GPU80G' if key in v.get('GPU80_retry',{}) else 'GPU40G'),'--qos=low','--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=4','--time=5-00:00:00','--no-requeue','--job-name=SP500_COARSE_'+key,'--output='+str(R/'logs/%j.out'),'--error='+str(R/'logs/%j.err'),str(R/'case.slurm'),'formal',data['kind'],str(data['n'])]
 v['formal'][key]=subprocess.check_output(cmd).decode().strip().split(';')[0];save()
v.update(status='all_six_formal500M_submitted',UTC=datetime.datetime.utcnow().isoformat()+'Z');save()
