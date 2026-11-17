from pathlib import Path
import datetime,json,socket,subprocess,time
assert socket.gethostname().startswith('wm2-login')
r=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu/native-tov-reference');p=r/'VALIDATED_SUBMISSIONS.json';v=json.loads(p.read_text());assert not v['formal']
for item in ('64','96'):assert item in v['validation']
def save():
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
finished=set()
while len(finished)<2:
 for n in (64,96):
  key=str(n)
  if key in finished:continue
  f=r/('N%d_VALIDATION.json'%n)
  if not f.exists():continue
  proof=json.loads(f.read_text());assert proof['job_id']==v['validation'][key]
  if proof['status']=='validation_failed':
   v.setdefault('failed_validations',{})[key]=proof.get('error','see receipt');finished.add(key);save();continue
  if proof['status']!='validation_passed_formal_ready':continue
  cmd=['sbatch','--parsable','--partition=GPU40G','--qos=low','--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=4','--time=5-00:00:00','--no-requeue','--job-name=ET_TOV_N%d_T100'%n,'--output='+str(r/'logs/formal.%j.out'),'--error='+str(r/'logs/formal.%j.err'),str(r/'validated.slurm'),key,'formal']
  v['formal'][key]=subprocess.check_output(cmd).decode().strip().split(';')[0];finished.add(key);v['updated_utc']=datetime.datetime.utcnow().isoformat()+'Z';save()
 if len(finished)<2:time.sleep(20)
v['status']='all_formal100_submitted' if len(v['formal'])==2 else 'qualified_cases_submitted_other_validation_failed';save()
