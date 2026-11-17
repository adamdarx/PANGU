from pathlib import Path
import datetime,json,subprocess
R=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-single-continuation')
assert not (R/'SUBMISSIONS.json').exists(),'Already submitted: inspect existing job IDs before retry'
out={'authorized_utc':datetime.datetime.utcnow().isoformat()+'Z','user_instruction':'继续续跑，athenak已有数据要用上','target':500,'original_running_jobs_unchanged':True,'cases':{}}
for kind,old in [('stationary','17260150'),('spinning','17260147')]:
 def submit(mode,dep=None):
  a=['sbatch','--parsable','--partition=GPU80G','--qos='+('high' if mode=='admission' else 'low'),'--nodes=1','--ntasks-per-node=4','--cpus-per-task=16','--gres=gpu:4','--no-requeue','--kill-on-invalid-dep=yes','--time='+('01:00:00' if mode=='admission' else '5-00:00:00'),'--job-name=SP192_RESTART_'+kind+'_'+mode,'--output='+str(R/'logs/%x-%j.log')]
  if dep:a+=['--dependency='+dep]
  a+=[str(R/'case.slurm'),mode,kind]
  return subprocess.check_output(a).decode().strip().split(';')[0]
 gate=submit('admission');formal=submit('formal','afterany:'+old+',afterok:'+gate)
 out['cases'][kind]=dict(original=old,admission=gate,formal=formal,dependency='afterany:'+old+',afterok:'+gate)
 (R/'SUBMISSIONS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
