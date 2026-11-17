"""LOGIN single-owner queue controller; submit once after the CUDA build succeeds."""
from pathlib import Path
import datetime,fcntl,json,os,socket,subprocess,time
R=Path(__file__).resolve().parent
assert socket.gethostname().startswith('wm2-login')
with (R/'queue.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (R/'SUBMISSIONS.json').exists():raise SystemExit('Already queued; inspect manifest')
 while True:
  v=json.loads((R/'BUILD_STATUS.json').read_text()) if (R/'BUILD_STATUS.json').exists() else {}
  if v.get('status')=='build_failed':raise SystemExit('Build failed; no GPU submissions')
  if v.get('status')=='compiled_frozen_GPU_admission_required':break
  time.sleep(15)
 result=dict(admission={},formal={},status='GPU_gated_N64_N96_queued',physical_acceptance=False,utc=datetime.datetime.utcnow().isoformat()+'Z')
 for n,gpus in [(64,1),(96,2)]:
  base=['sbatch','--parsable','--partition=GPU80G','--nodes=1','--ntasks='+str(gpus),'--ntasks-per-node='+str(gpus),'--gres=gpu:'+str(gpus),'--cpus-per-task=4','--no-requeue','--output='+str(R/'logs/%j.out'),'--error='+str(R/'logs/%j.err')]
  pre=[] if n==64 else ['--dependency=afterok:'+result['admission']['64'],'--kill-on-invalid-dep=yes']
  gate=subprocess.check_output(base+pre+['--qos=high','--time=00:30:00','--job-name=ET_BNS_N%d_admission'%n,str(R/'case.slurm'),'admission',str(n)]).decode().strip().split(';')[0];result['admission'][str(n)]=gate
  formal=subprocess.check_output(base+['--qos=low','--time=5-00:00:00','--dependency=afterok:'+gate,'--kill-on-invalid-dep=yes','--job-name=ET_BNS_N%d_T200'%n,str(R/'case.slurm'),'formal',str(n)]).decode().strip().split(';')[0];result['formal'][str(n)]=formal
  (R/'SUBMISSIONS.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
