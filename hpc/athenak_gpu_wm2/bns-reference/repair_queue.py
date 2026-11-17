"""LOGIN-only: after diagnostic adapter rebuild, queue fresh GPU gates and200M."""
from pathlib import Path
import os,time,json,subprocess,fcntl,sys,datetime
R=Path(__file__).resolve().parent;pid=int(sys.argv[1]);receipt=R/'BNS_REPAIR_SUBMISSIONS.json'
with (R/'bns-repair-submit.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not receipt.exists(),'Repair already submitted; inspect existing receipt'
 while Path('/proc/%d/stat'%pid).exists():
  if Path('/proc/%d/stat'%pid).read_text().split()[2]=='Z':break
  time.sleep(5)
 d=json.loads((R/'BNS_BUILD_STATUS.json').read_text());assert d['status']=='BNS_sm80_GPU_binary_frozen_input_validation_pending' and d['returncode']==0,d
 old=json.loads((R/'BNS_SUBMISSIONS.json').read_text());v=dict(admission={},formal={},target=200,binary_sha256=d['sha256'],superseded=old,reason='Time-zero initializer failed independently checked PANGU match; adapter-stage diagnostic rebuild; all new runs gated')
 for n,g in [(64,1),(96,4)]:
  for mode in ['admission','formal']:
   cmd=['sbatch','--parsable','--partition='+('GPU80G' if n==64 else 'GPU40G'),'--qos='+('high' if mode=='admission' else 'low'),'--nodes=1','--ntasks='+str(g),'--gres=gpu:'+str(g),'--cpus-per-task=4','--time='+('00:25:00' if mode=='admission' else '5-00:00:00'),'--no-requeue','--job-name=AK_BNS_N%d_%s_RECHECK'%(n,mode),'--output='+str(R/'logs/bns.%j.out'),'--error='+str(R/'logs/bns.%j.err')]
   dep=v['admission'].get(str(n)) if mode=='formal' else v['admission'].get('64') if n==96 else None
   if dep:cmd+=['--dependency=afterok:'+dep,'--kill-on-invalid-dep=yes']
   cmd += [str(R/'bns.slurm'),mode,str(n)];v[mode][str(n)]=subprocess.check_output(cmd).decode().strip().split(';')[0];receipt.write_text(json.dumps(v,indent=2));(R/'BNS_SUBMISSIONS.json').write_text(json.dumps(v,indent=2))
 v['queued_utc']=datetime.datetime.utcnow().isoformat()+'Z';receipt.write_text(json.dumps(v,indent=2));(R/'BNS_SUBMISSIONS.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
