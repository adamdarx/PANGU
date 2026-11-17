"""LOGIN monitor: stop old paired allocation only AFTER validated PANGU500M."""
from pathlib import Path
import json,subprocess,time,fcntl,datetime
R=Path(__file__).resolve().parent;old=Path('/lustre/home/2601110206/CODEX_WORKPLACE/SinglePuncture/20261030-spin-accuracy-rerun/runs')
items={'d200_spinning_n128':'17313452','d200_spinning_n192':'17260147','d200_stationary_n128':'17260149','d200_stationary_n192':'17260150'}
p=R/'SERIAL_STOP_STATUS.json';state=json.loads(p.read_text()) if p.exists() else dict(completed={})
with (R/'serial-stop.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while len(state['completed'])<len(items):
  for key,job in items.items():
   if key in state['completed']:continue
   proof=old/key/job/'attempt-0/pangu/phase-validated.json'
   if not proof.exists():continue
   d=json.loads(proof.read_text());assert d['status']=='completed' and d['returncode']==0 and d['last_logged_time']==500
   subprocess.check_call(['scancel',job]);state['completed'][key]=dict(job_id=job,reason='PANGU500M validated; stop obsolete serial AthenaK phase; independent AthenaK queued',UTC=datetime.datetime.utcnow().isoformat()+'Z')
   t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
  time.sleep(10)
