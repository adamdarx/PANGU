"""Load archive-priority changes only between complete validation/transfer cycles."""
from pathlib import Path
import datetime,fcntl,json,os,signal,subprocess,time
R=Path(__file__).resolve().parent;state=R/'archive-state';record=R/'PRIORITY_RELOAD.json'
with (R/'reload.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 old=int((state/'watch.pid').read_text());v=dict(status='waiting_for_archive_idle',old_pid=old,policy='Do not interrupt validation/transfer; restart the same journal owner at its normal sleep; evolution jobs untouched')
 record.write_text(json.dumps(v,indent=2))
 while Path('/proc/%d'%old).exists():
  try:
   command=Path('/proc/%d/cmdline'%old).read_bytes()
   if not command:break
   assert str(R/'watch_archive.py').encode() in command
   children=Path('/proc/%d/task/%d/children'%(old,old)).read_text().split()
   idle='hrtimer_nanosleep' in Path('/proc/%d/wchan'%old).read_text()
  except FileNotFoundError:break
  if idle and not children:break
  time.sleep(10)
 if Path('/proc/%d'%old).exists():os.kill(old,signal.SIGTERM)
 with (state/'watch.lock').open('a') as owner_lock:
  fcntl.flock(owner_lock,fcntl.LOCK_EX)
  # Lock ownership proves no second worker is writing this journal.
 with (R/'archive-watch.log').open('ab') as log:p=subprocess.Popen(['/home/adamdarx/.local/pangu-host/bin/python','-u',str(R/'watch_archive.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 v.update(status='archive_priority_reload_started',new_pid=p.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());record.write_text(json.dumps(v,indent=2));print(json.dumps(v))
