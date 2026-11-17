"""Archive closed ET formal runs; short-test raw output is outside this owner."""
from pathlib import Path
import datetime,fcntl,json,os,subprocess,sys,time
H=Path(__file__).resolve().parent
sys.path.insert(0,'/home/adamdarx/archive-local/ArchiveControl/20261114/runtime/validation/hpc/binary_stall_20261103')
import archive_qnm as w
a=w.archive;ssh=['ssh','-S','/home/adamdarx/.ssh/pku-data-5d-v1','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','2601110206@wm2-data.pku.edu.cn'];a.SSH=ssh;w.SSH=ssh+['bash --noprofile --norc -s'];a.HERE=w.HERE=H;w.STAGING=H/'staging';w.STAGING.mkdir(exist_ok=True)
a.BASE='/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu';a.DEST=Path('/mnt/d/PANGU_TEST_DATA/PANGU_II_NR_DATA/Operations/20261117-einstein-toolkit-gpu');a.ALLOWED={'native-tov-reference','vacuum-reference'};a.RECORDS=a.DEST/'formal-archive-records';a.REMOTE_READ_ONLY=True;a.ZSTD='/home/adamdarx/.local/pangu-host/bin/zstd'
SOURCE=r'''
from pathlib import Path
import json,time
r=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu');proofs=[]
for p in (r/'native-tov-reference').glob('N*_FORMAL.json'):proofs.append(p)
for p in (r/'vacuum-reference').glob('FORMAL_*.json'):proofs.append(p)
rows=[]
for proof in proofs:
 d=json.loads(proof.read_text())
 if d.get('status') not in ['formal_completed_quantitative_analysis_pending','formal_completed_analysis_pending']:continue
 folder=Path(d['work']);assert folder.is_relative_to(r) if hasattr(folder,'is_relative_to') else str(folder).startswith(str(r)+'/');assert '/runs/' in str(folder)
 for p in list(folder.rglob('*'))+[proof]:
  if p.is_symlink() or not p.is_file() or p.name.endswith('.archive.json'):continue
  s=p.stat()
  if s.st_uid!=71767 or time.time()-s.st_mtime<120:continue
  rows.append(dict(path=str(p.relative_to(r)),bytes=s.st_size,mtime_ns=s.st_mtime_ns,remove_after_verified=False,closed_phase=True,physical_acceptance='pending independent quantitative analysis'))
print(json.dumps(rows))
'''
with (H/'archive.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(H/'archive.pid').write_text(str(os.getpid()));p=H/'live-archive-index.json';index=json.loads(p.read_text()) if p.exists() else dict(files={})
 while True:
  try:
   entries=json.loads(subprocess.check_output(ssh+['python3 -'],input=SOURCE.encode(),timeout=120))
   for e in entries:
    old=index['files'].get(e['path'])
    if old and old.get('server_removed'):continue
    if not old:a.transfer(e,index);old=index['files'][e['path']]
    if e['bytes']>=64*1024**2 and '/runs/' in e['path']:
     assert a.archive_digest(Path(old['archive']))==old['sha256']
     src=a.BASE+'/'+e['path'];code='from pathlib import Path\nimport hashlib,stat\np=Path('+repr(src)+')\nassert not any(q.is_symlink() for q in [p]+list(p.parents))\ns=p.stat();assert stat.S_ISREG(s.st_mode) and s.st_uid==71767 and s.st_size=='+repr(e['bytes'])+' and s.st_mtime_ns=='+repr(e['mtime_ns'])+'\nh=hashlib.sha256()\nwith p.open("rb") as f:\n for c in iter(lambda:f.read(8388608),b""):h.update(c)\nassert h.hexdigest()=='+repr(old['sha256'])+'\ns2=p.stat();assert (s.st_size,s.st_mtime_ns)==(s2.st_size,s2.st_mtime_ns)\np.unlink()\n'
     subprocess.check_output(ssh+['python3 -'],input=code.encode(),timeout=900);old['server_removed']=True;a.save(index)
   index.update(status='watching_completed_formal_phases',pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());index.pop('error',None);a.save(index)
  except Exception as e:index.update(status='retry_pending',error=repr(e));a.save(index);print(repr(e),flush=True)
  time.sleep(60)
