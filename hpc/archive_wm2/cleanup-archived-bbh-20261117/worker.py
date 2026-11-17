"""Unique owner of frozen historical manifest; verify archives before cleanup."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shlex,subprocess,sys,time
H=Path(__file__).resolve().parent
if (H/'STOP').exists(): raise SystemExit('Historical blanket archival disabled by latest user policy')
sys.path.insert(0,'/home/adamdarx/archive-local/ArchiveControl/20261114/runtime/validation/hpc/binary_stall_20261103')
import archive_qnm as w
a=w.archive
ssh=['ssh','-S','/home/adamdarx/.ssh/pku-data-5d-v1','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','2601110206@wm2-data.pku.edu.cn']
login=ssh.copy();login[2]='/home/adamdarx/.ssh/pku-login-5d-v1';login[-1]='2601110206@wm2-login.pku.edu.cn'
a.HERE=w.HERE=H;w.STAGING=H/'staging';w.STAGING.mkdir(exist_ok=True)
a.BASE='/lustre/home/2601110206/CODEX_WORKPLACE';a.DEST=Path('/mnt/d/PANGU_TEST_DATA/PANGU_II_NR_DATA');a.ALLOWED={'Matter','BinaryPuncture','SinglePuncture','ApplesWithApples','Performance','BinaryPlasma','Operations'}
a.RECORDS=a.DEST/'ArchiveRecords/cleanup-archived-bbh-20261117';a.REMOTE_READ_ONLY=True;a.ZSTD='/home/adamdarx/.local/pangu-host/bin/zstd';a.SSH=ssh;w.SSH=ssh+['bash --noprofile --norc -s']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def remote(code):return subprocess.check_output(ssh+['python3 -'],input=code.encode(),timeout=900)
def protected():
 code="import subprocess,re,json\nids=subprocess.check_output(['squeue','-h','-u','2601110206','-o','%i']).decode().split()\na=[]\nfor j in ids:\n s=subprocess.check_output(['scontrol','show','job','-o',j]).decode()\n a.extend(re.findall(r'(?:Command|WorkDir)=(\\S+)',s))\nprint(json.dumps(a))"
 return [x for x in json.loads(subprocess.check_output(login+['python3 -'],input=code.encode(),timeout=90)) if x.startswith(a.BASE+'/')]
def guard(entry,sha=None,remove=False):
 source=a.BASE+'/'+entry['path']
 code='from pathlib import Path\nimport os,stat,hashlib,json,time\np=Path('+repr(source)+')\ns=p.lstat()\nassert stat.S_ISREG(s.st_mode) and s.st_uid==71767\nassert not any(q.is_symlink() for q in [p]+list(p.parents))\nassert s.st_size=='+repr(entry['bytes'])+' and s.st_mtime_ns=='+repr(entry['mtime_ns'])+'\nassert time.time()-s.st_mtime>3600\nh=hashlib.sha256()\nwith p.open("rb") as f:\n for chunk in iter(lambda:f.read(8*1024**2),b""):h.update(chunk)\ns2=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(s2.st_ino,s2.st_size,s2.st_mtime_ns)\n'
 if sha:code+='assert h.hexdigest()=='+repr(sha)+'\n'
 if remove:
  code+='p.with_name(p.name+".archive.json").write_text('+repr(json.dumps(dict(archive=str(a.DEST/(entry['path']+'.zst')),sha256=sha,bytes=entry['bytes'],owner=str(H))))+')\np.unlink()\n'
 code+='print(json.dumps(dict(sha256=h.hexdigest(),removed='+repr(remove)+')))\n'
 return json.loads(remote(code))
with (H/'archive.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(H/'archive.pid').write_text(str(os.getpid()))
 inv=json.loads((H/'INVENTORY.json').read_text());entries=inv['entries'];assert entries
 index=json.loads((H/'live-archive-index.json').read_text()) if (H/'live-archive-index.json').exists() else dict(files={})
 # Preserve the newest two native restart files in each historical output directory.
 restart={}
 for e in entries:
  p=Path(e['path'])
  if p.suffix in ('.rhdf','.rst'):restart.setdefault(str(p.parent),[]).append(e)
 keep={e['path'] for values in restart.values() for e in sorted(values,key=lambda x:x['mtime_ns'])[-2:]}
 entries.sort(key=lambda e:(not e['large_output'],not (a.DEST/(e['path']+'.zst')).exists(),-e['bytes']))
 index.update(status='running',pid=os.getpid(),inventory_files=len(entries),inventory_raw_bytes=inv['raw_bytes']);a.save(index)
 cache=[];last=0;errors=[]
 for e in entries:
  assert any(e['path'].startswith(r) for r in ['BinaryPuncture/20261103-n32-three-dt5/'])
  if (H/'STOP').exists(): break
  try:
   if time.monotonic()-last>60:cache=protected();last=time.monotonic()
   source=a.BASE+'/'+e['path']
   if any(source==x or source.startswith(x.rstrip('/')+'/') or (x.startswith(a.BASE+'/') and source.startswith(str(Path(x).parent)+'/')) for x in cache):continue
   old=index['files'].get(e['path'])
   if old and old.get('server_removed'):continue
   target=a.DEST/(e['path']+'.zst')
   index.update(current=e['path'],updated_utc=now());a.save(index)
   before=guard(e);verified_now=False
   if not old and target.exists():
    assert a.archive_digest(target)==before['sha256'],'Existing D copy differs; retain both for inspection'
    verified_now=True
    old=dict(e,sha256=before['sha256'],compressed_bytes=target.stat().st_size,archive=str(target),verified=now(),server_removed=False,adopted_existing_archive=True)
    index['files'][e['path']]=old;assert a.save(index)
   elif not old:
    raise RuntimeError('Existing verified D copy required; source retained without new archival')
   assert old['sha256']==before['sha256']
   # Always re-read destination immediately before this worker's own cleanup.
   if e['large_output'] and e['path'] not in keep:
    assert a.archive_available()
    if not verified_now:assert a.archive_digest(target)==old['sha256']
    guard(e,old['sha256'],True);old['server_removed']=True;old['removed_utc']=now();assert a.save(index)
    print('REMOVED_VERIFIED_HISTORICAL',e['path'],e['bytes'],flush=True)
   else:print('VERIFIED_RETAINED',e['path'],flush=True)
  except Exception as err:
   errors.append(dict(path=e['path'],error=repr(err)));print('RETAINED_ON_ERROR',e['path'],repr(err),flush=True)
   index['errors']=errors[-100:];a.save(index)
 index.update(status='completed_with_retained_errors' if errors else 'completed',completed_utc=now(),errors=errors);a.save(index)
