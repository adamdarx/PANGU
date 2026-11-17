"""Archive formal outputs; directly discard closed resource-test raw data."""
from pathlib import Path
import datetime,fcntl,json,os,subprocess,sys,time
H=Path(__file__).resolve().parent;name='20261117-single-coarse-convergence'
sys.path.insert(0,'/home/adamdarx/archive-local/ArchiveControl/20261114/runtime/validation/hpc/binary_stall_20261103')
import archive_qnm as worker
a=worker.archive;a.HERE=worker.HERE=H;worker.STAGING=H/'staging';worker.STAGING.mkdir(exist_ok=True)
a.BASE='/lustre/home/2601110206/CODEX_WORKPLACE/Operations';a.DEST=Path('/mnt/d/PANGU_TEST_DATA/PANGU_II_NR_DATA/Operations');a.ALLOWED={name};a.RECORDS=a.DEST/name/'archive-records';a.REMOTE_READ_ONLY=False;a.ZSTD='/home/adamdarx/.local/pangu-host/bin/zstd'
ssh=['ssh','-S','/home/adamdarx/.ssh/pku-data-5d-v1','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','2601110206@wm2-data.pku.edu.cn'];a.SSH=ssh;worker.SSH=ssh+['bash --noprofile --norc -s']
INVENTORY=r'''
from pathlib import Path
import json,time,re
base=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations');r=base/'20261117-single-coarse-convergence';rows=[]
for d in (r/'runs').glob('*/*'):
 if not d.is_dir():continue
 case=d.parent.name
 match=re.fullmatch(r'(stationary|spinning)_n(32|48|64)',case)
 if not match or not d.name.isdigit():continue
 mode=None;state=None
 own=d/'STATUS.json'
 if own.exists():
  v=json.loads(own.read_text());state=v.get('status','')
  if state.startswith('admission_'):mode='ADMISSION'
  elif state.startswith('formal_'):mode='FORMAL'
 for candidate in ('ADMISSION','FORMAL'):
  if mode is not None:break
  p=r/('%s_%s.json'%(candidate,case))
  if p.exists():
   v=json.loads(p.read_text())
   if v['job_id']==d.name:mode=candidate;state=v['status'];break
 if mode is None:continue
 complete=state not in ('admission_running','formal_running')
 if mode=='ADMISSION' and not complete:continue
 output=d/'output'
 for suffix in ('phdf','rhdf'):
  files=list(output.glob('*.'+suffix))
  numbered=sorted(p.name for p in files if re.search(r'\.\d{5}\.'+suffix+'$',p.name))
  latest=set(numbered[-2:])
  for p in files:
   s=p.stat()
   if p.is_symlink() or time.time()-s.st_mtime<(120 if mode=='ADMISSION' else 600) or s.st_uid!=71767:continue
   m=re.search(r'\.(\d{5})\.'+suffix+'$',p.name)
   # Final data need a completed process; formal initial and newest two stay on HPC.
   if not m and not complete:continue
   remove=mode=='ADMISSION' and complete
   if mode=='FORMAL' and m and int(m[1])>0 and p.name not in latest:remove=True
   rows.append(dict(path=str(p.relative_to(base)),absolute_source=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,remove_after_verified=remove,direct_discard=(mode=='ADMISSION' and complete),phase=mode,physical_acceptance='pending'))
 if complete and mode=='FORMAL':
  for p in d.rglob('*'):
   if not p.is_file() or p.is_symlink() or p.suffix in ('.phdf','.rhdf') or p.name.endswith('.archive.json'):continue
   st=p.stat()
   if st.st_uid!=71767 or time.time()-st.st_mtime<600:continue
   rows.append(dict(path=str(p.relative_to(base)),absolute_source=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,remove_after_verified=False,phase=mode,physical_acceptance='pending'))
rows.sort(key=lambda v:v['mtime_ns']);print(json.dumps(rows))
'''
# Regular expressions in the remote raw string use one literal backslash.
INVENTORY=INVENTORY.replace('\\\\','\\')
with (H/'archive.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (H/'archive.pid').write_text(str(os.getpid())+'\n')
 p=H/'live-archive-index.json';index=json.loads(p.read_text()) if p.exists() else dict(files={})
 while True:
  try:
   assert a.archive_available(),'Archive drive unavailable; retain server sources'
   entries=json.loads(subprocess.check_output(ssh+['python3 -'],input=INVENTORY.encode(),timeout=90))
   for e in entries:
    if e.get('direct_discard'):
     code="from pathlib import Path\nimport json,stat\np=Path("+repr(e['absolute_source'])+")\nassert p.is_relative_to(Path("+repr(a.BASE+'/'+name+'/runs')+")) if hasattr(p,'is_relative_to') else str(p).startswith("+repr(a.BASE+'/'+name+'/runs/')+")\nassert not any(q.is_symlink() for q in [p]+list(p.parents))\ns=p.stat()\nassert stat.S_ISREG(s.st_mode) and s.st_uid==71767 and s.st_size=="+repr(e['bytes'])+" and s.st_mtime_ns=="+repr(e['mtime_ns'])+"\np.unlink()\n"
     subprocess.check_output(ssh+['python3 -'],input=code.encode(),timeout=90)
     index.setdefault('discarded_tests',{})[e['path']]=dict(bytes=e['bytes'],reason='Closed resource admission raw data; diagnostics retained; user requests direct deletion',removed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());a.save(index)
     continue
    old=index['files'].get(e['path'])
    if old and old.get('server_removed'):continue
    if old and not e['remove_after_verified'] and Path(old['archive']).exists():continue
    a.transfer(e,index)
   index.update(status='watching',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid());index.pop('error',None);a.save(index)
  except BaseException as e:
   if isinstance(e,KeyboardInterrupt):raise
   index.update(status='retry_pending',error=repr(e));a.save(index);print(repr(e),flush=True)
  time.sleep(60)
