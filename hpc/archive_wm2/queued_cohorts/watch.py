"""Own closed outputs only for the newly queued reference/BBH cohorts."""
from pathlib import Path
import datetime,fcntl,json,os,subprocess,sys,time
H=Path(__file__).resolve().parent
sys.path.insert(0,'/home/adamdarx/archive-local/ArchiveControl/20261114/runtime/validation/hpc/binary_stall_20261103')
import archive_qnm as w
a=w.archive;ssh=['ssh','-S','/home/adamdarx/.ssh/pku-data-5d-v1','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','2601110206@wm2-data.pku.edu.cn'];a.SSH=ssh;w.SSH=ssh+['bash --noprofile --norc -s'];a.HERE=w.HERE=H;w.STAGING=H/'staging';w.STAGING.mkdir(exist_ok=True)
a.BASE='/lustre/home/2601110206/CODEX_WORKPLACE/Operations';a.DEST=Path('/mnt/d/PANGU_TEST_DATA/PANGU_II_NR_DATA/Operations');a.ALLOWED={'20261117-single-athenak-parallel','20261117-bbh-convergence','20261117-athenak-grmhd-reference'};a.RECORDS=Path('/mnt/d/PANGU_TEST_DATA/PANGU_II_NR_DATA/ArchiveRecords/20261117-queued-cohorts');a.REMOTE_READ_ONLY=False;a.ZSTD='/home/adamdarx/.local/pangu-host/bin/zstd'
SOURCE=r'''
from pathlib import Path
import json,re,time
base=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations');scopes=[]
def load(p):return json.loads(p.read_text()) if p.exists() else {}
r=base/'20261117-single-athenak-parallel';v=load(r/'SUBMISSIONS.json')
for mode in ['admission','formal']:
 for key,j in v.get(mode,{}).items():scopes.append((r/'runs'/key/j,mode,load(r/('%s_%s.json'%(mode.upper(),key))),25,25))
r=base/'20261117-athenak-grmhd-reference';v=load(r/'BNS_SUBMISSIONS.json')
for mode in ['admission','formal']:
 for n,j in v.get(mode,{}).items():scopes.append((r/'runs'/('bns_n'+n)/j,mode,load(r/('BNS_%s_N%s.json'%(mode.upper(),n))),5,50))
# Existing TOV64 has a separate completed archive owner; only TOV96 is added here.
v=load(r/'TOV_SUBMISSIONS.json');j=v.get('formal',{}).get('96')
if j:scopes.append((r/'runs/tov_n96'/j,'formal',load(r/'TOV_FORMAL_N96.json'),5,25))
r=base/'20261117-bbh-convergence';v=load(r/'FORMAL_SUBMISSIONS.json')
for n,j in v.get('formal',{}).items():scopes.append((r/'runs'/('n'+n)/j,'formal',load(r/('FORMAL_N'+n+'.json')),5,10.25))
j=v.get('full_io_gate')
if j:
 for n in [32,48,64]:scopes.append((r/'runs'/('n%d'%n)/j,'admission',load(r/'FULL_IO_ADMISSION.json'),5,10.25))
rows=[]
for folder,mode,proof,dt,rdt in scopes:
 if not folder.exists():continue
 status=proof.get('status','');complete=status.startswith('completed') or status=='formal_evolution_completed_analysis_pending';passed=status in ('admission_passed','passed')
 if mode=='admission' and not passed:continue
 log=folder/'run.log';s=''
 if log.exists():
  with log.open('rb') as f:f.seek(max(0,log.stat().st_size-8000));s=f.read().decode(errors='replace')
 times=re.findall(r'(?<![a-zA-Z_])time\s*=\s*([0-9.eE+\-]+)',s);latest=float(times[-1]) if times else 0
 files=[p for p in folder.rglob('*') if p.is_file() and not p.is_symlink() and not p.name.endswith('.archive.json')];groups={}
 for p in files:
  m=re.search(r'\.(\d{5})(?:\.(\d{5}))?\.(bin|rst|phdf|rhdf)$',p.name)
  if m:groups.setdefault((str(p.parent),p.name[:m.start()],m[2],m[3]),[]).append(p.name)
 keep={name for values in groups.values() for name in sorted(values)[-2:]}
 for p in files:
  st=p.stat()
  if st.st_uid!=71767 or time.time()-st.st_mtime<120:continue
  field=p.suffix in ('.bin','.rst','.phdf','.rhdf');m=re.search(r'\.(\d{5})(?:\.(\d{5}))?\.(bin|rst|phdf|rhdf)$',p.name)
  if mode=='admission':
   if field:rows.append(dict(path=str(p.relative_to(base)),bytes=st.st_size,mtime_ns=st.st_mtime_ns,direct_discard=True,remove_after_verified=False))
   continue
  if not complete:
   if not m or time.time()-st.st_mtime<600:continue
   stamp=int(m[1])*(rdt if p.suffix in ('.rst','.rhdf') else dt)
   if latest<=stamp+1:continue
  remove=bool(field and m and int(m[1])>0 and p.name not in keep)
  rows.append(dict(path=str(p.relative_to(base)),bytes=st.st_size,mtime_ns=st.st_mtime_ns,remove_after_verified=remove,physical_acceptance='pending',closure_basis='Completed phase or immutable numbered output with writer>nominal stamp+1 and age>600s'))
rows.sort(key=lambda e:(not e.get('direct_discard'),e['mtime_ns']));print(json.dumps(rows))
'''
with (H/'archive.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(H/'archive.pid').write_text(str(os.getpid()));p=H/'live-archive-index.json';index=json.loads(p.read_text()) if p.exists() else dict(files={})
 while True:
  try:
   entries=json.loads(subprocess.check_output(ssh+['python3 -'],input=SOURCE.encode(),timeout=180))
   for e in entries:
    if e.get('direct_discard'):
     src=a.BASE+'/'+e['path'];code='from pathlib import Path\nimport stat\np=Path('+repr(src)+')\nassert not any(q.is_symlink() for q in [p]+list(p.parents))\ns=p.stat();assert stat.S_ISREG(s.st_mode) and s.st_uid==71767 and s.st_size=='+repr(e['bytes'])+' and s.st_mtime_ns=='+repr(e['mtime_ns'])+'\np.unlink()\n'
     subprocess.check_output(ssh+['python3 -'],input=code.encode(),timeout=90);index.setdefault('discarded_tests',{})[e['path']]=dict(bytes=e['bytes'],reason='Accepted temporary GPU test raw output; no archival requested');a.save(index);continue
    old=index['files'].get(e['path'])
    if old and old.get('server_removed'):continue
    if old and not e['remove_after_verified'] and Path(old['archive']).exists():continue
    a.transfer(e,index)
   index.update(status='watching',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid());index.pop('error',None);a.save(index)
  except Exception as e:index.update(status='retry_pending',error=repr(e));a.save(index);print(repr(e),flush=True)
  time.sleep(60)
