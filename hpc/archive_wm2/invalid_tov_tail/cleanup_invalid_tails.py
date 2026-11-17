"""Remove only archived, proven invalid TOV tails while their archive owner is idle."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,signal,subprocess
R=Path(__file__).resolve().parent;home=R.parent
SSH=['ssh','-S','/home/adamdarx/.ssh/pku-data-5d-v1','-o','ControlMaster=no','-o','BatchMode=yes','2601110206@wm2-data.pku.edu.cn']
base='/lustre/home/2601110206/CODEX_WORKPLACE/Operations/'
owner=home/'archive-queued-20261117';pid=int((owner/'archive.pid').read_text())
assert 'hrtimer_nanosleep' in Path('/proc/%d/wchan'%pid).read_text(),'Owner is busy; leave it running'
audit=json.loads((R/'ARCHIVED_FIELD_AUDIT.json').read_text())
assert all(audit[k]['any_nonfinite_cells']==audit[k]['cells'] for k in ['n64_i00020','n96_i00016'])
source=r'''
from pathlib import Path
import json,re,time
r=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-athenak-grmhd-reference');rows=[]
for n,j,cut in [(64,'17371502',96.4),(96,'17371507',78.9)]:
 proof=json.loads((r/('TOV_FORMAL_N%d.json'%n)).read_text());assert proof['status']=='formal_evolution_completed_analysis_pending'
 for p in (r/'runs'/('tov_n%d'%n)/j).rglob('*'):
  m=re.search(r'\.(\d{5})\.(bin|rst)$',p.name)
  if p.is_file() and not p.is_symlink() and m and int(m[1])*(25 if m[2]=='rst' else 5)>=cut:
   s=p.stat();assert s.st_uid==71767 and time.time()-s.st_mtime>120
   rows.append(dict(n=n,path=str(p.relative_to(r.parent)),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
print(json.dumps(rows))
'''
receipt=dict(status='running',files=[],policy='Completed external TOV tails proven nonfinite across the domain; fresh archived D SHA and unchanged source SHA/stat before deletion; finite prefixes and last-good restart retained')
def save():
 p=R/'INVALID_TAIL_CLEANUP.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(p)
os.kill(pid,signal.SIGTERM)
try:
 with (owner/'archive.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  rows=json.loads(subprocess.check_output(SSH+['python3 -'],input=source.encode(),timeout=120))
  journals={64:json.loads((home/'archive-athenak-tov64-20261117/live-archive-index.json').read_text())['files'],96:json.loads((owner/'live-archive-index.json').read_text())['files']}
  save()
  for row in rows:
   proof=journals[row['n']][row['path']];assert proof['bytes']==row['bytes']
   archived=Path(proof['archive']);assert archived.is_file();h=hashlib.sha256()
   p=subprocess.Popen(['/home/adamdarx/.local/pangu-host/bin/zstd','-d','-q','-c',str(archived)],stdout=subprocess.PIPE)
   for chunk in iter(lambda:p.stdout.read(8388608),b''):h.update(chunk)
   assert p.wait()==0 and h.hexdigest()==proof['sha256']
   code='''from pathlib import Path
import hashlib,stat
p=Path(%r)
assert not any(q.is_symlink() for q in [p]+list(p.parents))
s=p.stat();assert stat.S_ISREG(s.st_mode) and s.st_uid==71767 and s.st_size==%r and s.st_mtime_ns==%r
h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
assert h.hexdigest()==%r
s2=p.stat();assert (s.st_size,s.st_mtime_ns)==(s2.st_size,s2.st_mtime_ns)
p.unlink()
'''%(base+row['path'],row['bytes'],row['mtime_ns'],proof['sha256'])
   subprocess.check_output(SSH+['python3 -'],input=code.encode(),timeout=180)
   row.update(archive=str(archived),sha256=proof['sha256'],removed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());receipt['files'].append(row);save()
  receipt['status']='completed';receipt['raw_logical_bytes_removed']=sum(x['bytes'] for x in receipt['files']);save()
finally:
 with (owner/'archive.log').open('ab') as f:p=subprocess.Popen(['/home/adamdarx/.local/pangu-host/bin/python',str(owner/'watch.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 meta=json.loads((owner/'OWNER.json').read_text());meta['pid']=p.pid;(owner/'OWNER.json').write_text(json.dumps(meta));receipt['resumed_archive_owner_pid']=p.pid;save()
print(json.dumps({'status':receipt['status'],'files':len(receipt['files']),'raw_logical_bytes_removed':receipt.get('raw_logical_bytes_removed'),'resumed_pid':p.pid}))
