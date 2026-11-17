"""Delete only an explicitly frozen obsolete test-output manifest."""
from pathlib import Path
import json,os,stat,datetime,sys
base=Path('/lustre/home/2601110206/CODEX_WORKPLACE')
plan=json.loads(Path(sys.argv[1]).read_text());active=json.loads(Path(sys.argv[2]).read_text())
allowed={'Operations/20261114-single-2000M', 'Operations/20261114-bns-n96-romio-gpu'}
assert set(plan['roots'])==allowed and os.getuid()==71767
for x in active:
 for r in allowed:assert not x.startswith(str(base/r)+'/') and x!=str(base/r),(x,r)
receipt=dict(policy=plan['policy'],scope=plan['scope'],removed=[],skipped=[],archive_required=False)
for e in plan['entries']:
 p=base/e['path']
 try:
  assert '/'.join(Path(e['path']).parts[:2]) in allowed
  assert '..' not in Path(e['path']).parts and not Path(e['path']).is_absolute()
  assert p.suffix.lower() in {'.phdf','.rhdf','.h5','.hdf5','.bin','.rst','.vtk','.vtu','.npz'}
  assert not any(q.is_symlink() for q in [p]+list(p.parents))
  s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==71767 and not s.st_mode&0o111
  assert s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns'],'Changed since inventory: retain'
  p.unlink();receipt['removed'].append(dict(path=e['path'],bytes=s.st_size))
 except Exception as err:receipt['skipped'].append(dict(path=e['path'],reason=repr(err)))
receipt.update(removed_bytes=sum(e['bytes'] for e in receipt['removed']),completed_utc=datetime.datetime.utcnow().isoformat()+'Z')
print(json.dumps(receipt))
