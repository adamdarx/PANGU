"""Read-only inventory of historical output, excluding live archive scopes."""
from pathlib import Path
import os,json,time,stat
R=Path('/lustre/home/2601110206/CODEX_WORKPLACE')
exclude={'SinglePuncture/20261030-spin-accuracy-rerun','Operations/20261116-matter-n128','Operations/20261117-single-coarse-convergence','Operations/20261117-athenak-grmhd-reference','Operations/20261117-einstein-toolkit-gpu','Operations/20261117-bbh-convergence','Operations/20261117-single-athenak-parallel','Operations/20261116-spherical-nonpg'}
# Coordination and executable/build trees are outside this historical result sweep.
tops=['Matter','BinaryPuncture','SinglePuncture','ApplesWithApples','Performance','BinaryPlasma','Operations']
prune={'build','builds','source','src','.git','bin','lib','lib64','include','node_modules','__pycache__','staging','archives','archive-records'}
fields={'.phdf','.rhdf','.h5','.hdf5','.bin','.rst','.vtk','.vtu','.npz'}
meta={'.json','.txt','.log','.csv','.hst','.in','.par','.slurm','.sh','.py','.md','.out','.err'}
entries=[];now=time.time()
for top in tops:
 for folder,ds,fs in os.walk(str(R/top),followlinks=False):
  rel=str(Path(folder).relative_to(R))
  if any(rel==x or rel.startswith(x+'/') for x in exclude):ds[:]=[];continue
  # bin directories under runs are numerical output; outer bin directories are programs.
  ds[:]=[d for d in ds if not (Path(folder)/d).is_symlink() and (d not in prune or (d=='bin' and '/runs/' in folder))]
  for name in fs:
   p=Path(folder)/name
   try:s=p.lstat()
   except FileNotFoundError:continue
   if not stat.S_ISREG(s.st_mode) or s.st_uid!=71767 or now-s.st_mtime<3600 or name.endswith('.archive.json'):continue
   suffix=p.suffix.lower();isfield=suffix in fields and s.st_size>=1024*1024 and not s.st_mode&0o111
   ismeta=suffix in meta and s.st_size<=16*1024*1024
   if not (isfield or ismeta):continue
   # Sweep old operation records, but do not guess closure of today's other operations.
   if top=='Operations' and Path(rel).parts[1].startswith('20261117'):continue
   entries.append(dict(path=str(p.relative_to(R)),bytes=s.st_size,mtime_ns=s.st_mtime_ns,large_output=isfield,remove_after_verified=False,physical_acceptance='not evaluated by byte archival',closure_basis='Historical scope; no active scheduler command roots; age>1h; repeated stat and hash required'))
print(json.dumps(dict(root=str(R),excluded=sorted(exclude),entries=entries,files=len(entries),raw_bytes=sum(x['bytes'] for x in entries))))
