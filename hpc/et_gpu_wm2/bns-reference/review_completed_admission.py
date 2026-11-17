"""Review an already completed CUDA trajectory after fixing an exact-zero check.

The failed Slurm wrapper state is preserved. No time evolution is repeated.
"""
from pathlib import Path
import csv,datetime,hashlib,json,re,sys
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent))
from reference_history_guard import check_norms
n=64;job='17374252';receipt=R/'N64_ADMISSION.json';old=json.loads(receipt.read_text())
assert old['job_id']==job and old['status']=='admission_failed'
assert old['error']=="AssertionError('Initial model is not unmagnetized and from rest')"
assert old['returncode']==0 and old['actual_GPU_process_observed']
work=Path(old['work']);text=(work/'run.log').read_text(errors='replace');assert 'Done.' in text
build=json.loads((R/'BUILD_STATUS.json').read_text());case=next(c for c in json.loads((R/'PLAN.json').read_text())['cases'] if c['n']==n)
for p,sha in [(Path(build['binary']),old['binary_sha256']),(R/case['inputs']['admission']['file'],old['input_sha256']),(Path(case['table']),old['table_sha256'])]:assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
check_norms(work,full=True)
def norm(name):
 p=next(work.rglob(name+'.tsv'));header=p.read_text().splitlines()[0];cols={k:int(i)-1 for i,k in re.findall(r'(\d+):([^\s\\]+)',header)};d=np.atleast_2d(np.loadtxt(p));assert np.isfinite(d).all();return d,cols
d,c=norm('asterx-cons_vector');rho,rc=norm('hydrobasex-rho');mag,mc=norm('asteranalysis-mhd_analysis');vel,vc=norm('hydrobasex-vel');bvec,bc=norm('hydrobasex-bvec');hc,hcols=norm('z4c-hc')
mass=d[:,c['asterx::dens.sum']];density=rho[0,rc['hydrobasex::rho.max']];energy=mag[0,mc['asteranalysis::magnetic_energy.sum']];b=max(abs(bvec[0,i]) for k,i in bc.items() if k.endswith('.maxabs'));v=max(abs(vel[0,i]) for k,i in vc.items() if k.endswith('.maxabs'))
epsilon=np.finfo(float).eps;btol=64*epsilon*np.sqrt(density);etol=64*epsilon**2*mass[0]
assert int(d[-1,0])==32 and abs(d[-1,1]-.1)<1e-12
assert b<=btol and 0<=energy<=etol and v<1e-12
ref=json.loads((R/'INITIAL_REFERENCE.json').read_text())['64'];assert abs(mass[0]/ref['baryon_mass']-1)<.01 and abs(density/ref['rho_max']-1)<.05
assert np.max(abs(mass/mass[0]-1))<1e-4
for rank in range(1):
 rows=[(float(x[0]),float(x[1])) for x in csv.reader((work/('gpu-memory-%d.csv'%rank)).read_text().splitlines()) if len(x)==2];assert rows
 peak=max(x[0] for x in rows);total=rows[-1][1];assert peak<.8*total and total-peak>8192
 assert Path(build['binary']).name in (work/('gpu-apps-%d.csv'%rank)).read_text()
samples=[];t=None
for line in text.splitlines():
 a=re.search(r'Simulation time:\s*([0-9.eE+-]+)',line)
 if a:t=float(a[1])
 a=re.search(r'total evolution time:\s*([0-9.eE+-]+) sec',line)
 if a and t is not None and .025<=t<.1:samples.append((t,float(a[1])))
assert len(samples)>=8 and samples[-1][0]>samples[0][0]
hours=(samples[-1][1]-samples[0][1])*200/(samples[-1][0]-samples[0][0])/3600;assert hours<110
previous=R/'N64_ADMISSION.before_roundoff_review.json';assert not previous.exists();previous.write_text(json.dumps(old,indent=2))
old.update(status='admission_passed',projected200M_hours=hours,posthoc_review={'scope':'Completed solver exit0 CUDA trajectory rechecked; original Slurm wrapper remains FAILED due to the exact-zero diagnostic check, with all original logs retained','previous_receipt_sha256':hashlib.sha256(previous.read_bytes()).hexdigest(),'source_norm_sha256':{name:hashlib.sha256(next(work.rglob(name+'.tsv')).read_bytes()).hexdigest() for name in ['asterx-cons_vector','hydrobasex-rho','hydrobasex-bvec','asteranalysis-mhd_analysis','z4c-hc']},'projection_basis':'Native accumulated evolution timers, t0.025..0.096875; setup/final I/O and long-time cost not certified','reviewed_utc':datetime.datetime.utcnow().isoformat()+'Z'},limitations='32-step technical and initial-state acceptance only; long-time physical comparison pending.')
old.pop('error');old['measurements'].update(initial_B_maxabs=float(b),magnetic_roundoff_B_tolerance=float(btol),magnetic_roundoff_energy_tolerance=float(etol))
for p in [receipt,work/'STATUS.json']:
 q=p.with_suffix('.tmp');q.write_text(json.dumps(old,indent=2,allow_nan=False));q.replace(p)
(R/'N64_ROUNDOFF_REVIEW.json').write_text(json.dumps(old,indent=2));print(json.dumps(old,indent=2))
