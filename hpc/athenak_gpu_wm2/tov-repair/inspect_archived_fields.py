"""Locate finite/positivity failures in SHA-verified, archived AthenaK fields."""
from pathlib import Path
import gc,hashlib,json,subprocess,sys
import numpy as np
R=Path(__file__).resolve().parent;cache=R/'field-check';cache.mkdir(exist_ok=True)
sys.path.insert(0,'/home/adamdarx/pangu-setup/tov-reference-20261117/athenak-source/vis/python')
from bin_convert import read_binary
ZSTD='/home/adamdarx/.local/pangu-host/bin/zstd'
result={}
for n,indices,owner in [(64,[19,20],'archive-athenak-tov64-20261117'),(96,[15,16],'archive-queued-20261117')]:
 journal=json.loads((R.parent/owner/'live-archive-index.json').read_text())['files']
 for index in indices:
  suffix='n%d.z4c.%05d.bin'%(n,index);key=next(k for k in journal if k.endswith(suffix));proof=journal[key];p=cache/('n%d-i%05d.bin'%(n,index))
  if not p.exists():subprocess.run([ZSTD,'-d','-q','-f',proof['archive'],'-o',str(p)],check=True)
  with p.open('rb') as f:
   h=hashlib.sha256()
   for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
  assert h.hexdigest()==proof['sha256']
  data=read_binary(str(p));fields={k:np.asarray(v) for k,v in data['mb_data'].items()};chi=fields['z4c_chi'];alpha=fields['z4c_alpha'];geo=np.asarray(data['mb_geometry']);shape=chi.shape
  centers=[]
  for d in range(3):
   cells=shape[3-d];a=geo[:,2*d];width=(geo[:,2*d+1]-a)/cells;v=a[:,None]+(np.arange(cells)+.5)*width[:,None]
   centers.append(v.reshape((len(geo),)+(1,)*(2-d)+(cells,)+(1,)*d))
  rr=sum(x*x for x in centers);bad=np.zeros(shape,dtype=bool)
  for values in fields.values():bad|=~np.isfinite(values)
  g=[fields['z4c_'+name].astype(float) for name in ['gxx','gxy','gxz','gyy','gyz','gzz']]
  with np.errstate(invalid='ignore',over='ignore'):
   minor=g[0]*g[3]-g[1]**2
   det=g[0]*(g[3]*g[5]-g[4]**2)-g[1]*(g[1]*g[5]-g[2]*g[4])+g[2]*(g[1]*g[4]-g[2]*g[3])
   spd_bad=(g[0]<=0)|(minor<=0)|(det<=0)|~np.isfinite(det)
  shells={}
  for lo,hi in [(0,2),(2,5),(5,10),(10,30),(30,1000)]:
   mask=(rr>=lo*lo)&(rr<hi*hi);count=int(mask.sum())
   def limits(a):
    v=a[mask];v=v[np.isfinite(v)]
    return [float(v.min()),float(v.max())] if len(v) else None
   shells['%g<=r<%g'%(lo,hi)]=dict(cells=count,any_nonfinite_cells=int((bad&mask).sum()),nonpositive_chi_cells=int(((chi<=0)&mask).sum()),nonpositive_lapse_cells=int(((alpha<=0)&mask).sum()),conformal_metric_nonSPD_cells=int((spd_bad&mask).sum()),chi_range=limits(chi),lapse_range=limits(alpha))
  record=dict(n=n,index=index,time=float(data['time']),source_sha256=proof['sha256'],source_archive=proof['archive'],cells=int(np.prod(shape)),any_nonfinite_cells=int(bad.sum()),shells=shells,scope='Geometric radial shells, not an apparent-horizon mask. Nonfinite global histories alone do not establish exterior failure; these fields directly locate it.')
  result['n%d_i%05d'%(n,index)]=record;(R/'ARCHIVED_FIELD_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(record),flush=True)
  del data,fields,chi,alpha,g,rr,bad,mask,spd_bad,minor,det
  gc.collect()
