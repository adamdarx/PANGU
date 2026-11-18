"""Spatially locate endpoint baryon accounting differences; no evolution."""
from pathlib import Path
import csv,hashlib,importlib.util,json,math,subprocess
import h5py,numpy as np
R=Path(__file__).resolve().parent;BASE=Path('/home/adamdarx/pangu-setup')
PA=Path('/mnt/d/PANGU_TEST_DATA/PANGU_II_NR_DATA/Matter/20261102-stellar-formation/runs/collapse/16902889/unstable_b1')
index=json.loads((BASE/'archive-et-formal-20261117/live-archive-index.json').read_text())['files']
key='20261117-athenak-grmhd-reference/tov-repair/runs/tov_n64/17374027/bin/athenak_tov_collapse_n64.mhd_u.00020.bin'
spec=importlib.util.spec_from_file_location('reader',str(BASE/'tov-reference-20261117/athenak-source/vis/python/bin_convert.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def restore(entry,name):
 p=R/'scratch'/name;h=hashlib.sha256();size=0
 with p.open('wb') as f:
  child=subprocess.Popen(['/home/adamdarx/.local/pangu-host/bin/zstd','-dc',entry['archive']],stdout=subprocess.PIPE)
  for block in iter(lambda:child.stdout.read(8388608),b''):f.write(block);h.update(block);size+=len(block)
  child.stdout.close();assert child.wait()==0
 assert h.hexdigest()==entry['sha256'] and size==entry['bytes'];return p
pe=json.loads((PA/'sync_grmhd_tov.sync_grmhd_tov.final.phdf.archive.json').read_text());ap=restore(index[key],'endpoint_ak.bin');pp=restore(pe,'endpoint_pa.phdf')
try:
 a=m.read_binary(str(ap));ad=np.asarray(a['mb_data']['dens'],dtype=float);ag=np.asarray(a['mb_geometry']);akeys=[(int(x[3]),*map(int,x[:3])) for x in a['mb_logical']]
 with h5py.File(pp) as f:
  assert float(f['Info'].attrs['Time'])==a['time']==100
  pd=f['mhd.cons'][:,0];active=f['nr.constraint_mask'][:,0]>0
  pg=np.stack([f['Locations/'+ax][:,s] for ax in 'xyz' for s in (0,-1)],axis=1);pkeys=[tuple(np.ravel(l).tolist()+np.ravel(x).tolist()) for l,x in zip(f['Levels'][:],f['LogicalLocations'][:])]
  idx={k:i for i,k in enumerate(akeys)};order=np.array([idx[k] for k in pkeys]);assert np.allclose(pg,ag[order],rtol=0,atol=1e-12);ad=ad[order]
  bins=[(0,1),(1,1.8),(1.8,3),(3,30),(30,float('inf'))];totals=np.zeros((len(bins),2));l1=np.zeros(len(bins));inactive=np.zeros(2)
  for b,lo in enumerate(pg):
   dx=(lo[1::2]-lo[::2])/16;x,y,z=[lo[2*j]+(np.arange(16)+.5)*dx[j] for j in range(3)];r2=x[None,None,:]**2+y[None,:,None]**2+z[:,None,None]**2;dv=np.prod(dx)
   for j,(r0,r1) in enumerate(bins):
    mask=(r2>=r0*r0)&(r2<r1*r1);totals[j]=totals[j]+np.array([pd[b][mask].sum(),ad[b][mask].sum()])*dv;l1[j]+=float(np.abs(pd[b][mask]-ad[b][mask]).sum())*dv
   inactive+=np.array([pd[b][~active[b]].sum(),ad[b][~active[b]].sum()])*dv
 rows=[dict(r_min=a,r_max=b if math.isfinite(b) else None,PANGU_baryon_mass=float(p),AthenaK_baryon_mass=float(q),pair_difference_PANGU_minus_AthenaK=float(p-q),absolute_densitized_mass_difference_L1=float(error)) for (a,b),(p,q),error in zip(bins,totals,l1)]
 difference=totals[:,0].sum()-totals[:,1].sum();inner=totals[:2].sum(axis=0);outer=totals[2:].sum(axis=0)
 pa=list(csv.DictReader(l for l in (R/'source/PANGU/nr_horizon.csv').read_text().splitlines() if not l.startswith('#')))
 pa=[x for x in pa if int(x['found'])==1 and float(x['scaled_rms_expansion'])<=float(x['expansion_tolerance'])]
 coeff=None
 for line in (R/'source/AthenaK/athenak_tov_collapse_n64.horizon_shape_0.txt').read_text().splitlines():
  if line.strip() and not line.startswith('#'):coeff=np.fromstring(line,sep=' ')
 lower=float(coeff[0]/math.sqrt(4*math.pi)-np.linalg.norm(coeff[1:])*math.sqrt(168/(4*math.pi)))
 result=dict(time=100,rows=rows,absolute_densitized_mass_difference_L1_outside_r1p8=float(l1[2:].sum()),absolute_densitized_mass_difference_L1_global=float(l1.sum()),global_pair_difference=float(difference),pair_difference_inside_r1p8=float(inner[0]-inner[1]),pair_difference_outside_r1p8=float(outer[0]-outer[1]),fraction_of_pair_difference_inside_r1p8=float((inner[0]-inner[1])/difference),PANGU_inactive_constraint_mask_mass=float(inactive[0]),AthenaK_mass_on_same_PANGU_inactive_mask=float(inactive[1]),PANGU_latest_qualified_AH_time=float(pa[-1]['time']),PANGU_latest_qualified_min_radius=float(pa[-1]['min_radius']),AthenaK_final_reported_AH_lower_radius_bound=lower,interpretation='Endpoint spatial localization, not a complete conservation/flux budget. r<1.8 lies within both latest reported AH radii; PANGU last qualified AH precedes endpoint by~1.89time units. Native finder quality is a separate limitation.',source_archives=[pe,index[key]])
 assert 1.8<min(lower,float(pa[-1]['min_radius']))
 (R/'ENDPOINT_BARYON_REGIONS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
finally:ap.unlink();pp.unlink()
