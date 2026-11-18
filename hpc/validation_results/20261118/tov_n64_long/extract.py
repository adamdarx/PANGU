"""Read-only matched N64 field comparison from SHA256-verified archives."""
from pathlib import Path
import csv,hashlib,importlib.util,json,math,subprocess
import h5py,numpy as np
R=Path(__file__).resolve().parent
PA=Path('/mnt/d/PANGU_TEST_DATA/PANGU_II_NR_DATA/Matter/20261102-stellar-formation/runs/collapse/16902889/unstable_b1')
BASE=Path('/home/adamdarx/pangu-setup')
AK='20261117-athenak-grmhd-reference/tov-repair/runs/tov_n64/17374027/bin/'
INDEX=json.loads((BASE/'archive-et-formal-20261117/live-archive-index.json').read_text())['files']
spec=importlib.util.spec_from_file_location('reader',str(BASE/'tov-reference-20261117/athenak-source/vis/python/bin_convert.py'))
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
(R/'scratch').mkdir(exist_ok=True);(R/'source/PANGU').mkdir(exist_ok=True)
manifest=[]
cached={x['nominal_time']:x for x in json.loads((R/'FIELDS.json').read_text())} if (R/'FIELDS.json').exists() else {}
prior_hashes={x['archive']:x['sha256'] for x in json.loads((R/'FIELD_ARCHIVE_HASHES.json').read_text())} if (R/'FIELD_ARCHIVE_HASHES.json').exists() else {}

def unpack(e,label):
 target=R/'scratch'/label;h=hashlib.sha256();count=0
 with target.open('wb') as f:
  p=subprocess.Popen(['/home/adamdarx/.local/pangu-host/bin/zstd','-dc',e['archive']],stdout=subprocess.PIPE)
  try:
   for block in iter(lambda:p.stdout.read(8388608),b''):f.write(block);h.update(block);count+=len(block)
  finally:p.stdout.close()
  assert p.wait()==0
 assert h.hexdigest()==e['sha256'] and count==e['bytes'],'Archive verification failed'
 manifest.append(dict(archive=e['archive'],sha256=h.hexdigest(),bytes=count));return target
def ak(kind,i):
 e=INDEX[AK+'athenak_tov_collapse_n64.%s.%05d.bin'%(kind,i)];p=unpack(e,'ak_'+kind+'.bin');d=reader.read_binary(str(p));p.unlink();return d
def geom_masks(geo,n=16):
 masks=[];vol=[]
 for lo in geo:
  dx=(lo[1::2]-lo[::2])/n;x,y,z=[lo[2*i]+(np.arange(n)+.5)*dx[i] for i in range(3)]
  r2=x[None,None,:]**2+y[None,:,None]**2+z[:,None,None]**2
  masks.append((r2>=9)&(r2<=900));vol.append(float(np.prod(dx)))
 return np.asarray(masks),np.asarray(vol)
def reduce_fields(rho,dens,g,H,mask,volume):
 proper=h2=hmax=shellmass=totalmass=rhomax=0.;shellcells=0
 for b,m in enumerate(mask):
  assert np.isfinite(dens[b]).all() and np.isfinite(rho[b]).all()
  totalmass+=float(np.sum(dens[b]))*volume[b];rhomax=max(rhomax,float(rho[b].max()))
  if not m.any():continue
  a=[x[b] for x in g];D=a[0]*(a[3]*a[5]-a[4]**2)-a[1]*(a[1]*a[5]-a[2]*a[4])+a[2]*(a[1]*a[4]-a[2]*a[3])
  assert np.isfinite(H[b][m]).all() and np.isfinite(D[m]).all() and D[m].min()>0 and a[0][m].min()>0 and (a[0]*a[3]-a[1]**2)[m].min()>0
  w=np.sqrt(D[m])*volume[b];proper+=float(w.sum());h2+=float(np.sum(H[b][m]**2*w))
  hmax=max(hmax,float(np.max(abs(H[b][m]))));shellmass+=float(np.sum(dens[b][m]))*volume[b];shellcells+=int(m.sum())
 return dict(rho_max=rhomax,baryon_mass_field=totalmass,shell_baryon_inventory=shellmass,H_shell_proper_RMS=math.sqrt(h2/proper),H_shell_Linf=hmax,shell_proper_volume=proper,shell_cells=shellcells)
p=PA/'diagnostics/nr_horizon.csv';raw=p.read_bytes();(R/'source/PANGU/nr_horizon.csv').write_bytes(raw)
rows=list(csv.DictReader(l for l in raw.decode().splitlines() if not l.startswith('#')))
valid=[r for r in rows if int(r['found'])==1 and math.isfinite(float(r['scaled_rms_expansion'])) and float(r['scaled_rms_expansion'])<=float(r['expansion_tolerance'])*(1+1e-10)]
assert valid
pa_first=float(valid[0]['time']);pa_last_failure=max(float(r['time']) for r in rows if float(r['time'])<pa_first)
pa_bound=max(float(r['max_radius']) for r in valid);assert pa_bound<3
found=set();bound=0.
for line in (R/'source/AthenaK/athenak_tov_collapse_n64.horizon_shape_0.txt').read_text().splitlines():
 if line.startswith('# iter = '):found.add(float(line.split('Time = ')[1]))
 elif line.strip():
  c=np.fromstring(line,sep=' ');assert c.size==169
  bound=max(bound,float(c[0]/math.sqrt(4*math.pi)+np.linalg.norm(c[1:])*math.sqrt(168/(4*math.pi))))
assert bound<3,'Fixed shell intersects a reported horizon'
cutoff=5*math.floor(min(pa_first,min(found))/5);assert cutoff==40
for name in ['sync_grmhd_tov.out2.hst','phase-validated.json','command.json']:(R/'source/PANGU'/name).write_bytes((PA/name).read_bytes())
result=[]
try:
 for i in range(21):
  source=PA/('sync_grmhd_tov.sync_grmhd_tov.final.phdf' if i==20 else 'sync_grmhd_tov.sync_grmhd_tov.%05d.phdf'%i)
  e=json.loads(source.with_name(source.name+'.archive.json').read_text())
  reuse=i*5 in cached and prior_hashes.get(e['archive'])==e['sha256']
  if reuse:
   pa=dict(cached[i*5]['PANGU']);pt=pa['time'];manifest.append(dict(archive=e['archive'],sha256=e['sha256'],bytes=e['bytes'],PANGU_reduction_cached=True))
  else:p=unpack(e,'pa.phdf')
  adm=ak('adm',i);w=ak('mhd_w_bcc',i);u=ak('mhd_u',i);h=ak('con_H',i)
  for other in [w,u,h]:assert np.array_equal(adm['mb_logical'],other['mb_logical']) and abs(other['time']-adm['time'])<1e-10
  aa={k:np.asarray(v,dtype=float) for k,v in adm['mb_data'].items()};rw=np.asarray(w['mb_data']['dens'],dtype=float);rd=np.asarray(u['mb_data']['dens'],dtype=float);hh=np.asarray(h['mb_data']['con_H'],dtype=float)
  geo=np.asarray(adm['mb_geometry']);mask,volume=geom_masks(geo)
  a=reduce_fields(rw,rd,[aa['adm_g'+x] for x in ['xx','xy','xz','yy','yz','zz']],hh,mask,volume);a['time']=float(adm['time'])
  if not reuse:
   with h5py.File(p) as f:
    pt=float(f['Info'].attrs['Time']);assert abs(pt-adm['time'])<.01
    pk=[tuple(np.ravel(l).tolist()+np.ravel(x).tolist()) for l,x in zip(f['Levels'][:],f['LogicalLocations'][:])]
    akeys=[(int(x[3]),*map(int,x[:3])) for x in adm['mb_logical']];assert len(set(pk))==len(pk) and set(pk)==set(akeys)
    index={k:j for j,k in enumerate(akeys)};order=np.array([index[k] for k in pk])
    pg=np.stack([f['Locations/'+axis][:,side] for axis in 'xyz' for side in (0,-1)],axis=1);assert np.allclose(pg,geo[order],rtol=0,atol=1e-12)
    pm,pv=geom_masks(pg);pr=f['mhd.prim'][:,0];pd=f['mhd.cons'][:,0];gg=f['nr.adm'][:,:6];hc=f['nr.constraints'][:,1]
    active=f['nr.constraint_mask'][:,0]>0;assert active[pm].all(),'Shell not fully active'
    pa=reduce_fields(pr,pd,[gg[:,j] for j in range(6)],hc,pm,pv);pa['time']=pt
  if not reuse:p.unlink()
  assert pa['shell_cells']==a['shell_cells'] and abs(pa['time']-a['time'])<.01
  if i*5>cutoff:pa['rho_max']=None;a['rho_max']=None
  result.append(dict(nominal_time=i*5,PANGU=pa,AthenaK=a,pre_first_reported_AH=i*5<=cutoff))
  (R/'FIELDS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  (R/'FIELD_ARCHIVE_HASHES.json').write_text(json.dumps(manifest,indent=2)+'\n')
  print('sample',i,'times',pt,a['time'],'H',pa['H_shell_proper_RMS'],a['H_shell_proper_RMS'],flush=True)
finally:
 for p in (R/'scratch').glob('*'):p.unlink()
metadata=dict(reduction_dtype='float64',native_field_output_dtype='float32',PANGU_field_output_dtype='float64',histories_output_precision='native high-precision text; used for global conservation',status='21_pairs_read_only_verified_archive_field_analysis',N=64,common_shell=[3,30],
 proper_H_RMS='sqrt(integral(H^2 sqrt(det(gamma)) d3x)/integral(sqrt(det(gamma)) d3x))',
 H_definition='R + K^2 - K_ij K^ij -16pi E; stored native FD4/FD6 operators retained',
 global_baryon_definition='Both evolved conserved densities are densitized: integrate D d3x, without another sqrt(gamma)',
 shell_inventory_is_not_a_conservation_error=True,PANGU_max_reported_radius=pa_bound,AthenaK_spherical_harmonic_radius_upper_bound=bound,
 native_radius_bound='Orthonormal real Ylm,lmax12: c00/sqrt(4pi)+norm(c_l>0)*sqrt(168/(4pi)); single star and AH centered at origin',
 PANGU_first_qualified_AH_time=pa_first,PANGU_last_failed_AH_time=pa_last_failure,AthenaK_first_reported_AH_time=min(found),
 AthenaK_first_report_inside_PANGU_sampling_bracket=pa_last_failure<=min(found)<=pa_first,pre_AH_density_end_time=cutoff,physical_acceptance=False,
 limitations=['One resolution; no convergence order inferred.','Native FD4/FD6, atmosphere reset and boundary implementations differ.','AthenaK uses stabilized native KO0.1/kappa1.02/chi floors; PANGU archive unchanged.','Native AH found flag does not certify residual quality; mass differences cannot be assigned to evolution alone.','Global baryon histories include BH interior; shell inventories have boundary fluxes.'])
(R/'METHOD.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n');print('DONE',flush=True)
