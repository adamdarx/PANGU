from pathlib import Path
import numpy as np,h5py,hashlib,json
H=Path(__file__).resolve().parent;out=H/'initial-reference'
for n in (64,96):
 src=Path('/home/adamdarx/pangu-setup/bns-initial-adm-20261117/cache')/('n%d.initial.phdf'%n)
 with h5py.File(src) as f:
  assert float(f['Info'].attrs['Time'])==0
  geo=np.stack([f['Locations/'+a][:,i] for a in 'xyz' for i in (0,-1)],axis=1)
  rho=f['mhd.prim'][:,0];dens=f['mhd.cons'][:,0];b=f['mhd.b_cell'][:];alpha=f['nr.adm'][:,0]
  volume=np.prod((geo[:,1::2]-geo[:,::2])/(n//4),axis=1)[:,None,None,None]
  p=out/('bns_n%d_initial.npz'%n);np.savez_compressed(p,rho=rho,dens=dens,bfield=b,alpha=alpha,geometry=geo,locations=f['LogicalLocations'][:],levels=f['Levels'][:])
  proof=dict(source_phdf_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),npz_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),baryon_mass=float(np.sum(dens*volume)),resolution=n,blocks=len(geo),scope='PANGU actual initial fields, independently checked against AthenaK')
  p.with_suffix('.json').write_text(json.dumps(proof,indent=2));print(n,p.stat().st_size,proof['baryon_mass'],flush=True)
