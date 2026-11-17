"""Generate N64/N96 physical BNS references from the accepted native ET template."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent
template=(R/'native-tov-template.par').read_text();rows=[]
table='/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-athenak-grmhd-reference/source-records/bns/bns.dat'
for n in [64,96]:
 s=template.replace('$nlevels = 8','$nlevels = 9').replace('$ncells = 64','$ncells = '+str(n))
 s='\n'.join(line.rstrip() for line in s.splitlines() if not line.strip().startswith(('TOVSolverX::','AsterSeeds::','ADMBaseX::initial_data','ADMBaseX::initial_lapse','BoxInBox::radius_1','BoxInBox::position_','AHFinderDirect::find_every','CarpetX::out_norm_every','Cactus::terminate','Cactus::cctk_itlast','IO::out_every','IO::out_dir','CarpetX::out_tsv_vars')))
 s=s.replace('ActiveThorns = "TOVSolverX"','ActiveThorns = "TableBNS"')
 for axis in 'xyz':s=re.sub(r'(CarpetX::'+axis+r'min\s*=\s*)-128\b',r'\g<1>-256',s);s=re.sub(r'(CarpetX::'+axis+r'max\s*=\s*)128\b',r'\g<1>256',s)
 s=s.replace('ADMBaseX::initial_shift      = "tov"','ADMBaseX::initial_shift = "zero"')
 s=s.replace('ODESolvers::method = "RK2"','ODESolvers::method = "RK4"').replace('Con2PrimFactory::rho_abs_min = 1e-16','Con2PrimFactory::rho_abs_min = 1e-14').replace('Z4c::kappa1 = 0\n','Z4c::kappa1 = 0.02\n')
 s+='''
ADMBaseX::initial_data = "none"
ADMBaseX::initial_lapse = "none"
AsterSeeds::test_type = "None"
AsterX::zero_psi = yes
BoxInBox::radius_1 = [-1,64,32,16,9,8,4,2,1]
BoxInBox::radius_z_1 = [-1,128,64,32,21,8,4,2,1]
TableBNS::rho_floor = 1e-14
TableBNS::pressure_floor = 1e-26
CarpetX::out_tsv_vars = "HydroBaseX::rho HydroBaseX::vel HydroBaseX::Bvec AsterX::dens ADMBaseX::alp ADMBaseX::metric Z4c::HC Z4c::MtC"
'''
 s=s.replace('CarpetX::out_norm_vars = "HydroBaseX::rho', 'CarpetX::out_norm_vars = "HydroBaseX::vel HydroBaseX::rho')
 s+='TableBNS::table_file = "'+table+'"\n'
 dt=.1*(512/n)/256
 s+='AHFinderDirect::find_after_individual[1] = '+str(round(40/dt))+'\n'
 case=dict(n=n,target=200,dt=dt,table=table,table_sha256=json.loads((R/'TABLE_PROOF.json').read_text())['sha256'],inputs={})
 for mode in ['admission','formal']:
  tail='Cactus::terminate = "iteration"\nCactus::cctk_itlast = 32\n' if mode=='admission' else 'Cactus::terminate = "time"\nCactus::cctk_final_time = 200\n'
  tail+='IO::out_dir = "output"\nIO::out_every = '+str(32 if mode=='admission' else round(5/dt))+'\n'
  tail+='CarpetX::out_norm_every = '+str(1 if mode=='admission' else round(.1/dt))+'\nAHFinderDirect::find_every = '+str(round(.5/dt))+'\n'
  p=R/('bns_n%d_%s.par'%(n,mode));text=s+'\n'+tail
  seen={}
  for line in text.splitlines():
   m=re.match(r'\s*([A-Za-z][\w:]*(?:\[\d+\])?)\s*=',line)
   if m and m[1].lower()!='activethorns':
    key=m[1].lower();assert key not in seen,'Duplicate parameter '+key;seen[key]=True
  p.write_text(text);case['inputs'][mode]=dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 rows.append(case)
(R/'PLAN.json').write_text(json.dumps(dict(cases=rows,physical_acceptance=False,scope='Same from-rest constraint-solved binary table, native AsterX/Z4c evolution; independent grid staggering and AMR realization. No independent initial-data solver claim.'),indent=2))
