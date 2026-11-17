from pathlib import Path
import re,json,hashlib
r=Path(__file__).parent;template=(r/'native-template.par').read_text();rows=[]
for n in (64,96):
 s=template
 changes={'$nlevels':'8','$ncells':str(n),'CarpetX::xmin':'-128','CarpetX::ymin':'-128','CarpetX::zmin':'-128','CarpetX::xmax':'128','CarpetX::ymax':'128','CarpetX::zmax':'128','CarpetX::dtfac':'0.1','CarpetX::max_grid_size_x':'32','CarpetX::max_grid_size_y':'32','CarpetX::max_grid_size_z':'32','BoxInBox::position_x_1':'0','BoxInBox::position_y_1':'0','BoxInBox::position_z_1':'0','BoxInBox::radius_1':'[ -1,31.68,15.84,7.92,3.96,1.98,0.99,0.99 ]','TOVSolverX::TOV_Rho_Central[0]':'0.007993','AsterSeeds::Ab':'1.0','AsterSeeds::press_cut':'0.000156523802193188','AsterSeeds::press_max':str(100*.007993**2),'AsterSeeds::Avec_kappa':'1','Con2PrimFactory::rho_abs_min':'1e-16','EOSX::eps_min':'1e-16','Z4c::epsdiss':'0','Con2PrimFactory::atmo_tol':'0.02','ODESolvers::method':'"RK2"'}
 for key,value in changes.items():
  s,count=re.subn(r'(?m)^'+re.escape(key)+r'\s*=.*$',key+' = '+value,s);assert count==1,key
 s=re.sub(r'(?ms)CarpetX::out_tsv_vars\s*=\s*".*?"','CarpetX::out_tsv_vars = "HydroBaseX::rho HydroBaseX::vel HydroBaseX::Bvec AsterX::dens ADMBaseX::alp"',s)
 s=re.sub(r'(?m)^CarpetX::out_norm_vars.*$','CarpetX::out_norm_vars = "HydroBaseX::rho AsterX::dens HydroBaseX::Bvec"',s)
 s+='''
# Native physical-reference case; initial perturbation/seed differ from PANGU.
CarpetX::use_subcycling = no
TOVSolverX::TOV_Num_Radial = 400000
TOVSolverX::TOV_dr[0] = 0.001
AsterSeeds::ns_perturbation_mode = "m0"
AsterSeeds::ns_perturbation_strength = 0.01
Z4c::kappa1 = 0
Z4c::kappa2 = 0
IO::checkpoint_ID = no
IO::checkpoint_on_terminate = no
CarpetX::out_norm_every = 1
'''
 # Use an exact pressure cutoff1e-6 in the native Ab*(P-Pcut) seed.
 s=re.sub(r'(?m)^AsterSeeds::press_cut.*$','AsterSeeds::press_cut = '+repr(1e-6/(100*.007993**2)),s)
 p=r/f'native_tov_n{n}_admit.par';p.write_text(s)
 formal=s.replace('Cactus::terminate = "iteration"','Cactus::terminate = "time"').replace('Cactus::cctk_itlast = 16','Cactus::cctk_final_time = 100').replace('IO::out_every = 16',f'IO::out_every = {25*n}')
 (r/f'native_tov_n{n}_formal_candidate.par').write_text(formal)
 rows.append(dict(n=n,admission_input=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),formal_target=100))
(r/'NATIVE_TOV_PLAN.json').write_text(json.dumps(dict(status='native_physical_reference_prepared_GPU_resource_admission_required',cases=rows,physical_model='Unstable magnetized Gamma2/K100 TOV, rhoc0.007993, weak poloidal field, full Z4c/GRMHD',differences=['ET native pressure depletion1% triggers collapse; PANGU/AthenaK radial velocity perturbation-0.01','ET native pressure-poloidal seed with Ab1/Pcut1e-6 has no density taper; quantify magnetic energy before outcome comparisons','Native reconstruction, integrator, grid placement and atmosphere implementations are reported separately'],comparison_scope='Mass and baryon budgets, exterior constraints, BH formation and final mass normalized to each measured initial model; collapse-time difference alone is not solver error',formal_submitted=False),indent=2)+'\n')
