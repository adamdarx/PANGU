from pathlib import Path
import re,json
r=Path(__file__).parent
ah='''
ActiveThorns = "AEILocalInterp AHFinderDirect SphericalSurface AsterAnalysis CoordinatesX"
AsterAnalysis::compute_MHD = yes
AsterAnalysis::compute_MHD_spherical = no
AHFinderDirect::N_horizons = 1
AHFinderDirect::max_allowable_horizon_radius[1] = 6
AHFinderDirect::find_every = 16
AHFinderDirect::max_Newton_iterations__initial = 50
AHFinderDirect::max_Newton_iterations__subsequent = 50
AHFinderDirect::geometry_interpolator_name = "CarpetX"
AHFinderDirect::geometry_interpolator_pars = "order=4"
AHFinderDirect::surface_interpolator_name = "Lagrange polynomial interpolation"
AHFinderDirect::surface_interpolator_pars = "order=3"
AHFinderDirect::initial_guess_method[1] = "coordinate sphere"
AHFinderDirect::initial_guess__coord_sphere__radius[1] = 2
AHFinderDirect::which_surface_to_store_info[1] = 0
AHFinderDirect::set_mask_for_individual_horizon[1] = no
AHFinderDirect::reset_horizon_after_not_finding[1] = yes
AHFinderDirect::output_BH_diagnostics = yes
AHFinderDirect::output_h_every = 0
SphericalSurface::nsurfaces = 1
SphericalSurface::maxntheta = 66
SphericalSurface::maxnphi = 124
SphericalSurface::ntheta[0] = 41
SphericalSurface::nphi[0] = 80
SphericalSurface::nghoststheta[0] = 2
SphericalSurface::nghostsphi[0] = 2
CarpetX::interpolation_order = 4
'''
for n in (64,96):
 s=(r/f'native_tov_n{n}_admit.par').read_text()
 s=re.sub(r'(?m)^CarpetX::out_norm_vars.*$','CarpetX::out_norm_vars = "HydroBaseX::rho AsterX::dens HydroBaseX::Bvec AsterAnalysis::magnetic_energy AsterAnalysis::divB Z4c::HC Z4c::MtC"',s)
 s=re.sub(r'(?m)^CarpetX::out_tsv_vars.*$','CarpetX::out_tsv_vars = "HydroBaseX::rho HydroBaseX::vel HydroBaseX::Bvec AsterX::dens ADMBaseX::alp ADMBaseX::metric Z4c::HC Z4c::MtC"',s)
 s=s.replace('Con2PrimFactory::unit_test = "yes"','Con2PrimFactory::unit_test = "no"')
 Ab=json.loads((r/'SEED_ENERGY_CALIBRATION.json').read_text())['fine']['ET_Ab_equal_energy']
 s=re.sub(r'(?m)^AsterSeeds::Ab.*$','AsterSeeds::Ab = '+repr(Ab),s)
 s+=ah
 (r/f'native_tov_n{n}_validation.par').write_text(s)
 formal=s.replace('Cactus::terminate = "iteration"','Cactus::terminate = "time"').replace('Cactus::cctk_itlast = 16','Cactus::cctk_final_time = 100').replace('IO::out_every = 16',f'IO::out_every = {25*n}').replace('AHFinderDirect::find_every = 16',f'AHFinderDirect::find_every = {10*n}').replace('CarpetX::out_norm_every = 1',f'CarpetX::out_norm_every = {n//2}')
 (r/f'native_tov_n{n}_formal.par').write_text(formal)
