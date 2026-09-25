#include "app/parameter_compat.h"

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "estimator/estimator.h"
#include "geometry_assembly.h"
#include "pangu_config.h"
#include "plugin/registry.h"
#include "utils/error_checking.hpp"

namespace pangu::app {
namespace {

using KeyMap = std::map<std::string, std::string>;

const std::map<std::string, std::pair<std::string, KeyMap>> kLegacyBlocks{
    {"job",
     {"parthenon/job",
      {{"problem_id", "problem_id"}, {"basename", "basename"}}}},
    {"mesh",
     {"parthenon/mesh",
      {{"nx1", "nx1"},
       {"nx2", "nx2"},
       {"nx3", "nx3"},
       {"x1min", "x1min"},
       {"x1max", "x1max"},
       {"x2min", "x2min"},
       {"x2max", "x2max"},
       {"x3min", "x3min"},
       {"x3max", "x3max"},
       {"ix1_bc", "ix1_bc"},
       {"ox1_bc", "ox1_bc"},
       {"ix2_bc", "ix2_bc"},
       {"ox2_bc", "ox2_bc"},
       {"ix3_bc", "ix3_bc"},
       {"ox3_bc", "ox3_bc"},
       {"nghost", "nghost"},
       {"refinement", "refinement"},
       {"numlevel", "numlevel"}}}},
    {"meshblock",
     {"parthenon/meshblock", {{"nx1", "nx1"}, {"nx2", "nx2"}, {"nx3", "nx3"}}}},
    {"time",
     {"parthenon/time",
      {{"integrator", "integrator"},
       {"tlim", "tlim"},
       {"nlim", "nlim"},
       {"dt", "dt"},
       {"evolution", "evolution"},
       {"cfl_number", "cfl_number"},
       {"ndiag", "ndiag"},
       {"ncycle_out_mesh", "ncycle_out_mesh"}}}}};

const std::map<std::string, std::set<std::string>> kPanguSchema{
    {"pangu", {"strict_parameters", "check_input_only", "compatibility_mode"}},
    {"plugins", {"enabled"}},
    {"hydro",
     {"physics", "eos", "reconstruct", "rsolver", "gamma", "gamma_max",
      "iso_sound_speed", "cfl", "density_floor", "pressure_floor", "dfloor",
      "pfloor", "fofc", "nscalars", "accel1", "accel2", "accel3",
      "refine_tolerance", "derefine_tolerance"}},
    {"mhd",
     {"physics",
      "eos",
      "reconstruct",
      "rsolver",
      "gamma",
      "gamma_max",
      "sigma_max",
      "iso_sound_speed",
      "cfl",
      "density_floor",
      "pressure_floor",
      "entropy_floor",
      "dfloor",
      "pfloor",
      "sfloor",
      "fofc",
      "puncture_protection",
      "puncture_chi_threshold",
      "puncture_density",
      "puncture_pressure",
      "nscalars",
      "refine_tolerance",
      "derefine_tolerance",
      "amr_method",
      "amr_value_max",
      "amr_interval",
      "amr_initial_refinement"}},
    {"electrons",
     {"enabled",
      "on",
      "heating",
      "gamma_e",
      "gamma_p",
      "init_to_fel_0",
      "fel_0",
      "fel_constant",
      "constant",
      "howes",
      "kawazura",
      "werner",
      "rowan",
      "sharma",
      "enforce_positive_dissipation",
      "suppress_highb_heat",
      "sigma_heat_cutoff",
      "limit_kel",
      "tp_over_te_min",
      "tp_over_te_max",
      "reinitialize",
      "write_diagnostics"}},
    {"radiation",
     {"model",        "h_target",      "beta_cool",   "start_time",
      "ramp_time",    "rho_min",       "sigma_max",   "bound_only",
      "track_energy", "nlevel",        "rotate_geo",  "angular_fluxes",
      "reconstruct",  "rad_source",    "fixed_fluid", "affect_fluid",
      "kappa_a",      "kappa_s",       "kappa_p",     "arad",
      "n_0_floor",    "power_opacity", "beam_source", "dii_dt"}},
    {"units",
     {"length_cgs", "mass_cgs", "time_cgs", "density_cgs", "bhmass_msun",
      "mu"}},
    {"geometry",
     {"background", "expected_metric", "expected_mode", "bh_spin", "hslope",
      "excision", "excision_radius", "flux_excision_radius", "dexcise",
      "pexcise"}},
    {"numerical_relativity",
     {"cfl",
      "finite_difference_order",
      "boundary_extrapolation_order",
      "chi_psi_power",
      "chi_div_floor",
      "chi_min_floor",
      "floor_chi",
      "dissipation",
      "damp_kappa1",
      "damp_kappa2",
      "lapse_oplog",
      "lapse_harmonic_factor",
      "lapse_harmonic",
      "lapse_advect",
      "slow_start_lapse",
      "slow_start_amplitude",
      "slow_start_time",
      "slow_start_index",
      "shift_gamma",
      "shift_alpha2_gamma",
      "shift_harmonic",
      "shift_advect",
      "shift_eta",
      "use_z4c",
      "matter_source",
      "matter_sxx",
      "matter_sxy",
      "matter_sxz",
      "matter_syy",
      "matter_syz",
      "matter_szz",
      "matter_energy",
      "matter_momentum_x",
      "matter_momentum_y",
      "matter_momentum_z",
      "matter_amplitude",
      "matter_wave_number_x1",
      "matter_wave_number_x2",
      "matter_wave_number_x3",
      "matter_angular_frequency",
      "refine_tolerance",
      "derefine_tolerance",
      "amr_method",
      "amr_chi_min",
      "amr_dchi_max",
      "amr_radius_0",
      "amr_radius_0_refinement_level",
      "amr_radius_1",
      "amr_radius_1_refinement_level",
      "amr_radius_2",
      "amr_radius_2_refinement_level",
      "amr_radius_3",
      "amr_radius_3_refinement_level",
      "tracker_enabled",
      "tracker_count",
      "tracker_refinement_radius",
      "tracker_center_x1",
      "tracker_center_x2",
      "tracker_center_x3",
      "tracker_0_center_x1",
      "tracker_0_center_x2",
      "tracker_0_center_x3",
      "tracker_0_mass",
      "tracker_0_refinement_radius",
      "tracker_0_refinement_level",
      "tracker_1_center_x1",
      "tracker_1_center_x2",
      "tracker_1_center_x3",
      "tracker_1_mass",
      "tracker_1_refinement_radius",
      "tracker_1_refinement_level",
      "tracker_2_center_x1",
      "tracker_2_center_x2",
      "tracker_2_center_x3",
      "tracker_2_mass",
      "tracker_2_refinement_radius",
      "tracker_2_refinement_level",
      "tracker_3_center_x1",
      "tracker_3_center_x2",
      "tracker_3_center_x3",
      "tracker_3_mass",
      "tracker_3_refinement_radius",
      "tracker_3_refinement_level",
      "weyl_enabled",
      "waveform_enabled",
      "waveform_num_radii",
      "waveform_radius_1",
      "waveform_radius_2",
      "waveform_radius_3",
      "waveform_radius_4",
      "waveform_geodesic_level",
      "waveform_interpolation_points",
      "waveform_dt",
      "horizon_enabled",
      "horizon_output_grid",
      "horizon_count",
      "horizon_ntheta",
      "horizon_lmax",
      "horizon_iterations",
      "horizon_initial_radius",
      "horizon_initial_radius_0",
      "horizon_initial_radius_1",
      "horizon_initial_radius_2",
      "horizon_initial_radius_3",
      "horizon_tracker_0",
      "horizon_tracker_1",
      "horizon_tracker_2",
      "horizon_tracker_3",
      "horizon_center_x1_0",
      "horizon_center_x2_0",
      "horizon_center_x3_0",
      "horizon_center_x1_1",
      "horizon_center_x2_1",
      "horizon_center_x3_1",
      "horizon_center_x1_2",
      "horizon_center_x2_2",
      "horizon_center_x3_2",
      "horizon_center_x1_3",
      "horizon_center_x2_3",
      "horizon_center_x3_3",
      "horizon_start_time_0",
      "horizon_start_time_1",
      "horizon_start_time_2",
      "horizon_start_time_3",
      "horizon_stop_time_0",
      "horizon_stop_time_1",
      "horizon_stop_time_2",
      "horizon_stop_time_3",
      "horizon_wait_for_punctures_0",
      "horizon_wait_for_punctures_1",
      "horizon_wait_for_punctures_2",
      "horizon_wait_for_punctures_3",
      "horizon_mass_weighted_center_0",
      "horizon_mass_weighted_center_1",
      "horizon_mass_weighted_center_2",
      "horizon_mass_weighted_center_3",
      "horizon_flow_gain",
      "horizon_flow",
      "horizon_mass_tolerance",
      "horizon_hmean_limit",
      "horizon_expand_guess",
      "horizon_merger_distance",
      "diagnostic_directory"}},
    {"source_terms",
     {"cooling_rate", "heating_rate", "energy_floor", "point_mass", "softening",
      "constant_acceleration", "acceleration_value", "acceleration_direction",
      "ism_cooling", "ism_heating_rate", "relativistic_cooling",
      "relativistic_rate", "relativistic_power"}},
    {"problem",
     {"pgen_name",
      "amplitude",
      "amp",
      "mass",
      "punc_ADM_mass",
      "punc_velocity_x1",
      "punc_velocity_x2",
      "punc_velocity_x3",
      "punc_center_x1",
      "punc_center_x2",
      "punc_center_x3",
      "verbose",
      "par_b",
      "par_m_plus",
      "par_m_minus",
      "target_M_plus",
      "target_M_minus",
      "par_P_plus1",
      "par_P_plus2",
      "par_P_plus3",
      "par_P_minus1",
      "par_P_minus2",
      "par_P_minus3",
      "par_S_plus1",
      "par_S_plus2",
      "par_S_plus3",
      "par_S_minus1",
      "par_S_minus2",
      "par_S_minus3",
      "center_offset1",
      "center_offset2",
      "center_offset3",
      "give_bare_mass",
      "grid_setup_method",
      "npoints_A",
      "npoints_B",
      "npoints_phi",
      "Newton_tol",
      "Newton_maxit",
      "TP_epsilon",
      "TP_Tiny",
      "TP_Extend_Radius",
      "adm_tol",
      "do_residuum_debug_output",
      "solve_momentum_constraint",
      "initial_lapse_psi_exponent",
      "swap_xz",
      "constraint_mask_radius",
      "puncture_count",
      "puncture_0_mass",
      "puncture_0_center_x",
      "puncture_0_center_y",
      "puncture_0_center_z",
      "puncture_0_momentum_x",
      "puncture_0_momentum_y",
      "puncture_0_momentum_z",
      "puncture_0_spin_x",
      "puncture_0_spin_y",
      "puncture_0_spin_z",
      "puncture_1_mass",
      "puncture_1_center_x",
      "puncture_1_center_y",
      "puncture_1_center_z",
      "puncture_1_momentum_x",
      "puncture_1_momentum_y",
      "puncture_1_momentum_z",
      "puncture_1_spin_x",
      "puncture_1_spin_y",
      "puncture_1_spin_z",
      "puncture_spectral_points",
      "puncture_spectral_scale",
      "puncture_nonlinear_tolerance",
      "puncture_linear_tolerance",
      "puncture_maximum_newton_iterations",
      "puncture_maximum_linear_iterations",
      "atmosphere_density",
      "atmosphere_pressure",
      "btilde1",
      "btilde2",
      "btilde3",
      "rhoc",
      "kappa",
      "npoints",
      "rho_cut",
      "v_pert",
      "b_norm",
      "pcut",
      "magindex",
      "kx1",
      "kx2",
      "kx3",
      "radius",
      "inner_radius",
      "outer_radius",
      "width",
      "x0",
      "y0",
      "z0",
      "xshock",
      "shock_dir",
      "direction",
      "density_left",
      "dl",
      "density_right",
      "dr",
      "pressure_left",
      "pl",
      "pressure_right",
      "pr",
      "velocity_left",
      "ul",
      "vl",
      "wl",
      "velocity_right",
      "ur",
      "vr",
      "wr",
      "ambient_density",
      "ambient_pressure",
      "mach",
      "rho",
      "v0",
      "zero_ug",
      "centered",
      "inner_pressure",
      "pn_amb",
      "prat",
      "velocity",
      "density_inner",
      "density_outer",
      "shear_inner",
      "shear_outer",
      "vshear",
      "interface",
      "dens",
      "pgas",
      "vx0",
      "vy0",
      "vz0",
      "wave_flag",
      "kharma_mode",
      "mode_direction",
      "phase",
      "mode_density0",
      "mode_internal0",
      "mode_u10",
      "mode_u20",
      "mode_u30",
      "mode_b10",
      "mode_b20",
      "mode_b30",
      "along_x1",
      "along_x2",
      "along_x3",
      "omega_real",
      "omega_imag",
      "erad",
      "temp",
      "v1",
      "nu",
      "pos_1",
      "pos_2",
      "pos_3",
      "dir_1",
      "dir_2",
      "dir_3",
      "spread",
      "fxrad",
      "fyrad",
      "fzrad",
      "delta",
      "drho_real",
      "drho_imag",
      "dpgas_real",
      "dpgas_imag",
      "dux_real",
      "dux_imag",
      "duy_real",
      "duy_imag",
      "duz_real",
      "duz_imag",
      "derad_real",
      "derad_imag",
      "dfxrad_real",
      "dfxrad_imag",
      "dfyrad_real",
      "dfyrad_imag",
      "dfzrad_real",
      "dfzrad_imag",
      "iprob",
      "drat",
      "smooth_interface",
      "advect_dens",
      "flow_dir",
      "iproblem",
      "vflow",
      "b_par",
      "b_perp",
      "v_perp",
      "v_par",
      "pres",
      "pressure",
      "right_polar",
      "velocity_amplitude",
      "b1",
      "b2",
      "b3",
      "bx0",
      "by0",
      "bz0",
      "sigma_norm",
      "sigma_pow",
      "rhomin",
      "umin",
      "a_norm",
      "sigma_target",
      "sigma_rmin",
      "write_final_csv",
      "b2_left",
      "b2_right",
      "b3_left",
      "b3_right",
      "velocity_1_left",
      "velocity_2_left",
      "velocity_3_left",
      "velocity_1_right",
      "velocity_2_right",
      "velocity_3_right",
      "profile",
      "wavenumber",
      "spread_x1",
      "spread_x2",
      "spread_x3",
      "vel_comp",
      "x10",
      "x20",
      "x30",
      "ipert",
      "d0",
      "p0",
      "nwx",
      "nwy",
      "nwz",
      "ifield",
      "zlimit",
      "beta",
      "k_adi",
      "r_crit",
      "fm_torus",
      "chakrabarti_torus",
      "prograde",
      "r_edge",
      "r_peak",
      "rho_max",
      "rho_min",
      "rho_pow",
      "pgas_min",
      "pgas_pow",
      "tilt_angle",
      "pert_amp",
      "potential_beta_min",
      "potential_cutoff",
      "potential_falloff",
      "potential_r_pow",
      "potential_rho_pow",
      "vertical_field",
      "n_param",
      "l"}}};

bool StartsWith(const std::string &value, const std::string &prefix) {
  return value.rfind(prefix, 0) == 0;
}

bool IsIntegerParameter(const std::string &block, const std::string &name) {
  if (block == "parthenon/mesh") {
    return name == "nx1" || name == "nx2" || name == "nx3" ||
           name == "nghost" || name == "numlevel";
  }
  if (block == "parthenon/meshblock") {
    return name == "nx1" || name == "nx2" || name == "nx3";
  }
  if (block == "parthenon/time") {
    return name == "nlim" || name == "ndiag" || name == "ncycle_out_mesh";
  }
  return false;
}

bool IsRealParameter(const std::string &block, const std::string &name) {
  if (block == "parthenon/mesh") {
    return name == "x1min" || name == "x1max" || name == "x2min" ||
           name == "x2max" || name == "x3min" || name == "x3max";
  }
  if (block == "parthenon/time") {
    return name == "tlim" || name == "dt" || name == "cfl_number";
  }
  return false;
}

void CopyLegacyParameter(parthenon::ParameterInput *pin,
                         const std::string &source_block,
                         const std::string &source_name,
                         const std::string &target_block,
                         const std::string &target_name) {
  if (IsIntegerParameter(target_block, target_name)) {
    pin->SetInteger(target_block, target_name,
                    pin->GetInteger(source_block, source_name));
  } else if (IsRealParameter(target_block, target_name)) {
    pin->SetReal(target_block, target_name,
                 pin->GetReal(source_block, source_name));
  } else {
    pin->SetString(target_block, target_name,
                   pin->GetString(source_block, source_name));
  }
}

void CopyRealIfMissing(parthenon::ParameterInput *pin,
                       const std::string &source_block,
                       const std::string &source_name,
                       const std::string &target_block,
                       const std::string &target_name) {
  if (pin->DoesParameterExist(source_block, source_name) &&
      !pin->DoesParameterExist(target_block, target_name)) {
    pin->SetReal(target_block, target_name,
                 pin->GetReal(source_block, source_name));
  }
}

std::string NormalizeProblemName(const std::string &name) {
  if (name == "shock_tube" || name == "Sod" || name == "sod")
    return "sod";
  if (name == "linear_wave" || name == "LinWave")
    return "linear_wave";
  if (name == "Blast" || name == "blast")
    return "blast";
  if (name == "KH" || name == "kh")
    return "kh";
  if (name == "RTI" || name == "rt")
    return "rt";
  if (name == "advection" || name == "Advect")
    return "hydro_advection";
  return name;
}

} // namespace

void NormalizeParameters(parthenon::ParameterInput *pin) {
  pin->GetOrAddBoolean("pangu", "strict_parameters", true);
  pin->GetOrAddBoolean("pangu", "check_input_only", false);
  if (pin->DoesParameterExist("radiation", "enabled")) {
    PARTHENON_REQUIRE(pin->GetBoolean("radiation", "enabled"),
                      "legacy radiation/enabled=false cannot be resumed by a "
                      "compile-time cooling build");
    // RC-1 and later select the radiation backend at compile time.  Retain
    // restart compatibility with pre-RC-1 cooling trajectories whose embedded
    // parameter database recorded the equivalent enabled=true switch.
    pin->RemoveParameter("radiation", "enabled");
  }
  const auto mode =
      pin->GetOrAddString("pangu", "compatibility_mode", "translate");
  if (mode != "translate" && mode != "native") {
    PARTHENON_FAIL("pangu/compatibility_mode must be 'translate' or 'native'");
  }
  if (mode == "translate") {
    for (const auto &[legacy_block, target] : kLegacyBlocks) {
      if (!pin->DoesBlockExist(legacy_block))
        continue;
      const auto &[target_block, keys] = target;
      for (const auto &name : pin->GetParameterNames(legacy_block)) {
        const auto key = keys.find(name);
        if (key == keys.end()) {
          PARTHENON_FAIL("Unsupported legacy parameter <" + legacy_block +
                         ">/" + name);
        }
        if (!pin->DoesParameterExist(target_block, key->second)) {
          CopyLegacyParameter(pin, legacy_block, name, target_block,
                              key->second);
        }
      }
    }

    if (!pin->DoesParameterExist("parthenon/job", "problem_id")) {
      if (pin->DoesParameterExist("problem", "pgen_name")) {
        pin->SetString(
            "parthenon/job", "problem_id",
            NormalizeProblemName(pin->GetString("problem", "pgen_name")));
      } else if (pin->DoesParameterExist("job", "basename")) {
        pin->SetString("parthenon/job", "problem_id",
                       NormalizeProblemName(pin->GetString("job", "basename")));
      }
    }

    CopyRealIfMissing(pin, "time", "cfl_number", "hydro", "cfl");
    CopyRealIfMissing(pin, "hydro", "dfloor", "hydro", "density_floor");
    CopyRealIfMissing(pin, "hydro", "pfloor", "hydro", "pressure_floor");
    CopyRealIfMissing(pin, "time", "cfl_number", "mhd", "cfl");
    CopyRealIfMissing(pin, "mhd", "dfloor", "mhd", "density_floor");
    CopyRealIfMissing(pin, "mhd", "pfloor", "mhd", "pressure_floor");
    CopyRealIfMissing(pin, "mhd", "sfloor", "mhd", "entropy_floor");
    CopyRealIfMissing(pin, "problem", "amp", "problem", "amplitude");
    CopyRealIfMissing(pin, "problem", "xshock", "problem", "x0");
    CopyRealIfMissing(pin, "problem", "shock_dir", "problem", "direction");
    CopyRealIfMissing(pin, "problem", "dl", "problem", "density_left");
    CopyRealIfMissing(pin, "problem", "dr", "problem", "density_right");
    CopyRealIfMissing(pin, "problem", "pl", "problem", "pressure_left");
    CopyRealIfMissing(pin, "problem", "pr", "problem", "pressure_right");
    CopyRealIfMissing(pin, "problem", "ul", "problem", "velocity_left");
    CopyRealIfMissing(pin, "problem", "ur", "problem", "velocity_right");
    CopyRealIfMissing(pin, "problem", "dens", "problem", "ambient_density");
    CopyRealIfMissing(pin, "problem", "pgas", "problem", "ambient_pressure");
    CopyRealIfMissing(pin, "problem", "inner_radius", "problem", "radius");
    CopyRealIfMissing(pin, "problem", "pn_amb", "problem", "ambient_pressure");
    if (pin->DoesParameterExist("mesh", "nghost")) {
      pin->SetInteger("parthenon/mesh", "nghost",
                      pin->GetInteger("mesh", "nghost"));
    }

    if (pin->DoesParameterExist("problem", "pn_amb") &&
        pin->DoesParameterExist("problem", "prat") &&
        !pin->DoesParameterExist("problem", "inner_pressure")) {
      pin->SetReal("problem", "inner_pressure",
                   pin->GetReal("problem", "pn_amb") *
                       pin->GetReal("problem", "prat"));
    }
    if (pin->DoesParameterExist("problem", "vshear")) {
      const auto shear = pin->GetReal("problem", "vshear");
      if (!pin->DoesParameterExist("problem", "shear_inner"))
        pin->SetReal("problem", "shear_inner", 0.5 * shear);
      if (!pin->DoesParameterExist("problem", "shear_outer"))
        pin->SetReal("problem", "shear_outer", -0.5 * shear);
    }
    if (pin->DoesParameterExist("problem", "drat")) {
      CopyRealIfMissing(pin, "problem", "drat", "problem", "density_inner");
      if (!pin->DoesParameterExist("problem", "density_outer"))
        pin->SetReal("problem", "density_outer", 1.0);
    }

    if (pin->DoesParameterExist("hydro", "rsolver") &&
        pin->GetString("hydro", "rsolver") == "advect") {
      pin->SetString("hydro", "rsolver", "llf");
    }
    const bool kinematic =
        pin->DoesParameterExist("parthenon/time", "evolution") &&
        pin->GetString("parthenon/time", "evolution") == "kinematic";
    if (kinematic && pin->DoesBlockExist("hydro"))
      pin->SetString("hydro", "rsolver", "none");
    if (kinematic && pin->DoesBlockExist("mhd"))
      pin->SetString("mhd", "rsolver", "none");
    if (pin->DoesParameterExist("hydro_srcterms", "const_accel") &&
        pin->GetBoolean("hydro_srcterms", "const_accel")) {
      const auto direction =
          pin->GetInteger("hydro_srcterms", "const_accel_dir");
      const auto acceleration =
          pin->GetReal("hydro_srcterms", "const_accel_val");
      if (direction >= 1 && direction <= 3) {
        pin->SetReal("hydro", "accel" + std::to_string(direction),
                     acceleration);
      }
    }

    for (const auto &boundary :
         {"ix1_bc", "ox1_bc", "ix2_bc", "ox2_bc", "ix3_bc", "ox3_bc"}) {
      if (pin->DoesParameterExist("parthenon/mesh", boundary) &&
          pin->GetString("parthenon/mesh", boundary) == "reflect") {
        pin->SetString("parthenon/mesh", boundary, "reflecting");
      }
    }
  }

  // Avoid evaluating the MKS map exactly on its polar coordinate singularities
  // while retaining essentially complete angular coverage.  This topology is
  // metric-wide rather than problem-specific.  Parthenon has already attached
  // Cartesian defaults here, so these are deliberate compile-line overrides.
  if (std::string(geometry::ConfiguredMetricName()) == "mks") {
    pin->SetReal("parthenon/mesh", "x2min", 1.0e-5);
    pin->SetReal("parthenon/mesh", "x2max", 1.0 - 1.0e-5);
    pin->SetReal("parthenon/mesh", "x3min", 0.0);
    pin->SetReal("parthenon/mesh", "x3max",
                 2.0 * 3.141592653589793238462643383279502884);
    pin->SetString("parthenon/mesh", "ix1_bc", "outflow");
    pin->SetString("parthenon/mesh", "ox1_bc", "outflow");
    pin->SetString("parthenon/mesh", "ix2_bc", "reflecting");
    pin->SetString("parthenon/mesh", "ox2_bc", "reflecting");
    pin->SetString("parthenon/mesh", "ix3_bc", "periodic");
    pin->SetString("parthenon/mesh", "ox3_bc", "periodic");
  }

  PARTHENON_REQUIRE(pin->DoesParameterExist("parthenon/job", "problem_id"),
                    "<parthenon/job>/problem_id is required");
}

void ValidateParameters(parthenon::ParameterInput *pin) {
  if (!pin->GetOrAddBoolean("pangu", "strict_parameters", true))
    return;
  const auto compatibility_mode = pin->GetString("pangu", "compatibility_mode");

  for (const auto &block : pin->GetBlockNames()) {
    if (StartsWith(block, "parthenon/"))
      continue;
    // Parameters inside a plugin namespace are validated by its initializer,
    // but the namespace itself must name a plugin compiled into this build.
    if (StartsWith(block, "plugin/")) {
      const auto plugin_name =
          block.substr(std::char_traits<char>::length("plugin/"));
      PARTHENON_REQUIRE(plugin::IsCompiled(plugin_name),
                        "Input block <" + block +
                            "> belongs to a plugin that is not installed");
      continue;
    }
    if (compatibility_mode == "translate" && kLegacyBlocks.count(block) != 0)
      continue;
    if (compatibility_mode == "translate" &&
        (block == "comment" || block == "hydro_srcterms" ||
         StartsWith(block, "output")))
      continue;
    const auto schema = kPanguSchema.find(block);
    if (schema == kPanguSchema.end()) {
      PARTHENON_FAIL("Unknown PANGU input block <" + block + ">");
    }
    for (const auto &name : pin->GetParameterNames(block)) {
      bool indexed_waveform_radius = false;
      constexpr const char *waveform_radius_prefix = "waveform_radius_";
      if (block == "numerical_relativity" &&
          StartsWith(name, waveform_radius_prefix)) {
        const std::string suffix =
            name.substr(std::char_traits<char>::length(waveform_radius_prefix));
        indexed_waveform_radius = !suffix.empty();
        for (const char character : suffix)
          indexed_waveform_radius =
              indexed_waveform_radius && character >= '0' && character <= '9';
      }
      bool indexed_amr_radius = false;
      constexpr const char *amr_radius_prefix = "amr_radius_";
      if (block == "numerical_relativity" &&
          StartsWith(name, amr_radius_prefix)) {
        std::string suffix =
            name.substr(std::char_traits<char>::length(amr_radius_prefix));
        constexpr const char *refinement_suffix = "_refinement_level";
        const std::size_t refinement_position = suffix.find(refinement_suffix);
        if (refinement_position != std::string::npos &&
            refinement_position +
                    std::char_traits<char>::length(refinement_suffix) ==
                suffix.size()) {
          suffix.erase(refinement_position);
        }
        indexed_amr_radius = !suffix.empty();
        for (const char character : suffix)
          indexed_amr_radius =
              indexed_amr_radius && character >= '0' && character <= '9';
      }
      if (schema->second.count(name) == 0 && !indexed_waveform_radius &&
          !indexed_amr_radius) {
        PARTHENON_FAIL("Unknown parameter <" + block + ">/" + name);
      }
    }
  }
}

void PrintConfiguration(parthenon::ParameterInput *pin, std::ostream &os) {
  os << "PANGU " << PANGU_VERSION << "\n"
     << "  PANGU commit: " << PANGU_GIT_COMMIT << " (" << PANGU_GIT_DIRTY
     << ")\n"
     << "  Parthenon commit: " << PANGU_PARTHENON_COMMIT << "\n"
     << "  configured: " << PANGU_CONFIGURE_TIME << "\n"
     << "  problem_id: " << pin->GetString("parthenon/job", "problem_id")
     << "\n"
     << "  estimator: " << pangu::estimator::estimator_name
     << " (compile-time)\n"
     << "  backends: MPI=" << PANGU_ENABLE_MPI << " HDF5=" << PANGU_ENABLE_HDF5
     << " OpenMP=" << PANGU_ENABLE_OPENMP << " CUDA=" << PANGU_ENABLE_CUDA
     << "\n";
}

} // namespace pangu::app
