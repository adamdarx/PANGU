cmake_minimum_required(VERSION 3.20)

project(PANGU VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "PANGU does not support in-source builds. Use cmake -S . -B build-<backend>.")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PANGUOptions)
include(PANGUProvenance)
include(PANGUPlugins)

pangu_configure_parthenon()
add_subdirectory(parthenon EXCLUDE_FROM_ALL)

pangu_discover_plugins()

pangu_collect_provenance()
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/pangu_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/pangu_config.h"
  @ONLY)

add_subdirectory(src)

include(CTest)
if(BUILD_TESTING AND PANGU_ENABLE_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  if(PHYSICS STREQUAL "gr" AND METRIC STREQUAL "mks")
    # MKS executables accept only problems on the MKS background.
    set(PANGU_SMOKE_INPUT
        "${CMAKE_CURRENT_SOURCE_DIR}/input/relativity/magnetised_bondi_mks.in")
  else()
    set(PANGU_SMOKE_INPUT
        "${CMAKE_CURRENT_SOURCE_DIR}/input/hydro/hydro_advection.in")
  endif()
  # A synchronized Z4c executable deliberately accepts only NR and
  # synchronized-matter problems. Keep fixed-background input checks on
  # assemblies where the hydro problem is part of the executable domain.
  if(NOT (PHYSICS STREQUAL "gr" AND METRIC STREQUAL "z4c" AND MODE STREQUAL "sync"))
    add_test(
      NAME pangu.stage1.input_check
      COMMAND $<TARGET_FILE:pangu> --check-input -i
              "${PANGU_SMOKE_INPUT}")
    set_tests_properties(pangu.stage1.input_check PROPERTIES
      LABELS "stage1;smoke"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

  endif()

  add_test(
    NAME pangu.stage1.reject_unknown_parameter
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tst/smoke/expect_failure.py"
            --needle "Unknown parameter <pangu>/unknown_parameter" --
            $<TARGET_FILE:pangu> --check-input -i
            "${PANGU_SMOKE_INPUT}"
            pangu/unknown_parameter=1)
  set_tests_properties(pangu.stage1.reject_unknown_parameter PROPERTIES
    LABELS "stage1;negative"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

  add_test(
    NAME pangu.stage1.source_layout
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tst/smoke/source_layout_audit.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}")
  set_tests_properties(pangu.stage1.source_layout PROPERTIES
    LABELS "stage1;architecture")

  add_test(
    NAME pangu.naming.conventions
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tst/smoke/naming_conventions_audit.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}")
  set_tests_properties(pangu.naming.conventions PROPERTIES
    LABELS "architecture;naming")

  add_test(
    NAME pangu.geo1.geometry_contract
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tst/smoke/geometry_contract_audit.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --manifest
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/reference/contracts/geo-1-access-contract.json")
  set_tests_properties(pangu.geo1.geometry_contract PROPERTIES
    LABELS "geometry;geo1;architecture")

  add_test(
    NAME pangu.eh0.electron_contract
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tst/smoke/electron_contract_audit.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --contract "${CMAKE_CURRENT_SOURCE_DIR}/tst/reference/contracts/eh-0-contract.json"
            --reference "${CMAKE_CURRENT_SOURCE_DIR}/tst/reference/contracts/eh-0-reference.json")
  set_tests_properties(pangu.eh0.electron_contract PROPERTIES
    LABELS "electrons;eh0;architecture;reference")

  add_test(NAME pangu.command_line COMMAND $<TARGET_FILE:pangu_command_line>)
  set_tests_properties(pangu.command_line PROPERTIES LABELS "app;unit")

  add_test(NAME pangu.plugin.registry COMMAND $<TARGET_FILE:pangu_plugin_registry>)
  set_tests_properties(pangu.plugin.registry PROPERTIES LABELS "plugin;unit")

  add_test(NAME pangu.stage2.hydro_kernels COMMAND $<TARGET_FILE:pangu_hydro_kernels>)
  set_tests_properties(pangu.stage2.hydro_kernels PROPERTIES LABELS "stage2;unit")

  add_test(NAME pangu.stage3.mhd_kernels COMMAND $<TARGET_FILE:pangu_mhd_kernels>)
  set_tests_properties(pangu.stage3.mhd_kernels PROPERTIES LABELS "stage3;unit")

  add_test(NAME pangu.driver.contributor COMMAND $<TARGET_FILE:pangu_driver_contributor>)
  set_tests_properties(pangu.driver.contributor PROPERTIES LABELS "driver;unit")

  add_test(NAME pangu.estimator.registry COMMAND $<TARGET_FILE:pangu_estimator_registry>)
  set_tests_properties(pangu.estimator.registry PROPERTIES LABELS "estimator;unit")

  add_test(NAME pangu.reconstruct.registry COMMAND $<TARGET_FILE:pangu_reconstruct_registry>)
  set_tests_properties(pangu.reconstruct.registry PROPERTIES LABELS "reconstruct;unit;gpu")

  add_test(NAME pangu.stage5.relativity_kernels COMMAND $<TARGET_FILE:pangu_phase5_relativity>)
  set_tests_properties(pangu.stage5.relativity_kernels PROPERTIES LABELS "stage5;unit")

  add_test(NAME pangu.mks1.geometry COMMAND $<TARGET_FILE:pangu_mks_geometry>)
  set_tests_properties(pangu.mks1.geometry PROPERTIES LABELS "geometry;mks1;unit;gpu")

  if(PHYSICS STREQUAL "gr" AND METRIC STREQUAL "mks")
    add_test(
      NAME pangu.mks1.wide_meshblock_timestep
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/wide_meshblock_timestep.py"
              --executable $<TARGET_FILE:pangu>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/input/relativity/magnetised_bondi_mks.in"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/mks1-wide-meshblock")
    set_tests_properties(pangu.mks1.wide_meshblock_timestep PROPERTIES
      LABELS "mks1;regression;gpu;timestep" TIMEOUT 600)
  endif()

  add_test(NAME pangu.eh1.electron_transport COMMAND $<TARGET_FILE:pangu_electron_transport>)
  set_tests_properties(pangu.eh1.electron_transport PROPERTIES LABELS "electrons;eh1;unit")

  add_test(NAME pangu.eh2.electron_heating COMMAND $<TARGET_FILE:pangu_electron_heating>)
  set_tests_properties(pangu.eh2.electron_heating PROPERTIES LABELS "electrons;eh2;unit")

  add_test(NAME pangu.eh3.kinetic_heating COMMAND $<TARGET_FILE:pangu_electron_microphysics>)
  set_tests_properties(pangu.eh3.kinetic_heating PROPERTIES LABELS "electrons;eh3;unit;gpu;reference")

  add_test(NAME pangu.radiation.rc2_covariant COMMAND $<TARGET_FILE:pangu_radiation_cooling>)
  set_tests_properties(pangu.radiation.rc2_covariant PROPERTIES
    LABELS "radiation;rc2;unit;gpu;covariant")
  add_test(NAME pangu.radiation.rc4_target_thickness
    COMMAND $<TARGET_FILE:pangu_radiation_target_thickness>)
  set_tests_properties(pangu.radiation.rc4_target_thickness PROPERTIES
    LABELS "radiation;rc4;unit;gpu;target-thickness;energy")

  if(PHYSICS STREQUAL "gr" AND METRIC STREQUAL "z4c" AND MODE STREQUAL "sync")
    add_test(NAME pangu.nr0.sync_geometry COMMAND $<TARGET_FILE:pangu_nr0_sync_geometry>)
    set_tests_properties(pangu.nr0.sync_geometry PROPERTIES
      LABELS "numerical-relativity;nr0;unit;gpu;geometry;sync")
    add_test(
      NAME pangu.sync1.grhd_initialization
      COMMAND $<TARGET_FILE:pangu> -i
              "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grhd_shock.in")
    set_tests_properties(pangu.sync1.grhd_initialization PROPERTIES
      LABELS "numerical-relativity;sync1;grhd;initialization;gpu"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(
      NAME pangu.sync1.grhd_evolution
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync1_grhd_evolution.py"
              --pangu $<TARGET_FILE:pangu>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grhd_linear_wave.in"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync1_grhd_evolution")
    set_tests_properties(pangu.sync1.grhd_evolution PROPERTIES
      LABELS "numerical-relativity;sync1;grhd;evolution;gpu"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(
      NAME pangu.sync1.tov_initialization
      COMMAND $<TARGET_FILE:pangu> -i
              "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grhd_tov.in")
    set_tests_properties(pangu.sync1.tov_initialization PROPERTIES
      LABELS "numerical-relativity;sync1;grhd;tov;initialization;gpu"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(NAME pangu.sync1.tov_profile COMMAND $<TARGET_FILE:pangu_sync1_tov_profile>)
    set_tests_properties(pangu.sync1.tov_profile PROPERTIES
      LABELS "numerical-relativity;sync1;grhd;tov;unit;gpu;reference")
    add_test(NAME pangu.sync2.grmhd_state COMMAND $<TARGET_FILE:pangu_sync2_grmhd_state>)
    set_tests_properties(pangu.sync2.grmhd_state PROPERTIES
      LABELS "numerical-relativity;sync2;grmhd;unit;gpu;athenak")
    add_test(
      NAME pangu.sync2.grmhd_evolution
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync2_grmhd_evolution.py"
              --pangu $<TARGET_FILE:pangu>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grmhd_coupling.in"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync2_grmhd_evolution")
    set_tests_properties(pangu.sync2.grmhd_evolution PROPERTIES
      LABELS "numerical-relativity;sync2;grmhd;ct;evolution;gpu"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(
      NAME pangu.sync3.grmhd_linear_wave
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync2_grmhd_linear_wave.py"
              --pangu $<TARGET_FILE:pangu>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grmhd_linear_wave_3d.in"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync3_grmhd_linear_wave")
    set_tests_properties(pangu.sync3.grmhd_linear_wave PROPERTIES
      LABELS "numerical-relativity;sync3;grmhd;ct;amr;linear-wave;convergence;gpu;athenak"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      TIMEOUT 1800)
    add_test(
      NAME pangu.sync3.grmhd_restart
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync3_grmhd_restart.py"
              --pangu $<TARGET_FILE:pangu>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grmhd_boosted_puncture.in"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync3_grmhd_restart")
    set_tests_properties(pangu.sync3.grmhd_restart PROPERTIES
      LABELS "numerical-relativity;sync3;grmhd;restart;gpu"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      TIMEOUT 900)
    add_test(
      NAME pangu.sync3.grmhd_amr
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync3_grmhd_amr.py"
              --pangu $<TARGET_FILE:pangu>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grmhd_boosted_puncture.in"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync3_grmhd_amr")
    set_tests_properties(pangu.sync3.grmhd_amr PROPERTIES
      LABELS "numerical-relativity;sync3;grmhd;amr;moving-puncture;fofc;gpu"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      TIMEOUT 1800)
    add_test(
      NAME pangu.sync3.magnetized_tov
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync3_magnetized_tov.py"
              --pangu $<TARGET_FILE:pangu>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grmhd_tov.in"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync3_magnetized_tov")
    set_tests_properties(pangu.sync3.magnetized_tov PROPERTIES
      LABELS "numerical-relativity;sync3;grmhd;tov;stability;gpu;athenak"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    if(MPIEXEC_EXECUTABLE)
      add_test(
        NAME pangu.sync3.grmhd_mpi
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync3_grmhd_mpi.py"
                --pangu $<TARGET_FILE:pangu>
                --input "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grmhd_boosted_puncture.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync3_grmhd_mpi"
                --mpi-launcher "${MPIEXEC_EXECUTABLE}")
      set_tests_properties(pangu.sync3.grmhd_mpi PROPERTIES
        LABELS "numerical-relativity;sync3;grmhd;mpi;cuda-aware;gpu"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        TIMEOUT 900)
    endif()
    add_test(
      NAME pangu.sync1.tov_regression
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync1_tov_regression.py"
              --pangu $<TARGET_FILE:pangu>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grhd_tov.in"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync1_tov_regression")
    set_tests_properties(pangu.sync1.tov_regression PROPERTIES
      LABELS "numerical-relativity;sync1;grhd;tov;convergence;evolution;gpu"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(
      NAME pangu.sync1.tov_restart
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync1_tov_restart.py"
              --pangu $<TARGET_FILE:pangu>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grhd_tov.in"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync1_tov_restart")
    set_tests_properties(pangu.sync1.tov_restart PROPERTIES
      LABELS "numerical-relativity;sync1;grhd;tov;restart;gpu"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      TIMEOUT 600)
    if(PANGU_ATHENAK_TOV_EXECUTABLE
       AND EXISTS "${PANGU_ATHENAK_TOV_EXECUTABLE}"
       AND PANGU_ATHENAK_REFERENCE_DIR
       AND EXISTS "${PANGU_ATHENAK_REFERENCE_DIR}/vis/python/bin_convert.py")
      add_test(
        NAME pangu.sync1.tov_athenak_multitime
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync1_tov_athenak.py"
                --pangu $<TARGET_FILE:pangu>
                --athenak "${PANGU_ATHENAK_TOV_EXECUTABLE}"
                --athenak-source "${PANGU_ATHENAK_REFERENCE_DIR}"
                --pangu-input
                  "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grhd_tov.in"
                --athenak-input
                  "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/athenak_sync_grhd_tov.athinput"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync1_tov_athenak")
      set_tests_properties(pangu.sync1.tov_athenak_multitime PROPERTIES
        LABELS "numerical-relativity;sync1;grhd;tov;multitime;gpu;athenak"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        TIMEOUT 900)
      add_test(
        NAME pangu.sync1.tov_radial_athenak
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync1_tov_athenak.py"
                --pangu $<TARGET_FILE:pangu>
                --athenak "${PANGU_ATHENAK_TOV_EXECUTABLE}"
                --athenak-source "${PANGU_ATHENAK_REFERENCE_DIR}"
                --pangu-input
                  "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grhd_tov.in"
                --athenak-input
                  "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/athenak_sync_grhd_tov.athinput"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync1_tov_radial_athenak"
                --v-pert -0.024 --tlim 20.0 --output-dt 4.99999999999999
                --expected-samples 5)
      set_tests_properties(pangu.sync1.tov_radial_athenak PROPERTIES
        LABELS "numerical-relativity;sync1;grhd;tov;radial;multitime;gpu;athenak"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        TIMEOUT 900)
    endif()
    add_test(
      NAME pangu.sync1.reject_non_3d
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/smoke/expect_failure.py"
              --needle "sync Z4c evolution requires a three-dimensional mesh" --
              $<TARGET_FILE:pangu> --check-input -i
              "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/sync_grhd_linear_wave.in"
              parthenon/mesh/nx2=1 parthenon/mesh/nx3=1
              parthenon/meshblock/nx2=1 parthenon/meshblock/nx3=1)
    set_tests_properties(pangu.sync1.reject_non_3d PROPERTIES
      LABELS "numerical-relativity;sync1;input;negative;gpu"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(
      NAME pangu.nr0.minkowski_smoke
      COMMAND $<TARGET_FILE:pangu> -i
              "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/minkowski_geometry.in")
    set_tests_properties(pangu.nr0.minkowski_smoke PROPERTIES
      LABELS "numerical-relativity;nr0;smoke;gpu;sync"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(
      NAME pangu.nr0.reject_fixed_problem
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/smoke/expect_failure.py"
              --needle "accepts only numerical-relativity or synchronized-matter" --
              $<TARGET_FILE:pangu> --check-input -i
              "${CMAKE_CURRENT_SOURCE_DIR}/input/hydro/sod.in")
    set_tests_properties(pangu.nr0.reject_fixed_problem PROPERTIES
      LABELS "numerical-relativity;nr0;negative;assembly;sync")
    add_test(
      NAME pangu.nr0.reject_insufficient_ghosts
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/smoke/expect_failure.py"
              --needle "requires at least order/2+1" --
              $<TARGET_FILE:pangu> --check-input -i
              "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/minkowski_geometry.in"
              numerical_relativity/finite_difference_order=6)
    set_tests_properties(pangu.nr0.reject_insufficient_ghosts PROPERTIES
      LABELS "numerical-relativity;nr0;negative;stencil;sync")
    add_test(NAME pangu.nr1.finite_difference
      COMMAND $<TARGET_FILE:pangu_nr1_finite_difference>)
    set_tests_properties(pangu.nr1.finite_difference PROPERTIES
      LABELS "numerical-relativity;nr1;unit;gpu;finite-difference")
    add_test(NAME pangu.nr1.host_oracle COMMAND $<TARGET_FILE:pangu_nr1_host_oracle>)
    set_tests_properties(pangu.nr1.host_oracle PROPERTIES
      LABELS "numerical-relativity;nr1;unit;host-oracle;finite-difference")
    add_test(NAME pangu.nr1.single_block COMMAND $<TARGET_FILE:pangu_nr1_decomposition> 1)
    set_tests_properties(pangu.nr1.single_block PROPERTIES
      LABELS "numerical-relativity;nr1;gpu;finite-difference;single-block")
    add_test(NAME pangu.nr1.multi_block COMMAND $<TARGET_FILE:pangu_nr1_decomposition> 4)
    set_tests_properties(pangu.nr1.multi_block PROPERTIES
      LABELS "numerical-relativity;nr1;gpu;finite-difference;multi-block")
    if(MPIEXEC_EXECUTABLE)
      add_test(NAME pangu.nr1.mpi_boundaries
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
                $<TARGET_FILE:pangu_nr1_decomposition> 2)
      set_tests_properties(pangu.nr1.mpi_boundaries PROPERTIES
        LABELS "numerical-relativity;nr1;gpu;finite-difference;mpi;boundary")
    endif()
    add_test(NAME pangu.nr1.finite_difference_benchmark
      COMMAND $<TARGET_FILE:pangu_nr1_finite_difference_benchmark>)
    set_tests_properties(pangu.nr1.finite_difference_benchmark PROPERTIES
      LABELS "numerical-relativity;nr1;gpu;finite-difference;benchmark"
      TIMEOUT 120)
    add_test(NAME pangu.nr1.finite_difference_convergence
      COMMAND $<TARGET_FILE:pangu_nr1_finite_difference_convergence>)
    set_tests_properties(pangu.nr1.finite_difference_convergence PROPERTIES
      LABELS "numerical-relativity;nr1;convergence;gpu;finite-difference")
    if(TARGET pangu_nr1_athenak_finite_difference)
      add_test(NAME pangu.nr1.athenak_finite_difference
        COMMAND $<TARGET_FILE:pangu_nr1_athenak_finite_difference>)
      set_tests_properties(pangu.nr1.athenak_finite_difference PROPERTIES
        LABELS "numerical-relativity;nr1;reference;gpu;finite-difference;athenak")
    endif()
    add_test(NAME pangu.nr2.local_rhs COMMAND $<TARGET_FILE:pangu_nr2_local_rhs>)
    set_tests_properties(pangu.nr2.local_rhs PROPERTIES
      LABELS "numerical-relativity;nr2;unit;gpu;z4c;rhs")
    add_test(NAME pangu.sync0.stress_energy COMMAND $<TARGET_FILE:pangu_sync0_stress_energy>)
    set_tests_properties(pangu.sync0.stress_energy PROPERTIES
      LABELS "numerical-relativity;sync0;unit;gpu;tmunu;athenak")
    add_test(NAME pangu.sync1.grhd_state COMMAND $<TARGET_FILE:pangu_sync1_grhd_state>)
    set_tests_properties(pangu.sync1.grhd_state PROPERTIES
      LABELS "numerical-relativity;sync1;unit;gpu;grhd;athenak")
    if(PANGU_ENABLE_HDF5)
      add_test(
        NAME pangu.sync0.analytic_source
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync0_analytic_source.py"
                --pangu $<TARGET_FILE:pangu>
                --input "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/sync0_analytic_source.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync0-analytic-source")
      set_tests_properties(pangu.sync0.analytic_source PROPERTIES
        LABELS "numerical-relativity;sync0;regression;gpu;tmunu;meshdata;stage-time")
      add_test(
        NAME pangu.sync0.zero_source
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/sync0_zero_source.py"
                --pangu $<TARGET_FILE:pangu>
                --input "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/minkowski_geometry.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/sync0-zero-source")
      set_tests_properties(pangu.sync0.zero_source PROPERTIES
        LABELS "numerical-relativity;sync0;regression;gpu;tmunu;vacuum;zero-source")
    endif()
    add_test(NAME pangu.nr3.adm_conversion COMMAND $<TARGET_FILE:pangu_nr3_adm_conversion>)
    set_tests_properties(pangu.nr3.adm_conversion PROPERTIES
      LABELS "numerical-relativity;nr3;unit;gpu;adm;conversion")
    add_test(NAME pangu.nr3.constraints COMMAND $<TARGET_FILE:pangu_nr3_constraints>)
    set_tests_properties(pangu.nr3.constraints PROPERTIES
      LABELS "numerical-relativity;nr3;unit;gpu;constraints")
    add_test(NAME pangu.nr3.schwarzschild
      COMMAND $<TARGET_FILE:pangu_nr3_schwarzschild>)
    set_tests_properties(pangu.nr3.schwarzschild PROPERTIES
      LABELS "numerical-relativity;nr3;unit;gpu;analytic;schwarzschild")
    add_test(NAME pangu.nr3.kerr_schild
      COMMAND $<TARGET_FILE:pangu_nr3_kerr_schild>)
    set_tests_properties(pangu.nr3.kerr_schild PROPERTIES
      LABELS "numerical-relativity;nr3;unit;gpu;analytic;kerr-schild")
    add_test(NAME pangu.nr5.boosted_puncture
      COMMAND $<TARGET_FILE:pangu_nr5_boosted_puncture>)
    add_test(NAME pangu.nr6.weyl COMMAND $<TARGET_FILE:pangu_nr6_weyl>)
    set_tests_properties(pangu.nr6.weyl PROPERTIES
      LABELS "unit;gpu;numerical-relativity;nr6;weyl")
    add_test(NAME pangu.nr6.amr COMMAND $<TARGET_FILE:pangu_nr6_amr>)
    set_tests_properties(pangu.nr6.amr PROPERTIES
      LABELS "unit;gpu;numerical-relativity;nr6;amr;athenak")
    set_tests_properties(pangu.nr5.boosted_puncture PROPERTIES
      LABELS "numerical-relativity;nr5;unit;gpu;analytic;puncture")
    add_test(NAME pangu.nr7.bowen_york COMMAND $<TARGET_FILE:pangu_nr7_bowen_york>)
    set_tests_properties(pangu.nr7.bowen_york PROPERTIES
      LABELS "numerical-relativity;nr7;unit;gpu;puncture;bowen-york")
    add_test(NAME pangu.nr7.spectral_grid COMMAND $<TARGET_FILE:pangu_nr7_spectral_grid>)
    set_tests_properties(pangu.nr7.spectral_grid PROPERTIES
      LABELS "numerical-relativity;nr7;unit;puncture;spectral;compactification")
    add_test(NAME pangu.nr7.hamiltonian_solver
      COMMAND $<TARGET_FILE:pangu_nr7_hamiltonian_solver>)
    set_tests_properties(pangu.nr7.hamiltonian_solver PROPERTIES
      LABELS "numerical-relativity;nr7;unit;puncture;spectral;hamiltonian")
    add_test(NAME pangu.nr7.puncture_interpolation
      COMMAND $<TARGET_FILE:pangu_nr7_puncture_interpolation>)
    set_tests_properties(pangu.nr7.puncture_interpolation PROPERTIES
      LABELS "numerical-relativity;nr7;unit;puncture;interpolation")
    if(PANGU_ENABLE_HDF5)
      add_test(
        NAME pangu.nr6.waveform_multiblock
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/nr6_waveform_regression.py"
                --pangu $<TARGET_FILE:pangu>
                --pangu-input
                  "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/nr6_linear_wave_diagnostics.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/nr6-waveform-multiblock")
      set_tests_properties(pangu.nr6.waveform_multiblock PROPERTIES
        LABELS "numerical-relativity;nr6;regression;gpu;weyl;waveform;multi-block"
        TIMEOUT 300)
      if(PANGU_ENABLE_TWOPUNCTURES)
        add_test(
          NAME pangu.nr6.two_punctures
          COMMAND "${Python3_EXECUTABLE}"
                  "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/nr6_two_punctures_regression.py"
                  --executable $<TARGET_FILE:pangu>
                  --input
                    "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/nr6_two_punctures_initial.in"
                  --workdir "${CMAKE_CURRENT_BINARY_DIR}/nr6-two-punctures")
        set_tests_properties(pangu.nr6.two_punctures PROPERTIES
          LABELS "numerical-relativity;nr6;regression;gpu;two-punctures;tracker"
          TIMEOUT 300)
      endif()
      add_test(
        NAME pangu.nr3.physical_boundary_mask
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/nr3_boundary_mask.py"
                --executable $<TARGET_FILE:pangu>
                --input
                  "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/nr3_schwarzschild_ks_outflow.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/nr3-boundary-mask")
      set_tests_properties(pangu.nr3.physical_boundary_mask PROPERTIES
        LABELS "numerical-relativity;nr3;regression;gpu;boundary;mask"
        TIMEOUT 300)
      add_test(
        NAME pangu.nr4.reflecting_tensor_parity
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/nr4_reflecting_boundaries.py"
                --executable $<TARGET_FILE:pangu>
                --input
                  "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/nr4_reflecting_boundaries.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/nr4-reflecting-boundaries")
      set_tests_properties(pangu.nr4.reflecting_tensor_parity PROPERTIES
        LABELS "numerical-relativity;nr4;regression;gpu;boundary;reflection"
        TIMEOUT 300)
      add_test(
        NAME pangu.nr4.extrapolation_order2
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/nr4_extrapolation_boundaries.py"
                --executable $<TARGET_FILE:pangu>
                --input
                  "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/nr4_extrapolation_boundaries.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/nr4-extrapolation-order2"
                --nghost 2 --order 2)
      set_tests_properties(pangu.nr4.extrapolation_order2 PROPERTIES
        LABELS "numerical-relativity;nr4;nr6;regression;gpu;boundary;extrapolation;athenak"
        TIMEOUT 300)
      foreach(order IN ITEMS 3 4)
        add_test(
          NAME pangu.nr4.extrapolation_order${order}
          COMMAND "${Python3_EXECUTABLE}"
                  "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/nr4_extrapolation_boundaries.py"
                  --executable $<TARGET_FILE:pangu>
                  --input
                    "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/nr4_extrapolation_boundaries_order6.in"
                  --workdir "${CMAKE_CURRENT_BINARY_DIR}/nr4-extrapolation-order${order}"
                  --nghost 4 --order ${order})
        set_tests_properties(pangu.nr4.extrapolation_order${order} PROPERTIES
          LABELS "numerical-relativity;nr4;nr6;regression;gpu;boundary;extrapolation"
          TIMEOUT 300)
      endforeach()
    endif()
    add_test(
      NAME pangu.nr2.minkowski_1000_steps
      COMMAND $<TARGET_FILE:pangu> -i
              "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/minkowski_geometry.in"
              parthenon/time/nlim=1000 parthenon/time/tlim=1000
              parthenon/output1/dt=-1 parthenon/output2/dt=-1)
    set_tests_properties(pangu.nr2.minkowski_1000_steps PROPERTIES
      LABELS "numerical-relativity;nr2;regression;gpu;minkowski"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      TIMEOUT 300)
    if(PANGU_ENABLE_HDF5)
      add_test(
        NAME pangu.nr2.evolution
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/nr2_evolution.py"
                --executable $<TARGET_FILE:pangu>
                --linear-input
                  "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/linear_wave.in"
                --gauge-input
                  "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/gauge_wave.in"
                --robust-input
                  "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/robust_stability.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/nr2-evolution")
      set_tests_properties(pangu.nr2.evolution PROPERTIES
        LABELS "numerical-relativity;nr2;regression;gpu;convergence;z4c"
        TIMEOUT 1800)
      if(PANGU_ATHENAK_EXECUTABLE AND EXISTS "${PANGU_ATHENAK_EXECUTABLE}"
         AND PANGU_ATHENAK_GAUGE_EXECUTABLE
         AND EXISTS "${PANGU_ATHENAK_GAUGE_EXECUTABLE}")
        add_test(
          NAME pangu.nr2.athenak_multitime
          COMMAND "${Python3_EXECUTABLE}"
                  "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/nr2_athenak_multitime.py"
                  --pangu $<TARGET_FILE:pangu>
                  --athenak-linear "${PANGU_ATHENAK_EXECUTABLE}"
                  --athenak-gauge "${PANGU_ATHENAK_GAUGE_EXECUTABLE}"
                  --pangu-linear-input
                    "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/linear_wave.in"
                  --athenak-linear-input
                    "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/nr2_athenak_linear_wave.athinput"
                  --pangu-gauge-input
                    "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/gauge_wave.in"
                  --athenak-gauge-input
                    "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/nr2_athenak_gauge_wave.athinput"
                  --workdir "${CMAKE_CURRENT_BINARY_DIR}/nr2-athenak-multitime")
        set_tests_properties(pangu.nr2.athenak_multitime PROPERTIES
          LABELS "numerical-relativity;nr2;reference;gpu;multitime;athenak"
          TIMEOUT 1800)
      endif()
      if(PANGU_ATHENAK_ROBUST_EXECUTABLE
         AND EXISTS "${PANGU_ATHENAK_ROBUST_EXECUTABLE}")
        add_test(
          NAME pangu.nr2.athenak_robust
          COMMAND "${Python3_EXECUTABLE}"
                  "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/nr2_athenak_robust.py"
                  --pangu $<TARGET_FILE:pangu>
                  --athenak "${PANGU_ATHENAK_ROBUST_EXECUTABLE}"
                  --pangu-input
                    "${CMAKE_CURRENT_SOURCE_DIR}/input/numerical_relativity/robust_stability.in"
                  --athenak-input
                    "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/nr2_athenak_robust_stability.athinput"
                  --workdir "${CMAKE_CURRENT_BINARY_DIR}/nr2-athenak-robust")
        set_tests_properties(pangu.nr2.athenak_robust PROPERTIES
          LABELS "numerical-relativity;nr2;reference;gpu;robust;athenak"
          TIMEOUT 1800)
      endif()
    endif()
  endif()

  if(PANGU_ENABLE_HDF5)
    if(PHYSICS STREQUAL "gr" AND METRIC STREQUAL "mks" AND MODE STREQUAL "static")
      add_test(
        NAME pangu.radiation.rc3_restart
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/radiation_cooling_restart_regression.py"
                --executable $<TARGET_FILE:pangu>
                --input
                  "${CMAKE_CURRENT_SOURCE_DIR}/input/relativity/mks_torus_sane_2d_512x256_t5000.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/radiation-rc3-restart")
      set_tests_properties(pangu.radiation.rc3_restart PROPERTIES
        LABELS "radiation;rc3;gpu;mks;static;meshdata;multiblock;restart"
        TIMEOUT 300)
    endif()

    # The EH-2/EH-3 assembly regressions exercise fixed-background GRMHD
    # inputs.  A z4c/sync executable deliberately accepts only numerical-
    # relativity and synchronized-matter problems, so registering a fixed-
    # geometry input there would test an unsupported assembly rather than
    # electron-module independence. Cooling builds reject electron thermodynamics.
    if(METRIC STREQUAL "cks" OR METRIC STREQUAL "mks")
      if(METRIC STREQUAL "cks" AND MODE STREQUAL "dynamic")
        set(PANGU_EH2_ASSEMBLY_INPUT
            "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/electron_cks_dynamic_smr.in")
      elseif(METRIC STREQUAL "cks")
        set(PANGU_EH2_ASSEMBLY_INPUT
            "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/electron_cks_static.in")
      else()
        set(PANGU_EH2_ASSEMBLY_INPUT
            "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/electron_mks_heating.in")
      endif()
      add_test(
        NAME pangu.eh2.assembly_heating
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/electron_heating_assembly_regression.py"
                --executable $<TARGET_FILE:pangu>
                --input "${PANGU_EH2_ASSEMBLY_INPUT}"
                --metric "${METRIC}"
                --mode "${MODE}"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/eh2-assembly-heating")
      set_tests_properties(pangu.eh2.assembly_heating PROPERTIES
        LABELS "electrons;eh2;regression;gpu;assembly;${METRIC};${MODE}")

      add_test(
        NAME pangu.eh3.assembly_models
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/electron_microphysics_assembly_regression.py"
                --executable $<TARGET_FILE:pangu>
                --input "${PANGU_EH2_ASSEMBLY_INPUT}"
                --metric "${METRIC}"
                --mode "${MODE}"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/eh3-assembly-models")
      set_tests_properties(pangu.eh3.assembly_models PROPERTIES
        LABELS "electrons;eh3;regression;gpu;assembly;${METRIC};${MODE}")
    endif()
  endif()

  if(PHYSICS STREQUAL "gr" AND METRIC STREQUAL "cks" AND MODE STREQUAL "dynamic"
     AND PANGU_ENABLE_HDF5)
    # The legacy EH-1/EH-2 drivers combine SR-only analytic inputs with a
    # CKS/GR SMR input. They cannot be registered against one executable now
    # that PHYSICS is a compile-time choice. The distribution regression
    # matrix splits those drivers by physics assembly; retain the genuinely
    # CKS/GR SANE integration gate here.
    add_test(
      NAME pangu.eh4.cks_dynamic_sane
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/electron_cks_sane_regression.py"
              --executable $<TARGET_FILE:pangu>
              --input
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/input/electron_cks_dynamic_sane.in"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/eh4-cks-dynamic-sane")
    set_tests_properties(pangu.eh4.cks_dynamic_sane PROPERTIES
      LABELS "electrons;eh4;regression;gpu;cks;dynamic;grmhd;sane;restart;smr")
  endif()

  add_test(
    NAME pangu.stage5.acceptance_records
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/phase5_acceptance_gate.py"
            --records "${CMAKE_CURRENT_SOURCE_DIR}/tst/reference/comparisons")
  set_tests_properties(pangu.stage5.acceptance_records PROPERTIES
    LABELS "stage5;regression;comparison;multitime")

  # MKS and synchronized Z4c executables accept only problems on their own
  # background, and cooling executables accept only GRMHD, so the Newtonian
  # problem regressions belong to other assemblies.
  if(NOT (PHYSICS STREQUAL "gr" AND (METRIC STREQUAL "mks" OR MODE STREQUAL "sync")))
    add_test(
      NAME pangu.stage3.mhd_problems
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/mhd_problem_smoke.py"
              --executable $<TARGET_FILE:pangu>
              --input-dir "${CMAKE_CURRENT_SOURCE_DIR}/input/mhd"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/stage3-mhd-problems")
    set_tests_properties(pangu.stage3.mhd_problems PROPERTIES
      LABELS "stage3;regression;problems")

    add_test(
      NAME pangu.stage3.cpaw_convergence
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/mhd_cpaw_convergence.py"
              --executable $<TARGET_FILE:pangu>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/input/mhd/cpaw.in"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/stage3-cpaw-convergence")
    set_tests_properties(pangu.stage3.cpaw_convergence PROPERTIES
      LABELS "stage3;regression;convergence")

    add_test(
      NAME pangu.stage2.hydro_algorithm_matrix
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/hydro_algorithm_matrix.py"
              --executable $<TARGET_FILE:pangu>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/input/hydro/sod.in"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/stage2-hydro-matrix")
    set_tests_properties(pangu.stage2.hydro_algorithm_matrix PROPERTIES
      LABELS "stage2;regression")

    add_test(
      NAME pangu.stage2.reject_isothermal_hllc
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tst/smoke/expect_failure.py"
              --needle "Isothermal Hydro does not support the HLLC Riemann solver" --
              $<TARGET_FILE:pangu> --check-input -i
              "${CMAKE_CURRENT_SOURCE_DIR}/input/hydro/sod.in"
              hydro/eos=isothermal hydro/rsolver=hllc)
    set_tests_properties(pangu.stage2.reject_isothermal_hllc PROPERTIES
      LABELS "stage2;negative")

    set(_pangu_hydro_smoke_command
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/hydro_problem_smoke.py"
        --executable $<TARGET_FILE:pangu>
        --input-dir "${CMAKE_CURRENT_SOURCE_DIR}/input/hydro"
        --workdir "${CMAKE_CURRENT_BINARY_DIR}/stage2-hydro-problems")
    if(PANGU_ENABLE_HDF5)
      list(APPEND _pangu_hydro_smoke_command --hdf5-enabled)
    endif()
    add_test(NAME pangu.stage2.hydro_problems COMMAND ${_pangu_hydro_smoke_command})
    set_tests_properties(pangu.stage2.hydro_problems PROPERTIES
      LABELS "stage2;regression;problems")

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/athenak/build-stage0-serial/src/athena" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/athenapk/build-stage0-serial/bin/athenaPK")
      add_test(
        NAME pangu.stage2.three_way_sod
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/three_way_sod.py"
                --pangu $<TARGET_FILE:pangu>
                --pangu-input "${CMAKE_CURRENT_SOURCE_DIR}/input/hydro/sod.in"
                --athenak "${CMAKE_CURRENT_SOURCE_DIR}/athenak/build-stage0-serial/src/athena"
                --athenak-input "${CMAKE_CURRENT_SOURCE_DIR}/athenak/inputs/hydro/sod.athinput"
                --athenapk "${CMAKE_CURRENT_SOURCE_DIR}/athenapk/build-stage0-serial/bin/athenaPK"
                --athenapk-input "${CMAKE_CURRENT_SOURCE_DIR}/athenapk/inputs/sod.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/stage2-three-way-sod")
      set_tests_properties(pangu.stage2.three_way_sod PROPERTIES
        LABELS "stage2;comparison;regression")
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/athenak/build-stage0-serial/src/athena")
      add_test(
        NAME pangu.stage2.athenak_hydro_matrix
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/athenak_hydro_matrix.py"
                --pangu $<TARGET_FILE:pangu>
                --pangu-sod "${CMAKE_CURRENT_SOURCE_DIR}/input/hydro/sod.in"
                --pangu-advection "${CMAKE_CURRENT_SOURCE_DIR}/input/hydro/hydro_advection.in"
                --athenak "${CMAKE_CURRENT_SOURCE_DIR}/athenak/build-stage0-serial/src/athena"
                --athenak-sod "${CMAKE_CURRENT_SOURCE_DIR}/athenak/inputs/hydro/sod.athinput"
                --athenak-advection
                  "${CMAKE_CURRENT_SOURCE_DIR}/athenak/inputs/tests/advect_hyd.athinput"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/stage2-athenak-hydro-matrix")
      set_tests_properties(pangu.stage2.athenak_hydro_matrix PROPERTIES
        LABELS "stage2;comparison;regression")
    endif()

    if(PANGU_ENABLE_HDF5 AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/athenak/build-stage0-serial/src/athena" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/athenapk/build-stage2-hdf/bin/athenaPK")
      add_test(
        NAME pangu.stage2.hydro_multitime_references
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/hydro_multitime_references.py"
                --pangu $<TARGET_FILE:pangu>
                --pangu-sod "${CMAKE_CURRENT_SOURCE_DIR}/input/hydro/sod.in"
                --pangu-advection "${CMAKE_CURRENT_SOURCE_DIR}/input/hydro/hydro_advection.in"
                --athenak "${CMAKE_CURRENT_SOURCE_DIR}/athenak/build-stage0-serial/src/athena"
                --athenak-sod "${CMAKE_CURRENT_SOURCE_DIR}/athenak/inputs/hydro/sod.athinput"
                --athenak-advection
                  "${CMAKE_CURRENT_SOURCE_DIR}/athenak/inputs/tests/advect_hyd.athinput"
                --athenapk "${CMAKE_CURRENT_SOURCE_DIR}/athenapk/build-stage2-hdf/bin/athenaPK"
                --athenapk-sod "${CMAKE_CURRENT_SOURCE_DIR}/athenapk/inputs/sod.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/stage2-hydro-multitime-references")
      set_tests_properties(pangu.stage2.hydro_multitime_references PROPERTIES
        LABELS "stage2;comparison;regression;multitime")
    endif()

    if(PANGU_ENABLE_HDF5)
      add_test(
        NAME pangu.stage2.hydro_restart
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/hydro_restart.py"
                --executable $<TARGET_FILE:pangu>
                --input "${CMAKE_CURRENT_SOURCE_DIR}/input/hydro/blast.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/stage2-hydro-restart")
      set_tests_properties(pangu.stage2.hydro_restart PROPERTIES
        LABELS "stage2;regression;restart")

      add_test(
        NAME pangu.stage3.mhd_restart
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tst/regression/mhd_restart.py"
                --executable $<TARGET_FILE:pangu>
                --input "${CMAKE_CURRENT_SOURCE_DIR}/input/mhd/field_loop.in"
                --workdir "${CMAKE_CURRENT_BINARY_DIR}/stage3-mhd-restart")
      set_tests_properties(pangu.stage3.mhd_restart PROPERTIES
        LABELS "stage3;regression;restart")
    endif()
  endif()
endif()
