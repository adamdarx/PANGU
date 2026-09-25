string(TOLOWER "${PHYSICS}" PHYSICS)
string(TOLOWER "${METRIC}" METRIC)
string(TOLOWER "${MODE}" MODE)
string(TOLOWER "${ESTIMATOR}" ESTIMATOR)
set(PHYSICS "${PHYSICS}" CACHE STRING "Compile-time relativistic physics" FORCE)
set(METRIC "${METRIC}" CACHE STRING "Compile-time spacetime metric" FORCE)
set(MODE "${MODE}" CACHE STRING "Compile-time geometry access mode" FORCE)
set(ESTIMATOR "${ESTIMATOR}" CACHE STRING "Compile-time relativistic timestep estimator" FORCE)

if(NOT PHYSICS STREQUAL "sr" AND NOT PHYSICS STREQUAL "gr")
  message(FATAL_ERROR
    "Unsupported PHYSICS=${PHYSICS}. Supported relativistic physics: sr, gr.")
endif()

if(PHYSICS STREQUAL "sr")
  set(PANGU_PHYSICS_SR ON)
  set(PANGU_PHYSICS_GR OFF)
  if(NOT METRIC STREQUAL "")
    message(WARNING
      "PHYSICS=sr ignores METRIC=${METRIC}; special relativity uses an internal flat-spacetime geometry.")
  endif()
  if(NOT MODE STREQUAL "")
    message(WARNING
      "PHYSICS=sr ignores MODE=${MODE}; no GR geometry access mode is assembled.")
  endif()
else()
  set(PANGU_PHYSICS_SR OFF)
  set(PANGU_PHYSICS_GR ON)
  if(METRIC STREQUAL "")
    message(WARNING "PHYSICS=gr was configured without METRIC; defaulting to METRIC=cks.")
    set(METRIC "cks")
  endif()
  if(MODE STREQUAL "")
    message(WARNING "PHYSICS=gr was configured without MODE; defaulting to MODE=dynamic.")
    set(MODE "dynamic")
  endif()
endif()

if(PHYSICS STREQUAL "gr" AND METRIC STREQUAL "mks" AND NOT MODE STREQUAL "static")
  message(FATAL_ERROR
    "METRIC=mks requires MODE=static; MKS geometry is stored and refinement is disabled.")
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/estimator")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/estimator/estimator.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/estimator/estimator.h"
  @ONLY)

if(PHYSICS STREQUAL "sr")
  set(PANGU_METRIC_TYPE "metric::Minkowski")
  set(PANGU_METRIC_NAME "none")
  set(PANGU_METRIC_HEADER "geometry/metric/minkowski.h")
  set(PANGU_METRIC_SOURCES geometry/metric/minkowski.h)
  set(PANGU_MODE_TYPE "DynamicMode")
  set(PANGU_MODE_NAME "none")
  set(PANGU_MODE_HEADER "geometry/mode/dynamic.h")
  set(PANGU_MODE_SOURCES geometry/mode/dynamic.h)
elseif(METRIC STREQUAL "cks")
  set(PANGU_METRIC_TYPE "metric::CKS")
  set(PANGU_METRIC_NAME "cks")
  set(PANGU_METRIC_HEADER "geometry/metric/cks.h")
  set(PANGU_METRIC_SOURCES
    geometry/metric/cks.cc
    geometry/metric/cks.h)
elseif(METRIC STREQUAL "mks")
  set(PANGU_METRIC_TYPE "metric::MKS")
  set(PANGU_METRIC_NAME "mks")
  set(PANGU_METRIC_HEADER "geometry/metric/mks.h")
  set(PANGU_METRIC_SOURCES
    geometry/metric/mks.cc
    geometry/metric/mks.h)
elseif(METRIC STREQUAL "z4c")
  set(PANGU_METRIC_TYPE "metric::Z4c")
  set(PANGU_METRIC_NAME "z4c")
  set(PANGU_METRIC_HEADER "geometry/metric/z4c.h")
  set(PANGU_METRIC_SOURCES geometry/metric/z4c.h)
else()
  message(FATAL_ERROR
    "Unsupported METRIC=${METRIC}. Supported metrics: cks, mks, z4c.")
endif()

if(PHYSICS STREQUAL "sr")
  # The internal flat geometry above is a zero-storage compile-time façade.
elseif(MODE STREQUAL "dynamic")
  set(PANGU_MODE_TYPE "DynamicMode")
  set(PANGU_MODE_NAME "dynamic")
  set(PANGU_MODE_HEADER "geometry/mode/dynamic.h")
  set(PANGU_MODE_SOURCES geometry/mode/dynamic.h)
elseif(MODE STREQUAL "static")
  set(PANGU_MODE_TYPE "StaticMode")
  set(PANGU_MODE_NAME "static")
  set(PANGU_MODE_HEADER "geometry/mode/static.h")
  set(PANGU_MODE_SOURCES geometry/mode/static.h)
elseif(MODE STREQUAL "sync")
  set(PANGU_MODE_TYPE "SyncMode")
  set(PANGU_MODE_NAME "sync")
  set(PANGU_MODE_HEADER "geometry/mode/sync.h")
  set(PANGU_MODE_SOURCES geometry/mode/sync.h)
else()
  message(FATAL_ERROR
    "Unsupported MODE=${MODE}. Supported geometry modes: dynamic, static, sync.")
endif()

if(PHYSICS STREQUAL "gr" AND
   ((METRIC STREQUAL "z4c" AND NOT MODE STREQUAL "sync") OR
   (MODE STREQUAL "sync" AND NOT METRIC STREQUAL "z4c")))
  message(FATAL_ERROR
    "The evolving Z4c metric and synchronized geometry mode must be selected together: "
    "use -DMETRIC=z4c -DMODE=sync. Existing dynamic mode remains analytic on-demand access.")
endif()

set(PANGU_NUMERICAL_RELATIVITY_SOURCES)
if(PHYSICS STREQUAL "gr" AND METRIC STREQUAL "z4c" AND MODE STREQUAL "sync")
  list(APPEND PANGU_NUMERICAL_RELATIVITY_SOURCES
    z4c/core/component_indices.h
    z4c/core/adm_conversion.h
    z4c/initial_data/analytic.h
    z4c/evolution/constraints.h
    z4c/core/finite_difference.h
    z4c/evolution/amr.h
    z4c/diagnostics/diagnostics.cc
    z4c/diagnostics/diagnostics.h
    z4c/evolution/package.cc
    z4c/evolution/package.h
    z4c/coupling/stress_energy.h
    z4c/coupling/sync_grhd.cc
    z4c/coupling/sync_grhd.h
    z4c/coupling/sync_grmhd.cc
    z4c/coupling/sync_grmhd.h
    z4c/evolution/tasks.cc
    z4c/evolution/tasks.h
    z4c/initial_data/tov.cc
    z4c/initial_data/tov.h
    z4c/puncture/puncture_data.h
    z4c/puncture/bowen_york.h
    z4c/puncture/spectral_solver.h
    z4c/puncture/spectral_solver.cc
    z4c/diagnostics/weyl.h
    z4c/evolution/z4c_rhs.h
    pgen/numerical_relativity.cc
    pgen/numerical_relativity.h)
endif()

set(PANGU_RADIATION_PROBLEM_SOURCES
    pgen/radiation_beam.cc
    pgen/radiation_beam.h
    pgen/radiation_diffusion.cc
    pgen/radiation_diffusion.h
    pgen/radiation_hohlraum.cc
    pgen/radiation_hohlraum.h
    pgen/radiation_linear_wave.cc
    pgen/radiation_linear_wave.h
    pgen/radiation_relax.cc
    pgen/radiation_relax.h
    pgen/radiation_shadow.cc
    pgen/radiation_shadow.h)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/geometry/geometry_assembly.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/geometry_assembly.h"
  @ONLY)

set(PANGU_SOURCES
  main.cc
  eos/ideal_gas.h
  pangu.h
  app/application_input.cc
  app/application_input.h
  app/command_line.h
  app/package_registry.cc
  app/package_registry.h
  app/parameter_compat.cc
  app/parameter_compat.h
  app/problem_registry.cc
  app/problem_registry.h
  plugin/registry.cc
  plugin/registry.h
  pangu/plugin_api/diffusion.h
  pangu/plugin_api/plugin.h
  geometry/geometry.cc
  geometry/geometry.h
  geometry/geometry_assembly.h.in
  geometry/assembly_contract.cc
  geometry/metric/minkowski.h
  ${PANGU_METRIC_SOURCES}
  ${PANGU_MODE_SOURCES}
  driver/driver.cc
  driver/driver.h
  driver/contributor.h
  driver/stage_extension.h
  driver/task_assembly.cc
  driver/task_assembly.h
  eos/newtonian_eos.h
  eos/newtonian_mhd_eos.h
  eos/relativistic_eos.h
  hydro/hydro_package.cc
  hydro/hydro_package.h
  hydro/task_contribution.cc
  hydro/task_contribution.h
  hydro/hydro_types.h
  mhd/mhd_package.cc
  mhd/mhd_package.h
  mhd/face_update.cc
  mhd/task_contribution.cc
  mhd/task_contribution.h
  mhd/passive_transport.h
  mhd/mhd_types.h
  estimator/estimator.h.in
  estimator/registry.h
  estimator/model/light.h
  estimator/model/wave.h
  ${PANGU_NUMERICAL_RELATIVITY_SOURCES}
  ${PANGU_RADIATION_PROBLEM_SOURCES}
  reconstruct/hydro_reconstruction.h
  reconstruct/registry.h
  riemann/registry.h
  riemann/newtonian_hydro/hllc.h
  riemann/newtonian_hydro/hlle.h
  riemann/newtonian_hydro/llf.h
  riemann/newtonian_hydro/roe.h
  riemann/newtonian_mhd/electric_field.h
  riemann/newtonian_mhd/hlld.h
  riemann/newtonian_mhd/hlle.h
  riemann/newtonian_mhd/llf.h
  riemann/relativistic/hlle.h
  riemann/relativistic/llf.h
  riemann/relativistic_mhd/hlle.h
  riemann/relativistic_mhd/llf.h
  reconstruct/model/donor_cell.h
  reconstruct/model/plm.h
  reconstruct/model/ppm.h
  reconstruct/model/ppmc.h
  reconstruct/model/wenoz.h
  relativity/relativistic_hydro.cc
  relativity/relativistic_hydro.h
  relativity/physics_assembly.h
  relativity/relativistic_mhd.cc
  relativity/relativistic_mhd.h
  srcterms/source_terms.cc
  srcterms/source_terms.h
  units/units.cc
  units/units.h
  pgen/hydro_problems.cc
  pgen/hydro_problems.h
  pgen/mhd_problems.cc
  pgen/mhd_problems.h)

pangu_prepare_plugins(PANGU_PLUGIN_SOURCES PANGU_PLUGIN_INCLUDE_DIRS)
list(APPEND PANGU_SOURCES ${PANGU_PLUGIN_SOURCES})
include_directories(${PANGU_PLUGIN_INCLUDE_DIRS})

if(PANGU_ENABLE_CUDA)
  find_program(PANGU_HOST_CXX_COMPILER NAMES g++ clang++ c++)
  if(NOT PANGU_HOST_CXX_COMPILER)
    message(FATAL_ERROR "A host C++ compiler is required to compile the CLI11 command-line entry")
  endif()
  set(PANGU_COMMAND_LINE_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/command_line_host.o")
  add_custom_command(
    OUTPUT "${PANGU_COMMAND_LINE_OBJECT}"
    COMMAND "${PANGU_HOST_CXX_COMPILER}" -std=c++20 -O2
            -I"${CMAKE_CURRENT_SOURCE_DIR}"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/app/command_line.cc"
            -o "${PANGU_COMMAND_LINE_OBJECT}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/app/command_line.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/app/command_line.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/CLI11.h"
    COMMENT "Building the CLI11 entry with the host compiler")
  set_source_files_properties("${PANGU_COMMAND_LINE_OBJECT}"
    PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  list(APPEND PANGU_SOURCES "${PANGU_COMMAND_LINE_OBJECT}")
else()
  list(APPEND PANGU_SOURCES app/command_line.cc)
endif()

add_executable(pangu ${PANGU_SOURCES})

target_include_directories(pangu PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/../generated"
  ${PANGU_PLUGIN_INCLUDE_DIRS})
target_link_libraries(pangu PRIVATE Parthenon::parthenon)
target_compile_features(pangu PRIVATE cxx_std_20)

if(PANGU_ENABLE_TWOPUNCTURES)
  if(NOT (PHYSICS STREQUAL "gr" AND METRIC STREQUAL "z4c" AND MODE STREQUAL "sync"))
    message(FATAL_ERROR
      "PANGU_ENABLE_TWOPUNCTURES requires -DPHYSICS=gr -DMETRIC=z4c -DMODE=sync")
  endif()
  find_path(PANGU_TWOPUNCTURES_INCLUDE_DIR
    NAMES TwoPunctures.h
    HINTS "${PANGU_TWOPUNCTURES_ROOT}/include"
    REQUIRED)
  find_library(PANGU_TWOPUNCTURES_LIBRARY
    NAMES TwoPunctures
    HINTS "${PANGU_TWOPUNCTURES_ROOT}/lib"
    REQUIRED)
  find_package(GSL REQUIRED)
  target_include_directories(pangu PRIVATE "${PANGU_TWOPUNCTURES_INCLUDE_DIR}")
  target_link_libraries(pangu PRIVATE
    "${PANGU_TWOPUNCTURES_LIBRARY}" GSL::gsl GSL::gslcblas)
endif()

if(PANGU_WARNINGS_AS_ERRORS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pangu PRIVATE -Werror=return-type)
endif()

install(TARGETS pangu RUNTIME DESTINATION bin)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/input/" DESTINATION share/pangu/input)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pangu/plugin_api/"
        DESTINATION include/pangu/plugin_api)

if(BUILD_TESTING AND PANGU_ENABLE_TESTING)
  if(PANGU_ENABLE_CUDA)
    add_executable(pangu_command_line
      "${CMAKE_SOURCE_DIR}/tst/unit/command_line.cc"
      "${PANGU_COMMAND_LINE_OBJECT}")
  else()
    add_executable(pangu_command_line
      "${CMAKE_SOURCE_DIR}/tst/unit/command_line.cc"
      app/command_line.cc)
  endif()
  target_include_directories(pangu_command_line PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(pangu_command_line PRIVATE cxx_std_20)
  add_executable(pangu_plugin_registry
    "${CMAKE_SOURCE_DIR}/tst/unit/plugin_registry.cc"
    plugin/registry.cc)
  target_include_directories(pangu_plugin_registry PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_BINARY_DIR}/generated"
    ${PANGU_PLUGIN_INCLUDE_DIRS})
  target_link_libraries(pangu_plugin_registry PRIVATE Parthenon::parthenon)
  target_compile_definitions(pangu_plugin_registry PRIVATE PANGU_PLUGIN_REGISTRY_TESTING)
  target_compile_features(pangu_plugin_registry PRIVATE cxx_std_20)
  add_executable(pangu_hydro_kernels "${CMAKE_SOURCE_DIR}/tst/unit/hydro_kernels.cc")
  target_include_directories(pangu_hydro_kernels PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(pangu_hydro_kernels PRIVATE Parthenon::parthenon)
  target_compile_features(pangu_hydro_kernels PRIVATE cxx_std_20)
  add_executable(pangu_mhd_kernels "${CMAKE_SOURCE_DIR}/tst/unit/mhd_kernels.cc")
  target_include_directories(pangu_mhd_kernels PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(pangu_mhd_kernels PRIVATE Parthenon::parthenon)
  target_compile_features(pangu_mhd_kernels PRIVATE cxx_std_20)
  add_executable(pangu_phase5_relativity "${CMAKE_SOURCE_DIR}/tst/unit/phase5_relativity.cc")
  target_include_directories(pangu_phase5_relativity PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(pangu_phase5_relativity PRIVATE Parthenon::parthenon)
  target_compile_features(pangu_phase5_relativity PRIVATE cxx_std_20)
  add_executable(pangu_mks_geometry "${CMAKE_SOURCE_DIR}/tst/unit/mks_geometry.cc")
  target_include_directories(pangu_mks_geometry PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(pangu_mks_geometry PRIVATE Parthenon::parthenon)
  target_compile_features(pangu_mks_geometry PRIVATE cxx_std_20)
  add_executable(pangu_electron_transport "${CMAKE_SOURCE_DIR}/tst/unit/electron_transport.cc")
  target_include_directories(pangu_electron_transport PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(pangu_electron_transport PRIVATE Parthenon::parthenon)
  target_compile_features(pangu_electron_transport PRIVATE cxx_std_20)
  add_executable(pangu_driver_contributor "${CMAKE_SOURCE_DIR}/tst/unit/driver_contributor.cc")
  target_include_directories(pangu_driver_contributor PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(pangu_driver_contributor PRIVATE Parthenon::parthenon)
  target_compile_features(pangu_driver_contributor PRIVATE cxx_std_20)
  add_executable(pangu_estimator_registry "${CMAKE_SOURCE_DIR}/tst/unit/estimator_registry.cc")
  target_include_directories(pangu_estimator_registry PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(pangu_estimator_registry PRIVATE Parthenon::parthenon)
  target_compile_features(pangu_estimator_registry PRIVATE cxx_std_20)
  add_executable(pangu_reconstruct_registry
    "${CMAKE_SOURCE_DIR}/tst/unit/reconstruct_registry.cc")
  target_include_directories(pangu_reconstruct_registry PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(pangu_reconstruct_registry PRIVATE Parthenon::parthenon)
  target_compile_features(pangu_reconstruct_registry PRIVATE cxx_std_20)
  add_executable(pangu_electron_heating "${CMAKE_SOURCE_DIR}/tst/unit/electron_heating.cc")
  target_include_directories(pangu_electron_heating PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(pangu_electron_heating PRIVATE Parthenon::parthenon)
  target_compile_features(pangu_electron_heating PRIVATE cxx_std_20)
  add_executable(pangu_electron_microphysics
    "${CMAKE_SOURCE_DIR}/tst/unit/electron_microphysics.cc")
  target_include_directories(pangu_electron_microphysics PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(pangu_electron_microphysics PRIVATE Parthenon::parthenon)
  target_compile_features(pangu_electron_microphysics PRIVATE cxx_std_20)
  add_executable(pangu_radiation_cooling
    "${CMAKE_SOURCE_DIR}/tst/unit/radiation_cooling.cc")
  target_include_directories(pangu_radiation_cooling PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(pangu_radiation_cooling PRIVATE Parthenon::parthenon)
  target_compile_features(pangu_radiation_cooling PRIVATE cxx_std_20)
  add_executable(pangu_radiation_target_thickness
    "${CMAKE_SOURCE_DIR}/tst/unit/radiation_target_thickness.cc")
  target_include_directories(pangu_radiation_target_thickness PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(pangu_radiation_target_thickness PRIVATE Parthenon::parthenon)
  target_compile_features(pangu_radiation_target_thickness PRIVATE cxx_std_20)
  if(PHYSICS STREQUAL "gr" AND METRIC STREQUAL "z4c" AND MODE STREQUAL "sync")
    add_executable(pangu_nr0_sync_geometry
      "${CMAKE_SOURCE_DIR}/tst/unit/nr0_sync_geometry.cc")
    target_include_directories(pangu_nr0_sync_geometry PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(pangu_nr0_sync_geometry PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr0_sync_geometry PRIVATE cxx_std_20)
    add_executable(pangu_nr1_finite_difference
      "${CMAKE_SOURCE_DIR}/tst/unit/nr1_finite_difference.cc")
    target_include_directories(pangu_nr1_finite_difference PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr1_finite_difference PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr1_finite_difference PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr1_finite_difference PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_nr1_host_oracle
      "${CMAKE_SOURCE_DIR}/tst/unit/nr1_host_oracle.cc")
    target_include_directories(pangu_nr1_host_oracle PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr1_host_oracle PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr1_host_oracle PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr1_host_oracle PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_nr1_decomposition
      "${CMAKE_SOURCE_DIR}/tst/unit/nr1_decomposition.cc")
    target_include_directories(pangu_nr1_decomposition PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr1_decomposition PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr1_decomposition PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr1_decomposition PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_nr1_finite_difference_benchmark
      "${CMAKE_SOURCE_DIR}/tst/benchmark/nr1_finite_difference_benchmark.cc")
    target_include_directories(pangu_nr1_finite_difference_benchmark PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr1_finite_difference_benchmark PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr1_finite_difference_benchmark PRIVATE cxx_std_20)
    add_executable(pangu_nr1_finite_difference_convergence
      "${CMAKE_SOURCE_DIR}/tst/unit/nr1_finite_difference_convergence.cc")
    target_include_directories(pangu_nr1_finite_difference_convergence PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr1_finite_difference_convergence PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr1_finite_difference_convergence PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr1_finite_difference_convergence PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_nr2_local_rhs
      "${CMAKE_SOURCE_DIR}/tst/unit/nr2_local_rhs.cc")
    target_include_directories(pangu_nr2_local_rhs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr2_local_rhs PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr2_local_rhs PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr2_local_rhs PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_sync0_stress_energy
      "${CMAKE_SOURCE_DIR}/tst/unit/sync0_stress_energy.cc")
    target_include_directories(pangu_sync0_stress_energy PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_sync0_stress_energy PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_sync0_stress_energy PRIVATE cxx_std_20)
    target_compile_definitions(pangu_sync0_stress_energy PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_sync1_grhd_state
      "${CMAKE_SOURCE_DIR}/tst/unit/sync1_grhd_state.cc")
    target_include_directories(pangu_sync1_grhd_state PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_sync1_grhd_state PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_sync1_grhd_state PRIVATE cxx_std_20)
    target_compile_definitions(pangu_sync1_grhd_state PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_sync1_tov_profile
      "${CMAKE_SOURCE_DIR}/tst/unit/sync1_tov_profile.cc"
      z4c/initial_data/tov.cc)
    target_include_directories(pangu_sync1_tov_profile PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_sync1_tov_profile PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_sync1_tov_profile PRIVATE cxx_std_20)
    target_compile_definitions(pangu_sync1_tov_profile PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_sync2_grmhd_state
      "${CMAKE_SOURCE_DIR}/tst/unit/sync2_grmhd_state.cc")
    target_include_directories(pangu_sync2_grmhd_state PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_sync2_grmhd_state PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_sync2_grmhd_state PRIVATE cxx_std_20)
    target_compile_definitions(pangu_sync2_grmhd_state PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_nr3_adm_conversion
      "${CMAKE_SOURCE_DIR}/tst/unit/nr3_adm_conversion.cc")
    target_include_directories(pangu_nr3_adm_conversion PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr3_adm_conversion PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr3_adm_conversion PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr3_adm_conversion PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_nr3_constraints
      "${CMAKE_SOURCE_DIR}/tst/unit/nr3_constraints.cc")
    target_include_directories(pangu_nr3_constraints PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr3_constraints PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr3_constraints PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr3_constraints PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_nr3_schwarzschild
      "${CMAKE_SOURCE_DIR}/tst/unit/nr3_schwarzschild.cc")
    target_include_directories(pangu_nr3_schwarzschild PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr3_schwarzschild PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr3_schwarzschild PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr3_schwarzschild PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_nr3_kerr_schild
      "${CMAKE_SOURCE_DIR}/tst/unit/nr3_kerr_schild.cc")
    target_include_directories(pangu_nr3_kerr_schild PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr3_kerr_schild PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr3_kerr_schild PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr3_kerr_schild PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_nr5_boosted_puncture
      "${CMAKE_SOURCE_DIR}/tst/unit/nr5_boosted_puncture.cc")
    target_include_directories(pangu_nr5_boosted_puncture PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr5_boosted_puncture PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr5_boosted_puncture PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr5_boosted_puncture PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_nr6_weyl
      "${CMAKE_SOURCE_DIR}/tst/unit/nr6_weyl.cc")
    target_include_directories(pangu_nr6_weyl PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr6_weyl PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr6_weyl PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr6_weyl PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_nr7_bowen_york
      "${CMAKE_SOURCE_DIR}/tst/unit/nr7_bowen_york.cc")
    target_include_directories(pangu_nr7_bowen_york PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr7_bowen_york PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr7_bowen_york PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr7_bowen_york PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    add_executable(pangu_nr7_spectral_grid
      "${CMAKE_SOURCE_DIR}/tst/unit/nr7_spectral_grid.cc"
      z4c/puncture/spectral_solver.cc)
    target_include_directories(pangu_nr7_spectral_grid PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr7_spectral_grid PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr7_spectral_grid PRIVATE cxx_std_20)
    add_executable(pangu_nr6_amr
      "${CMAKE_SOURCE_DIR}/tst/unit/nr6_amr.cc")
    target_include_directories(pangu_nr6_amr PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(pangu_nr6_amr PRIVATE Parthenon::parthenon)
    target_compile_features(pangu_nr6_amr PRIVATE cxx_std_20)
    target_compile_definitions(pangu_nr6_amr PRIVATE
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    if(PANGU_ATHENAK_REFERENCE_DIR AND
       EXISTS "${PANGU_ATHENAK_REFERENCE_DIR}/src/utils/finite_diff.hpp")
      configure_file(
        "${CMAKE_SOURCE_DIR}/tst/reference/athenak_finite_difference_reference.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/athenak_finite_difference_reference.h"
        @ONLY)
      add_executable(pangu_nr1_athenak_finite_difference
        "${CMAKE_SOURCE_DIR}/tst/reference/nr1_athenak_finite_difference.cc")
      target_include_directories(pangu_nr1_athenak_finite_difference PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}")
      target_link_libraries(pangu_nr1_athenak_finite_difference PRIVATE Parthenon::parthenon)
      target_compile_features(pangu_nr1_athenak_finite_difference PRIVATE cxx_std_20)
      target_compile_definitions(pangu_nr1_athenak_finite_difference PRIVATE
        KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    endif()
  endif()
endif()

find_program(PANGU_CLANG_FORMAT clang-format)
if(PANGU_CLANG_FORMAT)
  set(PANGU_FORMAT_SOURCES ${PANGU_SOURCES})
  # CMake @ONLY placeholders are not valid C++ tokens and clang-format inserts
  # spaces inside them, corrupting the generated geometry assembly header.
  list(REMOVE_ITEM PANGU_FORMAT_SOURCES geometry/geometry_assembly.h.in)
  if(BUILD_TESTING AND PANGU_ENABLE_TESTING)
    list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/hydro_kernels.cc")
    list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/mhd_kernels.cc")
    list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/phase5_relativity.cc")
    list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/mks_geometry.cc")
    list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/reconstruct_registry.cc")
    list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/electron_transport.cc")
    list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/electron_heating.cc")
    list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/electron_microphysics.cc")
    list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/radiation_cooling.cc")
    list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/radiation_target_thickness.cc")
    if(PHYSICS STREQUAL "gr" AND METRIC STREQUAL "z4c" AND MODE STREQUAL "sync")
      list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/nr0_sync_geometry.cc")
      list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/nr1_finite_difference.cc")
      list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/nr1_host_oracle.cc")
      list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/nr1_decomposition.cc")
      list(APPEND PANGU_FORMAT_SOURCES
        "../tst/benchmark/nr1_finite_difference_benchmark.cc")
      list(APPEND PANGU_FORMAT_SOURCES
        "../tst/unit/nr1_finite_difference_convergence.cc")
      list(APPEND PANGU_FORMAT_SOURCES
        "../tst/reference/nr1_athenak_finite_difference.cc")
      list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/nr2_local_rhs.cc")
      list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/nr3_adm_conversion.cc")
      list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/nr3_constraints.cc")
      list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/nr6_weyl.cc")
      list(APPEND PANGU_FORMAT_SOURCES "../tst/unit/nr6_amr.cc")
    endif()
  endif()
  list(TRANSFORM PANGU_FORMAT_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
  add_custom_target(format
    COMMAND "${PANGU_CLANG_FORMAT}" -i ${PANGU_FORMAT_SOURCES}
    COMMENT "Formatting PANGU-owned source files")
  add_custom_target(format-check
    COMMAND "${PANGU_CLANG_FORMAT}" --dry-run --Werror ${PANGU_FORMAT_SOURCES}
    COMMENT "Checking PANGU-owned source formatting")
endif()
