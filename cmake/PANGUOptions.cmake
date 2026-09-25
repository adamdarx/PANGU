include_guard(GLOBAL)

option(PANGU_ENABLE_MPI "Build PANGU with MPI" ON)
option(PANGU_ENABLE_HDF5 "Build PANGU with HDF5 output/restart" ON)
option(PANGU_ENABLE_OPENMP "Build PANGU with the Kokkos OpenMP backend" OFF)
option(PANGU_ENABLE_CUDA "Build PANGU with the Kokkos CUDA backend" OFF)
option(PANGU_SINGLE_PRECISION "Use single precision Parthenon Real" OFF)
option(PANGU_ENABLE_TESTING "Register PANGU tests" ON)
option(PANGU_WARNINGS_AS_ERRORS "Treat PANGU warnings as errors" OFF)
set(PANGU_ATHENAK_REFERENCE_DIR "" CACHE PATH
    "Optional AthenaK source tree used only by cross-code validation targets")
set(PANGU_ATHENAK_EXECUTABLE "" CACHE FILEPATH
    "Optional built-in-pgen AthenaK executable used by cross-code evolution tests")
set(PANGU_ATHENAK_GAUGE_EXECUTABLE "" CACHE FILEPATH
    "Optional gauge-wave-pgen AthenaK executable used by cross-code evolution tests")
set(PANGU_ATHENAK_ROBUST_EXECUTABLE "" CACHE FILEPATH
    "Optional robust-stability-pgen AthenaK executable used by statistical evolution tests")
set(PANGU_ATHENAK_TOV_EXECUTABLE "" CACHE FILEPATH
    "Optional dynamic-GRMHD TOV AthenaK executable used by SYNC-1 cross-code tests")

set(PHYSICS "gr" CACHE STRING "Compile-time relativistic physics")
set_property(CACHE PHYSICS PROPERTY STRINGS sr gr)
set(METRIC "" CACHE STRING "Compile-time spacetime metric (GR only)")
set_property(CACHE METRIC PROPERTY STRINGS cks mks z4c)
set(MODE "" CACHE STRING "Compile-time geometry access mode (GR only)")
set_property(CACHE MODE PROPERTY STRINGS dynamic static sync)
set(ESTIMATOR "light" CACHE STRING
    "Compile-time timestep estimator registered in src/estimator/registry.h")

string(TOLOWER "${PHYSICS}" PHYSICS)
set(PHYSICS "${PHYSICS}" CACHE STRING "Compile-time relativistic physics" FORCE)
if(PHYSICS STREQUAL "sr")
  set(PANGU_PHYSICS_SR ON)
  set(PANGU_PHYSICS_GR OFF)
elseif(PHYSICS STREQUAL "gr")
  set(PANGU_PHYSICS_SR OFF)
  set(PANGU_PHYSICS_GR ON)
endif()

function(pangu_configure_parthenon)
  set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "" FORCE)
  set(PARTHENON_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(PARTHENON_ENABLE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
  set(PARTHENON_ENABLE_INTEGRATION_TESTS OFF CACHE BOOL "" FORCE)
  set(PARTHENON_ENABLE_PERFORMANCE_TESTS OFF CACHE BOOL "" FORCE)
  set(PARTHENON_ENABLE_REGRESSION_TESTS OFF CACHE BOOL "" FORCE)
  set(PARTHENON_ENABLE_PYTHON_MODULE_CHECK OFF CACHE BOOL "" FORCE)
  set(REGRESSION_GOLD_STANDARD_SYNC OFF CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_SPARSE ON CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_OPENPMD ON CACHE BOOL "" FORCE)

  if(PANGU_ENABLE_MPI)
    set(PARTHENON_DISABLE_MPI OFF CACHE BOOL "" FORCE)
  else()
    set(PARTHENON_DISABLE_MPI ON CACHE BOOL "" FORCE)
  endif()

  if(PANGU_ENABLE_HDF5)
    set(PARTHENON_DISABLE_HDF5 OFF CACHE BOOL "" FORCE)
  else()
    set(PARTHENON_DISABLE_HDF5 ON CACHE BOOL "" FORCE)
  endif()

  if(PANGU_SINGLE_PRECISION)
    set(PARTHENON_SINGLE_PRECISION ON CACHE BOOL "" FORCE)
  else()
    set(PARTHENON_SINGLE_PRECISION OFF CACHE BOOL "" FORCE)
  endif()

  if(PANGU_ENABLE_CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
  endif()

  if(PANGU_ENABLE_OPENMP)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
  endif()

  if(NOT PANGU_ENABLE_CUDA AND NOT PANGU_ENABLE_OPENMP)
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
  endif()
endfunction()
