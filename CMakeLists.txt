cmake_minimum_required(VERSION 3.10)
project(pangu LANGUAGES C CXX)

# We follow Parthenon in requiring C++17 going forward
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std=c++17")
set(PARTHENON_ENABLE_CPP17 ON CACHE BOOL "Override")
# Set the path to include cmake/ dir
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Parthenon options
set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "Override")
set(PARTHENON_LINT_DEFAULT OFF CACHE BOOL "Override")
# Attempt HDF5 compression, requires recent/standard HDF5. YMMV
set(PARTHENON_DISABLE_HDF5_COMPRESSION OFF CACHE BOOL "Override")
# Don't build sparse (selectively-allocated) variable support
set(PARTHENON_DISABLE_SPARSE ON CACHE BOOL "Override")
# Set to move MPI buffers to host; slower but less crashy
# Favor setting this per-machine in machines/
set(PARTHENON_ENABLE_HOST_COMM_BUFFERS OFF CACHE BOOL "Override")

# Parthenon internal build options
set(BUILD_TESTING OFF CACHE BOOL "Override")
set(ENABLE_COMPILER_WARNINGS OFF CACHE BOOL "Override")
# Always use static HDF5
set(HDF5_USE_STATIC_LIBRARIES ON CACHE BOOL "Override")

# Kokkos options
set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Override")
set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "Override")
set(Kokkos_ENABLE_HWLOC OFF CACHE BOOL "Override") # Possible speed improvement?
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL "Override")

add_subdirectory(parthenon)
include_directories(parthenon/src)

add_subdirectory(pangu/src)