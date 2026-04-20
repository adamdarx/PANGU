#
# pangu executable
#
if(Kokkos_ENABLE_CUDA)
    set(EXE_NAME "pangu.cuda")
else()
    set(EXE_NAME "pangu.host")
endif()

option(PANGU_GR "Build pangu with GRMHD simulator" OFF)

# Collect all source files under pangu/src recursively.
file(GLOB_RECURSE EXE_NAME_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

set(SIMULATOR_SRMHD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/task_list/ideal_srmhd.cc")
set(SIMULATOR_GRMHD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/task_list/ideal_grmhd.cc")

# Keep exactly one simulator implementation in the final target.
list(REMOVE_ITEM EXE_NAME_SRC
    "${SIMULATOR_SRMHD_SRC}"
    "${SIMULATOR_GRMHD_SRC}"
)

if(PANGU_GR)
    if(NOT EXISTS "${SIMULATOR_GRMHD_SRC}")
        message(FATAL_ERROR "Cannot find ${SIMULATOR_GRMHD_SRC}")
    endif()
    list(APPEND EXE_NAME_SRC "${SIMULATOR_GRMHD_SRC}")
    message(STATUS "Pangu simulator mode: GRMHD")
else()
    if(NOT EXISTS "${SIMULATOR_SRMHD_SRC}")
        message(FATAL_ERROR "Cannot find ${SIMULATOR_SRMHD_SRC}")
    endif()
    list(APPEND EXE_NAME_SRC "${SIMULATOR_SRMHD_SRC}")
    message(STATUS "Pangu simulator mode: SRMHD")
endif()

# Select problem implementation from pangu/problem/<PROBLEM>/problem_generator.cpp.
set(PROBLEM "" CACHE STRING
    "Problem directory under pangu/problem used for problem_generator.cpp")
if("${PROBLEM}" STREQUAL "")
    message(FATAL_ERROR
        "Please configure with -DPROBLEM=<problem_name>, e.g. -DPROBLEM=brio_wu_shocktube")
endif()

set(PROBLEM_GENERATOR_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/../problem/${PROBLEM}/problem_generator.cpp")
if(NOT EXISTS "${PROBLEM_GENERATOR_SRC}")
    message(FATAL_ERROR "Cannot find ${PROBLEM_GENERATOR_SRC}")
endif()
list(APPEND EXE_NAME_SRC "${PROBLEM_GENERATOR_SRC}")

add_executable(${EXE_NAME} ${EXE_NAME_SRC})

target_include_directories(${EXE_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../problem/${PROBLEM}
)

target_link_libraries(${EXE_NAME}
    PUBLIC
        kokkos
        parthenon
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build")
    target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=1)
    target_compile_options(${EXE_NAME} PRIVATE -g)
else()
    message(STATUS "Non-Debug build")
    target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=0)
endif()
