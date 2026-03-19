#
# pangu executable
#
if (Kokkos_ENABLE_CUDA)
    set(EXE_NAME "pangu.cuda")
else()
    set(EXE_NAME "pangu.host")
endif()

# Collect all source files under pangu/src recursively.
file(GLOB_RECURSE EXE_NAME_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# Select the problem implementation from pangu/problem/<PROBLEM>/ProblemGenerator.cpp.
set(PROBLEM "" CACHE STRING "Problem directory under pangu/problem used for ProblemGenerator.cpp")
if("${PROBLEM}" STREQUAL "")
    message(FATAL_ERROR "Please configure with -DPROBLEM=<problem_name>, e.g. -DPROBLEM=BrioWuShocktube")
endif()

set(PROBLEM_GENERATOR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../problem/${PROBLEM}/ProblemGenerator.cpp")
if(NOT EXISTS "${PROBLEM_GENERATOR_SRC}")
    message(FATAL_ERROR "Cannot find ${PROBLEM_GENERATOR_SRC}")
endif()
list(APPEND EXE_NAME_SRC "${PROBLEM_GENERATOR_SRC}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../problem/${PROBLEM})

add_executable(${EXE_NAME} ${EXE_NAME_SRC})

target_link_libraries(${EXE_NAME} PUBLIC kokkos)
target_link_libraries(${EXE_NAME} PUBLIC parthenon)

# FLAGS
if(CMAKE_BUILD_TYPE)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        message("Debug build")
        target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=1)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    else()
        message("Explicit non-Debug build")
        target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=0)
    endif()
else()
    message("Implicit non-Debug build")
    target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=0)
endif()
