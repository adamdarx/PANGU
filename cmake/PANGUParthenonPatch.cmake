# Keep the official GitHub submodule and Gitee mirror independently usable.
# The mirror may already contain this optimization; never apply it twice.
function(pangu_apply_parthenon_patch)
  if(NOT PANGU_ENABLE_CUDA)
    return()
  endif()
  find_package(Git REQUIRED)
  set(patch "${PROJECT_SOURCE_DIR}/cmake/patches/parthenon-cuda-boundary.patch")
  set(dependency "${PROJECT_SOURCE_DIR}/parthenon")
  execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${patch}"
    WORKING_DIRECTORY "${dependency}" RESULT_VARIABLE already_applied
    OUTPUT_QUIET ERROR_QUIET)
  if(already_applied EQUAL 0)
    return()
  endif()
  execute_process(COMMAND "${GIT_EXECUTABLE}" apply --check "${patch}"
    WORKING_DIRECTORY "${dependency}" RESULT_VARIABLE can_apply
    OUTPUT_QUIET ERROR_VARIABLE error)
  if(NOT can_apply EQUAL 0)
    message(FATAL_ERROR "Cannot apply PANGU CUDA boundary patch without conflicting with Parthenon changes: ${error}")
  endif()
  execute_process(COMMAND "${GIT_EXECUTABLE}" apply "${patch}"
    WORKING_DIRECTORY "${dependency}" RESULT_VARIABLE applied ERROR_VARIABLE error)
  if(NOT applied EQUAL 0)
    message(FATAL_ERROR "Parthenon CUDA boundary patch failed: ${error}")
  endif()
  message(STATUS "Applied PANGU CUDA boundary optimization to Parthenon")
endfunction()
