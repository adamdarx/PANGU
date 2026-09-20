include_guard(GLOBAL)

function(_pangu_git_value directory arguments output fallback)
  execute_process(
    COMMAND git -C "${directory}" ${arguments}
    OUTPUT_VARIABLE value
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE result)
  if(NOT result EQUAL 0 OR value STREQUAL "")
    set(value "${fallback}")
  endif()
  set(${output} "${value}" PARENT_SCOPE)
endfunction()

function(pangu_collect_provenance)
  _pangu_git_value("${CMAKE_SOURCE_DIR}" "rev-parse;--short=12;HEAD"
                   PANGU_GIT_COMMIT "unknown")
  # An empty porcelain result is the successful, clean-worktree case.  Do not
  # pass it through _pangu_git_value(), which intentionally replaces empty
  # command output with its fallback value.
  execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" status --porcelain
    OUTPUT_VARIABLE PANGU_GIT_STATUS
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PANGU_GIT_STATUS_RESULT)
  if(PANGU_GIT_STATUS_RESULT EQUAL 0 AND PANGU_GIT_STATUS STREQUAL "")
    set(PANGU_GIT_DIRTY "clean")
  else()
    set(PANGU_GIT_DIRTY "dirty")
  endif()
  _pangu_git_value("${CMAKE_SOURCE_DIR}/parthenon" "rev-parse;--short=12;HEAD"
                   PANGU_PARTHENON_COMMIT "unknown")
  string(TIMESTAMP PANGU_CONFIGURE_TIME "%Y-%m-%dT%H:%M:%SZ" UTC)
  set(PANGU_GIT_COMMIT "${PANGU_GIT_COMMIT}" PARENT_SCOPE)
  set(PANGU_GIT_DIRTY "${PANGU_GIT_DIRTY}" PARENT_SCOPE)
  set(PANGU_PARTHENON_COMMIT "${PANGU_PARTHENON_COMMIT}" PARENT_SCOPE)
  set(PANGU_CONFIGURE_TIME "${PANGU_CONFIGURE_TIME}" PARENT_SCOPE)
endfunction()
