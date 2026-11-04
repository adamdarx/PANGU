# These initializers consume the native Z4c and Valencia implementation.
if(PHYSICS STREQUAL "gr" AND METRIC STREQUAL "z4c" AND MODE STREQUAL "sync")
  pangu_register_plugin(
    NAME compact_objects
    VERSION 0.1.0
    REQUIRES_PANGU_API 1
    HEADER plugin.h
    TYPE pangu::plugins::compact_objects::Plugin
    SOURCES plugin.cc bns_initial_data.cc
    REQUIRES_CAPABILITY fluid.mhd geometry
    PROVIDES_CAPABILITY initial_data.grmhd
    DESCRIPTION "Compact-object matter initial data using the native Z4c stage graph")
endif()
