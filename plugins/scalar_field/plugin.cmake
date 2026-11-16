set(scalar_field_sources plugin.cc ems_coupled.cc ems_coordinates.cc ems_spherical.cc)
if(PHYSICS STREQUAL "gr" AND METRIC STREQUAL "z4c" AND MODE STREQUAL "sync")
  list(APPEND scalar_field_sources ems_cartesian.cc)
endif()
pangu_register_plugin(
  NAME scalar_field
  VERSION 1.1.0
  REQUIRES_PANGU_API 1
  HEADER plugin.h
  TYPE pangu::plugins::scalar_field::Plugin
  SOURCES ${scalar_field_sources}
  PROVIDES_CAPABILITY matter.scalar.test_field matter.scalar.ems
  CONFLICTS_PLUGIN scalar_collapse compact_objects
  DESCRIPTION "Scalar fields and coupled EMS with selectable coordinates")
