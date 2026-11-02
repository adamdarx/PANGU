pangu_register_plugin(
  NAME scalar_field
  VERSION 0.2.0
  REQUIRES_PANGU_API 1
  HEADER plugin.h
  TYPE pangu::plugins::scalar_field::Plugin
  SOURCES plugin.cc ems_coupled.cc
  PROVIDES_CAPABILITY matter.scalar.test_field matter.scalar.spherical_ems
  DESCRIPTION "Configurable scalar test fields and fully nonlinear spherical EMS")
