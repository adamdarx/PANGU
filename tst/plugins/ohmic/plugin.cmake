pangu_register_plugin(
  NAME external_probe
  VERSION 1.1.0
  REQUIRES_PANGU_API 1
  HEADER plugin.h
  TYPE pangu_test_plugins::ohmic::Plugin
  SOURCES plugin.cc
  REQUIRES_PLUGIN "diffusion>=1.0,<2.0"
  REQUIRES_CAPABILITY physics.diffusion fluid.mhd
  PROVIDES_CAPABILITY test.external_probe
  DESCRIPTION "External dependency fixture")
