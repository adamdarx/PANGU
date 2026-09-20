pangu_register_plugin(
  NAME ohmic
  VERSION 1.0.0
  REQUIRES_PANGU_API 1
  HEADER plugin.h
  TYPE pangu::plugins::ohmic::Plugin
  SOURCES plugin.cc
  REQUIRES_PLUGIN "diffusion>=1.0,<2.0"
  REQUIRES_CAPABILITY fluid.mhd physics.diffusion
  PROVIDES_CAPABILITY diffusion.operator.ohmic
  DESCRIPTION "Ohmic resistivity operator")
