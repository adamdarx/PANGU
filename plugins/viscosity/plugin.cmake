pangu_register_plugin(
  NAME viscosity
  VERSION 1.0.0
  REQUIRES_PANGU_API 1
  HEADER plugin.h
  TYPE pangu::plugins::viscosity::Plugin
  SOURCES plugin.cc
  REQUIRES_PLUGIN "diffusion>=1.0,<2.0"
  REQUIRES_CAPABILITY fluid physics.diffusion
  PROVIDES_CAPABILITY diffusion.operator.viscosity
  DESCRIPTION "Isotropic fluid viscosity operator")
