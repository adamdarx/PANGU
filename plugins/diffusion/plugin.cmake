pangu_register_plugin(
  NAME diffusion
  VERSION 1.0.0
  REQUIRES_PANGU_API 1
  HEADER plugin.h
  TYPE pangu::plugins::diffusion::Plugin
  SOURCES plugin.cc
  PROVIDES_CAPABILITY physics.diffusion
  DESCRIPTION "Diffusion operator assembly and integration foundation")
