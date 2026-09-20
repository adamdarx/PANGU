pangu_register_plugin(
  NAME radiation_cooling
  VERSION 1.0.0
  REQUIRES_PANGU_API 1
  HEADER plugin.h
  TYPE pangu::plugins::radiation_cooling::Plugin
  SOURCES radiation/cooling_source.cc radiation/radiation_package.cc plugin.cc
  REQUIRES_CAPABILITY fluid.mhd
  PROVIDES_CAPABILITY radiation.backend
  CONFLICTS_PLUGIN radiation_transport electron
  DESCRIPTION "Target-thickness radiation cooling")
