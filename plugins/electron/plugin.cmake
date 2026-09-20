pangu_register_plugin(
  NAME electron
  VERSION 1.0.0
  REQUIRES_PANGU_API 1
  HEADER plugin.h
  TYPE pangu::plugins::electron::Plugin
  SOURCES electron/electron_package.cc plugin.cc
  REQUIRES_CAPABILITY fluid.mhd
  PROVIDES_CAPABILITY electron.thermodynamics
  DESCRIPTION "Electron thermodynamics and heating models")
