pangu_register_plugin(
  NAME radiation_transport
  VERSION 1.0.0
  REQUIRES_PANGU_API 1
  HEADER plugin.h
  TYPE pangu::plugins::radiation_transport::Plugin
  SOURCES radiation/geodesic_grid.cc radiation/transport_package.cc
          radiation/task_contribution.cc plugin.cc
  REQUIRES_CAPABILITY fluid.hydro
  PROVIDES_CAPABILITY radiation.backend
  CONFLICTS_PLUGIN radiation_cooling
  DESCRIPTION "Radiation moment transport")
