if(PHYSICS STREQUAL "gr" AND METRIC STREQUAL "cks" AND MODE STREQUAL "dynamic")
  pangu_register_plugin(
    NAME scalar_collapse
    VERSION 1.0.0
    REQUIRES_PANGU_API 1
    HEADER plugin.h
    TYPE pangu::plugins::scalar_collapse::Plugin
    SOURCES plugin.cc
    CONFLICTS_PLUGIN scalar_field
    PROVIDES_CAPABILITY matter.scalar.spherical_collapse
    DESCRIPTION "Regular-origin Einstein-scalar collapse with causal dynamical excision")
endif()
