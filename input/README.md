# PANGU formal input decks

`input/` contains only user-facing, canonical physics configurations. Each physical problem has
one `<problem-name>.in` file. The sole exception is `mks_torus_sane`, whose six production decks
encode the approved two-dimensional and three-dimensional resolution/time matrix.

CTest-only, compatibility, benchmark, and module-assembly fixtures live under `tst/input/` and are
not production inputs. Resolution, backend, output cadence, or short-regression variants must not
be added to `input/` as separate files; pass such changes as command-line overrides or add a test
fixture under `tst/input/`.

Metric-dependent GR inputs must be used with the matching compile-time `METRIC` and `MODE` build.
All `mks_torus_sane_*` decks require `METRIC=mks`, `MODE=static`, disable refinement, use one global
MeshBlock, and disable the electron module.
