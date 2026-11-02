# Compact-object matter initial data

Development version 0.1.0. This plugin supplies initial data to the native Z4c
and Valencia GRMHD evolution graph through `problem_id=sync_plugin_grmhd`.
It is compiled for `PHYSICS=gr METRIC=z4c MODE=sync`.

The initial `puncture_plasma` model combines the existing native puncture
elliptic solver with a uniform plasma and an orbitally aligned magnetic field,
following Kelly et al., arXiv:1710.02132. One or two punctures are controlled by
the existing `<problem>` parameters. Plasma parameters are in
`<plugin/compact_objects>`: `density`, `pressure`, and `btilde1/2/3`.

The literature benchmark neglects plasma self-gravity and requires
`numerical_relativity/matter_source=zero`. The metric still evolves dynamically.
The initializer rejects using its independently normalized density as a full
Einstein source. The other stellar-collapse benchmarks test active backreaction.
Zero backreaction must not remove the fluid's geometric source terms: its
Valencia update uses its own stress-energy, independently of `nr.tmunu`.
`tst/regression/sync_test_fluid_gravity.py` checks the initial Schwarzschild
acceleration against an analytic solution, spatial refinement and a half-time
control while requiring `nr.tmunu` to remain exactly zero.

`mhd.b_cell` and `mhd.b_face` store the densitized magnetic field. A constant
densitized field has zero discrete divergence; its physical Eulerian value is
`Btilde/sqrt(gamma)`. Initially the fluid has zero Eulerian velocity. Pressure
is converted to the internal-energy primitive using the selected EOS.

Admission inputs use `puncture_chi_threshold=0.02`. For an initially static
Schwarzschild puncture, the isotropic horizon has chi=1/16; this cutoff places
the matter reset inside that initial horizon (coordinate radius about 0.30 M
for the one-hole model, compared with the horizon radius 0.50 M). A cutoff of
0.1 would reset an exterior shell as well. The separate radius-0.1 constraint
mask is a diagnostic exclusion; it does not reset matter. A production run
must check the moving protection region against the measured horizons.

This initial-data plugin does not replace the numerical fluxes, CT algorithm,
boundary communication, metric evolution or recovery algorithms. Its low
resolution input is an admission test, not an accepted inspiral/merger result.
The refinement examples use four ghost cells: Parthenon requires an even
ghost count with refinement, in addition to the evolution stencil's minimum.

`flux_diagnostics.py` computes pointwise ADM invariants and the coordinate
currents `sqrt(-g) rho u^i` and `-sqrt(-g) T^i_t(EM)` from native projected
velocity and densitized B. Its sphere integral includes the Cartesian
`r^2 dOmega` Jacobian; it requires separately supplied, qualified surface
samples. It does not interpolate the AMR mesh, measure flux through a moving
apparent horizon, or substitute a coordinate-sphere inflow for a horizon rate.
The EM energy current in a dynamical metric is coordinate dependent. It also
differs from the `S^z_(1,0)` Poynting proxy in Kelly et al. Equations (4)-(6)
there use a different placement of the absorbed `4 pi` magnetic normalization.
Run `python3 tst/regression/compact_object_flux.py` for analytic SR Poynting
checks, full curved-ADM tensor contractions and sphere-Jacobian checks.

`surface_flux.py` samples these currents on fixed coordinate spheres directly
from native three-dimensional UniformCartesian PHDF snapshots without ghosts:

```bash
python3 plugins/compact_objects/surface_flux.py snapshot.phdf \
  --radii 10 20 30 --center 0 0 0 --ntheta 256 --nphi 512 --output flux.json
```

The sampler selects the finest leaf, uses tensor cubic interpolation of cell
values, and integrates with Gauss-Legendre/polar and periodic/azimuthal
quadrature. Interior stencils become one-sided at block edges, with at most
half a cell of extrapolation; the report includes these counts. Entire spheres
must lie inside the mesh. Invalid geometry and uncovered points are rejected.
Memory use is bounded by a block cache. `--degree 1` selects linear sampling.
This interpolation order does not upgrade a finite-volume evolution's order.
Check both spatial and angular convergence of each physical measurement.

`python3 tst/regression/compact_object_surface.py --workdir /tmp/surface-check`
tests native AMR ownership, curved-metric currents, divergence-theorem surface
integrals, analytic radial inflow/Poynting flux and spatial/angular convergence.
Its fixtures are synthetic diagnostics tests, not black-hole evolution data.
The coordinate-sphere measurement does not supply a moving-horizon accretion
rate or a gauge-invariant luminosity.
