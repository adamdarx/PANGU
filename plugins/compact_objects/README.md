# Compact-object matter initial data

Development version 0.1.0. This plugin supplies initial data to the native Z4c
and Valencia GRMHD evolution graph through `problem_id=plugin_grmhd`.
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
