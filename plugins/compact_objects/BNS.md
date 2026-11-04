The `bns_time_symmetric` model supplies a constraint-solved, equal-star head-on
collision from rest, in units G=c=M_sun=1. It is a development feature of the
`compact_objects` plugin; the plugin remains version 0.1.0 pending evolution
qualification. It adds no evolution equations or changes to Parthenon.

The benchmark motivation is [Miller, Suen & Tobias,
gr-qc/9904041](https://arxiv.org/abs/gr-qc/9904041). That paper boosts the stars
and then solves York's coupled constraints with a nonflat conformal metric.
This implementation instead starts from rest on a conformally flat slice.
Its collision time, stellar masses and collapse threshold must not be called
a reproduction of the paper's numerical values. The conformal initial-value
construction follows the framework reviewed by [Cook,
Living Reviews 3, 5](https://doi.org/10.12942/lrr-2000-5).

With gamma_ij=psi^4 delta_ij, K_ij=0 and Eulerian fluid velocity zero, the
momentum constraint is identically satisfied. A positive conformal energy
profile Ebar is freely prescribed as the sum of two disjoint isotropic TOV
profiles psi_TOV^8 E_TOV. The remaining constraint is

```
Delta_flat psi = -2 pi Ebar psi^(-3),       psi -> 1 at infinity.
E = Ebar psi^(-8),    P = K rho^2,         E = rho + K rho^2.
rho = 2 E / (1 + sqrt(1 + 4 K E)).
```

The axisymmetric Legendre Green function solves the elliptic equation on all
of space. Radial quadrature, angular truncation and table interpolation have
separate numerical errors; the fixed-point iteration residual is **not** the
Cartesian Hamiltonian-constraint error. The test script evaluates the latter
with an independent Cartesian finite-difference stencil and checks the
isolated-star limit. Even angular multipoles impose the exact equal-star
reflection symmetry, but the evolution itself is fully three-dimensional.
Outside the matter-support sphere the reader uses the exact multipole vacuum
tail, rather than truncating or clamping the metric at the table edge.

The default TOV seed has K=100, Gamma=2, rho_c=0.00128 and isolated ADM mass
about 1.40016. A separation of 24 gives a solved total ADM mass about 2.65489
and baryon mass about 2.92683. These are **post-solve** masses: conformal energy
rescaling changes the physical density and mass of each seed. The stars are
not claimed to be in binary hydrostatic equilibrium. The initial lapse is
psi^(-2), the shift zero, and the conformal metric exactly the identity.

Generate a table, then supply its absolute path in the input:

```bash
OPENBLAS_NUM_THREADS=1 python3 plugins/compact_objects/bns_data.py \
  --output /absolute/path/bns.dat --step 0.0125 --lmax 128 --angular-points 384
python3 tst/regression/bns_initial_data.py --workdir /new/path/elliptic-check
```

Use `examples/bns_head_on_admission.in` with `model=bns_time_symmetric` and
`initial_data_file=/absolute/path/bns.dat`. The required native evolution is
`METRIC=z4c MODE=sync`, `matter_source=grmhd`, GR ideal Gamma=2, B=0 and zero
passive scalars. Gamma-law internal energy is evolved conservatively, so
shock heating is allowed; P=K rho^2 is used **only at initialization**. The
Gamma=2 cold-plus-thermal closure with Gamma_th=2 is equivalent to this ideal
Gamma-law evolution. Atmosphere density/pressure floors are absent from the
elliptic source; their tiny Hamiltonian source and mass budget must be bounded
and reported by admission, not silently included in the solved data.

Local acceptance must include the imported density, metric, stress energy,
Hamiltonian/momentum constraints, finite C2P fields, baryon conservation and
restart equality. Production admission additionally records actual AMR cell
sizes, resource use, common-horizon residuals, thermal energy and exterior
matter budget. Neither a successful table solve nor a successful process exit
certifies formation of a physical black hole.
