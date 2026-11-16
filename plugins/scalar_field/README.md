# Scalar-field plugin: selectable coordinates and coupled EMS

Plugin version: **1.1.0**. This release covers the documented test-field and
selectable-coordinate EMS interfaces, including spherical profile import and
controlled perturbations in PG coordinates. Collapse from an initially regular, untrapped center
is provided by the separate `scalar_collapse` plugin and is not part of this
package.

This package evolves actual Klein–Gordon test fields with Parthenon MeshData,
CUDA kernels, fourth-order finite differences and classical four-stage RK4.
The `cartesian_flat` and `radial_tortoise` backends retain the fixed-background
test-field problems. The `ems` backend selects `coordinates=pg` or `coordinates=cartesian` while
sharing the scalar potential and Maxwell coupling policies. PG uses the existing
spherical constraint solver. Cartesian coordinates use general ADM lapse, shift,
and spatial metric, evolving the scalar and Maxwell fields with the native Z4c
stage graph and supplying its Einstein sources. PG and the test-field modes
use uniform meshes. Cartesian EMS supports uniform, statically refined and
adaptive three-dimensional meshes with multiple MPI ranks. PG uses one radial
block and one MPI rank. GB gravitational
backreaction is not implemented.

## Configuration and extension

All modes use `problem_id=scalar_field` and `<plugins> enabled=scalar_field`.
A full `METRIC=z4c MODE=sync` executable selects PG or Cartesian EMS at runtime.
Only Cartesian EMS installs a numerical-relativity package and requires
`numerical_relativity/matter_source=external`. PG owns its dynamical geometry
and remains available in fixed-background builds and the minimal `pangu_scalar`
reproduction driver. Test-field modes retain their independent fixed geometry.

The `<plugin/scalar_field>` namespace contains:

| Parameter | Meaning |
|---|---|
| `backend` | `ems`, `radial_tortoise`, or `cartesian_flat` |
| `coordinates` | `pg` or `cartesian` for coupled EMS; defaults to `pg` |
| `fields` | 1 real scalar, or 2 real components of an uncharged complex scalar |
| `potential` | `polynomial` or `axion` |
| `mass`, `quartic`, `sextic` | Polynomial potential coefficients below |
| `decay_constant` | Axion potential scale, strictly positive |
| `bh_mass`, `bh_charge` | Fixed background M>0 and |Q|<M |
| `gb_eta` | Curvature coupling coefficient; zero disables it |
| `gb_coupling`, `gb_alpha` | `linear`, `quadratic`, `exponential`, with exponential coefficient |
| `em_coupling`, `em_alpha`, `em_linearized` | EMS `exponential` or `quadratic` coupling and optional linearization |
| `ell` | Angular harmonic number for a linear radial field; nonlinear cases require ell=0 |
| `initial_data` | `zero`, `gaussian`, `linear_hair`, `plane_wave`, `homogeneous` |
| `amplitude`, `center`, `width` | Initial profile controls |
| `wave_number`, `cfl` | Plane-wave k and explicit timestep factor (0<CFL<=0.5) |

For real components phi_a, define rho2=sum(phi_a^2). The polynomial convention is

    V = mass^2 rho2/2 + quartic rho2^2/2 + sextic rho2^3/6.

Thus a real field has V'=mass^2 phi+2 quartic phi^3+sextic phi^5. In particular,
`quartic` equals lambda in Eq. (3) of arXiv:2203.10341, rather than the alternative
lambda/4 convention. `potential=axion` uses
V=mass^2 decay_constant^2 [1-cos(phi/decay_constant)] for one real field.
Two real components share a U(1)-invariant polynomial potential, with the canonical convention Phi=(phi_1+i phi_2)/sqrt(2), so rho2=2|Phi|^2. Coefficients of a potential written directly in |Phi| must be converted accordingly. Nonlinear real
plane-wave momentum is not treated as an analytic solution and is rejected.

The public device API lives in `src/pangu/plugin_api/scalar_field.h`.
`Force<Potential,CurvatureCoupling,ElectromagneticCoupling>` separates the PDE from V and the GB coupling.
A new potential implements `Gradient(phi_a,rho2)` and can be instantiated in the
plugin without changing the stage graph. Coefficients and existing model choices
are runtime inputs; entirely new functions are compiled CUDA policies.
For gravitational backreaction it must also supply a consistent `Value`:
the potential energy sources Einstein's equations, and its derivative sources
the scalar equation. Supplying only a force is sufficient for a test field,
but is insufficient for a coupled theory.

GB `quadratic` means g=phi^2/2. Maxwell `quadratic` means f=1+alpha phi^2 and requires alpha>=0 so f stays positive. These normalizations are deliberately explicit.

## Equations and normalizations

One consistent unified real-field convention is

    S=(16 pi)^-1 integral sqrt(-g) [R-2(d phi)^2-4V-f(phi)F^2+4 eta g(phi)G].

For V=eta=0 this is the EMS action in Guo, Wang and Zhang. Removing the
Maxwell term and taking the test-field limit gives the scalar equation used
for the Zhang et al. hair test. The prototype evolves that scalar equation;
it does not evolve the gravitational field equations of this action.

Use the explicit convention

    Box(phi) = V'(phi) - eta g'(phi) G + f'(phi) F_ab F^ab/4.

The first two terms follow from L=-1/2(d phi)^2-V+eta g(phi) G. The scalar
EMS equation agrees with varying [R-2(d phi)^2-f(phi)F^2]/(16 pi) when V=0.
For this action the native Z4c convention `G_ab=8 pi T_ab` requires dividing
the canonical scalar-plus-Maxwell projections by `4 pi`. The Cartesian source
module applies that normalization to energy, momentum and spatial stress.

In spherical symmetry, Maxwell's Gauss constraint fixes the conserved electric
charge: E=Q/[r^2 f(phi)]. For f=exp(alpha phi^2), the EMS scalar force is

    -alpha Q^2 phi exp(-alpha phi^2)/r^4.

`em_linearized=true` replaces the exponential by one, yielding
m_eff^2=-alpha Q^2/r^4, the linear RN scalarization problem in Eq. (2) of
arXiv:2403.02089. The nonlinear option solves the spherical Maxwell constraint
algebraically on the fixed metric; it still neglects Einstein backreaction.

With F=1-2M/r+Q^2/r^2, dx/dr=1/F, u=r phi and v=partial_t u,

    partial_t u = v,
    partial_t v = partial_x^2 u
                  - F [F'/r + ell(ell+1)/r^2] u - F r Force(u/r).

Here G=48M^2/r^6-96MQ^2/r^7+40Q^4/r^8 and the electrovac F_ab F^ab=-2Q^2/r^4.
The inner/outer radial boundaries use ingoing/outgoing characteristic closures.
The flat backend evolves phi directly with a Cartesian Laplacian.
The current validation suite exercises one-dimensional Cartesian and spherical
problems; the Cartesian multidimensional branches are not yet validated.
`scalar.state` stores `(u,v)` per field in the radial backend, `(phi,d_t phi)`
in flat spacetime. `scalar.phi` always stores the physical scalar.

For Q=mass=quartic=sextic=0 and g(phi)=phi, the regular static solution is

    phi(r) = 2 eta/(M r) [1+M/r+4M^2/(3r^2)].

Its horizon value is 11 eta/(6M^2), and its asymptotic scalar charge is 2 eta/M.
This provides an exact test of source sign and normalization.

## Laptop CUDA reproduction

Configure a dedicated directory with CUDA enabled, a supported local GPU
architecture, MPI matching the installed parallel HDF5, and no OpenMP. For example:

```bash
cmake -S . -B build-cuda-scalar \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$PWD/parthenon/external/Kokkos/bin/nvcc_wrapper" \
  -DPANGU_ENABLE_CUDA=ON -DPANGU_ENABLE_MPI=ON \
  -DPANGU_ENABLE_HDF5=ON -DPANGU_ENABLE_OPENMP=OFF \
  -DPANGU_ENABLE_TESTING=OFF -DBUILD_TESTING=OFF \
  -DPHYSICS=gr -DMETRIC=cks -DMODE=dynamic -DESTIMATOR=wave \
  -DKokkos_ARCH_AMPERE86=ON -DPARTHENON_ENABLE_HOST_COMM_BUFFERS=OFF
cmake --build build-cuda-scalar --target pangu_scalar -j2
mkdir -p scalar-local
cd scalar-local
../build-cuda-scalar/src/pangu_scalar -i ../plugins/scalar_field/examples/schwarzschild_hair.in
```

`pangu_scalar` is a minimal local reproduction driver that links the same
StateDescriptor, kernels and stage contribution as the full PANGU plugin.
It skips compiling unrelated fluid/NR packages. The normal `pangu` target also
registers the problem when the plugin is compiled. Neither executable changes
an existing black-hole production input.

## Reproduction scope

The first target is the massless sf_1_1 branch and a lambda=10 sf_1_2 comparison
in Zhang et al. The reduction to spherical symmetry and a fixed Schwarzschild
slicing differs from their 3D moving-puncture implementation; coordinate-time
transients should not be claimed identical. Compare late exterior phi(r), the
analytic scalar charge, and broad self-interaction trends. Their grid spacing
M/32 in puncture coordinates is not this backend's tortoise-grid spacing.

There is a source-sign inconsistency between the positive GB term in Eq. (1)
and printed Eq. (6) of arXiv:2203.10341. This implementation derives the sign
from the action and uses the regular positive analytic solution and the
convention in Benkel et al. Eq. (3). It does not silently mix those signs.

The test-field backends do not imply geometric backreaction. The coupled EMS
backend described below supplies it in spherical symmetry. Full sGB changes the
gravitational principal part and needs a separate well-posed evolution
formulation; switching on `gb_eta` in a matter stress tensor would not suffice.

## Coupled EMS in PG coordinates

Use `backend=ems coordinates=pg` with `examples/ems_coupled_rn.in` or
`examples/ems_coupled_quartic.in`. The coordinate is now **areal radius r**, not
tortoise radius. The action is the unified EMS action above, with `gb_eta=0`.
The PG geometry in `scalar.state`/`ems.geometry` is dynamical and owned by
this coordinate formulation. It does not use the host build's analytic metric
or install an unrelated Z4c evolution. The same public plugin selects the
Cartesian Z4c formulation below when that coordinate mode is requested.
The potential and its derivative enter both the Einstein and scalar equations
through the same model policy. `potential=polynomial` and `potential=axion`
are supported; the current coupled version requires nonnegative polynomial
coefficients, nonnegative EM alpha, and one neutral real scalar.

The PG metric and field definitions are

    ds^2 = -a^2 dt^2 + (dr + a z dt)^2 + r^2 dOmega^2,
    X=partial_r phi, P=a^-1 partial_t phi-z X, m=r z^2/2.

Here P has the positive normal-derivative convention. The full equations are

    phi_t = a(P+z X),
    P_t = partial_r[r^2 a(X+z P)]/r^2 - a[V' - Q^2 f'/(2 r^4 f^2)],
    m_r = r^2[(P^2+X^2)/2 + z P X + V] + Q^2/(2 r^2 f),
    a_r/a = -r P X/z,
    m_t = r^2 a(P+z X)(z P+X).

These reduce to Eqs. (23)–(25) of Guo, Wang and Zhang when V=0. The electric
field E=Q/(r^2 f) changes with phi. Its energy density Q^2/(2 r^4 f) sources
the metric, while its coupling derivative sources the scalar. This is the
complete Maxwell reduction for a neutral, spherically symmetric electric
configuration: there are no independent electromagnetic radiation modes in
this sector. The scalar's stress energy also changes m and a, which feed
back into the scalar equation. No field amplitude floor or saturation cap is
used.

The default `metric_evolution=constrained` evolves the excision mass with its
Einstein flux equation and integrates the nonlinear radial mass constraint at
each classical RK4 substage. A parallel Picard/prefix integration checks its
final change against `constraint_tolerance` (default 1e-11). The maximum
`constraint_iterations` is configurable (default 12; example inputs use 8).
Nonconvergence aborts the run. A fourth-order integral of a_r/a fixes a(R)=1
on every substage. In addition to the radial Einstein constraint, an independent
temporal Einstein residual compares the reconstructed mass increment to the
RK-weighted m_t flux. This checks consistency of the constrained formulation.
`metric_evolution=free` retains the direct mass-evolution variant for diagnostic
comparisons; preliminary long runs exhibited growing constraint violations in
that variant, so it is not the recommended mode.

The initial Gaussian phi has P=0. Its mass profile solves the nonlinear
Hamiltonian constraint, including the potential and electric energy, rather
than simply keeping the RN metric. `bh_mass` specifies the seed RN mass;
the Gaussian adds a small amount of ADM energy, reported in the history.
`initial_data=zero` provides an exactly stationary RN check. The inner boundary
must lie within a trapped region; the code stops if this condition is lost.
The outer scalar boundary is radiative. The inner region uses outflow
excision, with no physical data prescribed inside the black hole.

### Relaxed profiles and controlled perturbations

`initial_data=profile` loads an evolved or independently constructed PG profile
without replacing it by an RN seed. Export a relaxed snapshot with:

```bash
python3 plugins/scalar_field/export_ems_profile.py run/scalar_field.out1.final.phdf \
  --input run/input.in --output relaxed.profile
```

Set `profile_file` to this file and set `amplitude=0` for an unperturbed hold.
For a perturbation, `amplitude` adds a Gaussian to phi and
`momentum_amplitude` adds a Gaussian to P; `center` and `width` specify their
common support. The mass constraint is solved again with the changed scalar
data. The first active cell retains the imported interior mass, so place the
perturbation outside the excision boundary. No change is made to the potential
or Maxwell coupling during this operation.

The file contains four columns `(r,phi,P,m)`, with fourth-order local polynomial
interpolation. It must cover every target cell center. The loader rejects
missing coverage, nonfinite data, nonmonotone radii, and mismatched action
coefficients. In particular, exporting a coarse grid does not authorize
extrapolation towards the excision boundary when importing a finer grid.
The first two lines specify the format and model:

```text
# pangu_ems_profile_v1
# polynomial exponential 0.9 5 0 0 0 1
```

The second line lists potential, electromagnetic coupling, Q, alpha, mu^2,
lambda, nu, and f_a in the action convention above. Profiles imported from a
relaxed numerical evolution must be described as such; the import operation
does not turn them into independent exact equilibrium solutions.

The portable CUDA regression checks import/export, action mismatch rejection
and restart consistency:

```bash
python3 tst/regression/scalar_profile_import.py \
  --pangu build-cuda-scalar/src/pangu_scalar --workdir /tmp/ems-profile-check
```

The work directory must not already exist. Long-time stability and resolution
convergence remain separate scientific tests under
`tst/benchmark/matter_suite/BENCHMARKS.txt`.

The coupled fields are:

| Field | Components |
|---|---|
| `scalar.state` | phi, P, m (checkpointed) |
| `scalar.background` | r, N, unused=0, unused=0, physical F_ab F^ab; zero slots are not curvature invariants |
| `ems.geometry` | lapse a, z, N=1-z^2, m |
| `ems.electric` | Eulerian radial E, enclosed charge r^2 f E |
| `ems.energy` | scalar rho, electric rho, total rho, radial momentum -P X |
| `ems.constraints` | radial mass residual, lapse residual, Gauss derivative, temporal Einstein residual |

Energy and momentum outputs use the canonical normalization in G_ab=2 T_ab
of this action. Converting to a Z4c interface that expects G_ab=8 pi T_ab
requires the explicit factor 1/(4 pi); these arrays must not be passed to that
interface unchanged.

The apparent horizon is the outermost root of N=0. Cubic interpolation gives
r_h and phi_h; the irreducible mass is r_h/2. `ems_history.csv` records these,
finite-radius ADM mass estimates, outer luminosity, and constraint residuals.
The mass estimate adds the electrovac Coulomb tail Q^2/(2R) to m(R); it does
not include scalar energy or radiation outside R. Conservation is therefore
tested with m(R) plus the integrated flux through that same finite surface.
`diagnostic_interval` sets its cadence. Restart reconstructs derived geometry
and truncates any later history rows before continuing, avoiding duplicate or
backwards time entries. `dissipation` controls a sixth-derivative KO filter of
the scalar variables; the electrovac mass profile is never filtered.

Use independent convergence, constraint, time-Einstein-residual and stationary
profile checks before changing resolution, boundary locations or model
parameters. This backend is a fully coupled spherical formulation, not an
implicit claim of AMR, multi-rank, rotating or three-dimensional EMS support.

References: [Zhang et al.](https://arxiv.org/abs/2203.10341),
[Guo, Wang and Zhang](https://arxiv.org/abs/2403.02089),
[Guo, Wang, Wu and Yang](https://arxiv.org/abs/2102.04015),
[Benkel, Sotiriou and Witek](https://arxiv.org/abs/1612.08184).

Local results, reference solvers, formula checks and figures are in the parent
workspace at `PANGU_NR/validation/scalar_field_20261102/`.
Run `python3 validation/scalar_field_20261102/validate_local.py` from that
workspace to reproduce the CUDA suites after building `pangu_scalar`.
The independent reference scripts require NumPy, SciPy, SymPy, h5py and Matplotlib.

For a completed spherical EMS run whose deck is saved as `input.in`, run
`python3 plugins/scalar_field/analyze_ems.py /path/to/run`. The script writes
`analysis_ems.json` and `static_reference_v2.npz`. It reads the actual potential
and coupling coefficients, compares the exterior with a static ODE in a
different gauge, and estimates finite-radius mass-plus-flux balance using the
recorded nonuniform history times. Its Simpson/decimation comparison diagnoses
history quadrature resolution; it does not replace a denser-output control when
that comparison is unresolved. The alpha=5, M=1, Q=0.9 massless exponential
benchmark uses `log|d phi_h/dt|` in a documented linear-amplitude/time window.
Unavailable growth data are explicitly marked; `log|phi_h|` is not silently
substituted for that observable. Measured horizon boundary values seed the
static ODE, so this checks the remaining exterior equations rather than making
an independent prediction of the horizon values.

Completed three-resolution spherical EMS families can be compared with:

```sh
python3 plugins/scalar_field/convergence_ems.py /run/n8192 /run/n16384 /run/n32768 --output /results/convergence.json
```

Run `analyze_ems.py` on each completed directory first. The comparison checks
matching actions, integrators, domain and final time, then measures both fields
on the same exterior points using degree-five interpolation. It records each
observed order; it does not impose fourth order on every observable or replace
the independent boundary and restart controls.

## Coupled EMS in Cartesian coordinates

Use the full CUDA `pangu` executable built with `PHYSICS=gr METRIC=z4c MODE=sync`:

```ini
<parthenon/job>
problem_id = scalar_field
<plugins>
enabled = scalar_field
<plugin/scalar_field>
backend = ems
coordinates = cartesian
<numerical_relativity>
matter_source = external
rk4_variant = classical
```

`examples/ems_cartesian_vacuum.in` is a small verification deck. The evolved
`scalar.state` components are `(phi, P, D_x, D_y, D_z, C_x, C_y, C_z)`, with
`P=n^a partial_a phi`, `D^i=sqrt(gamma) f E^i`, and `C^i=sqrt(gamma) B^i`.
`scalar.phi` provides the physical scalar in both coordinate modes. Cartesian
initial data include `minkowski`, small `scalar_wave` and `maxwell_wave` fixtures,
and unperturbed `reissner_nordstrom` data (`zero` is also accepted for the latter).
PG accepts `reissner_nordstrom` as the corresponding unperturbed RN state.
Coordinate-specific profile and excision inputs are rejected by the Cartesian
mode. Cartesian EMS requires a three-dimensional mesh and classical RK4.
Constraint-solved spherical scalar-puncture fixtures are described below;
a general nonspherical elliptic initial-data solver is separate.

Geometry and matter right-hand sides are evaluated from the same RK state before
either is updated. Maxwell's electric and magnetic Gauss constraints are output
as `ems.gauss`; native NR Hamiltonian and momentum constraints remain in
`nr.constraints`. No PG radial profile is relabeled as a Cartesian solution.
The coordinate regression covers PG RN preservation, Cartesian wave convergence,
backreaction and MPI-repartitioned restart. Passing these short checks is
separate from validating long scalarized-black-hole evolutions.

### Maxwell constraint cleaning

Cartesian EMS optionally enables covariant generalized-Lagrange-multiplier
cleaning with `maxwell_cleaning=true` in `<plugin/scalar_field>` and
`cleaning_damping=1` (a nonnegative inverse-time coefficient). The default is
`false`, retaining the eight-component layout and existing restart files.
Enabling it appends `psi_D,psi_B`; generate a fresh ten-component initial
checkpoint rather than changing the layout of an eight-component restart.
The scalar potential, Maxwell coupling, stress energy and NR representation
`chi=psi^-4` do not change.

For `g=sqrt(det(gamma))`, the additions to the existing antisymmetric-flux
Maxwell RHS are

```
d_t D^i += beta^i partial_j D^j - alpha g gamma^ij partial_j psi_D
d_t C^i += beta^i partial_j C^j - alpha g gamma^ij partial_j psi_B
d_t psi_D = beta^i partial_i psi_D - alpha (partial_i D^i/g + kappa psi_D)
d_t psi_B = beta^i partial_i psi_B - alpha (partial_i C^i/g + kappa psi_B)
```

The electric extension applies to the action's tensor `f(phi) F`, matching
the stored `D=g f E`; `psi_D` is not the unweighted electric-field multiplier.
The magnetic multiplier has the opposite sign to the `Phi` convention in
[Hirschmann et al.](https://arxiv.org/abs/1706.09875). These are numerical
constraint-control variables with no physical stress energy. They vanish on
the Maxwell constraint surface. Both propagate at light speed; the timestep
also bounds the explicit damping rate. RK4, ghost exchange, high-order AMR
transfer and native restart include both variables.

This follows the divergence-cleaning approach used for Einstein-Maxwell
evolution and in [three-dimensional EMS simulations](https://arxiv.org/abs/2606.21549),
adapted to this plugin's densitized variables. It does not by itself establish
long-time strong-field stability. The regression
`tst/regression/scalar_ems_cleaning.py` checks electric and magnetic longitudinal
damped waves, nonunit coupling, constant-shift transport, transverse-wave
equivalence and restart with a change in MPI rank count.

### Cartesian mesh refinement

`scalar.state` uses the native Z4c high-order point prolongation and restriction;
the fields are finite-difference point values, including the densitized electric
and magnetic components. The synchronized stage graph exchanges matter and
geometry together and reconstructs ADM fields and Einstein sources after mesh
changes. The fixed coordinate constraint mask is re-evaluated on each level's
cell centers rather than interpolated as a physical field. Tracker-following
masks retain the native NR update.

Use `parthenon/mesh/refinement=static` with standard
`parthenon/static_refinementN` boxes, or `refinement=adaptive` with native NR
tracker/chi criteria or Parthenon refinement criteria on `scalar.state`.
Refinement supports `nghost=2` or `4`. The native restriction stencil needs
three or five interior cells respectively on **every block axis**; because
AMR blocks have even sizes, the minimum block sizes are four or six.
An undersized block is rejected before evolution. In particular, FD4 with
`nghost=4` cannot use four-cell blocks. Sixteen-cell production blocks meet
this requirement.

The FD4 regression compares volume-weighted scalar/Maxwell wave errors and
electric/magnetic Gauss residuals across a coarse/fine interface, observes
actual refinement and derefinement from physical leaf boxes, and checks a
one-to-two-rank restart. Primary states have their own restart tolerance;
second-derivative constraint diagnostics use a separate machine-roundoff bound
proportional to `epsilon/h_min^2`. Gauss residuals at AMR interfaces are measured
for convergence; they are not assumed to vanish to roundoff.

```sh
python3 tst/regression/scalar_ems_amr.py \
  --pangu build-cuda-nr/src/pangu --workdir /tmp/ems-amr
```

The evolved conformal variable is `chi=psi^-4`. The compatibility input
`numerical_relativity/chi_psi_power` accepts only `-4`; other values are rejected.
The local `ems_cartesian_rn_hold.in` example explicitly selects `-4`, disables
the conformal-variable floor, and uses geometric KO=0.02. The KO=0.1 control for
this local charged-puncture problem loses positivity before0.5M; adding a chi
floor alone also fails. The smaller geometric coefficient is a problem-specific
numerical setting, and must still pass charge, constraints, horizon and restart
checks. It does not change the native solver's defaults or the archived vacuum
NR comparison configurations.

After the resolved RN hold passes `analyze_ems_rn_hold.py`, the following check
reuses its N96 result and runs fresh N64/N80 cases with the same numerical
settings. It checks initial and evolved Hamiltonian/electric Gauss convergence,
evolved momentum convergence, and fixed-sphere charge at0.5M. The reference
directory must retain `input.in`, its field outputs and `RN_HOLD_ANALYSIS.json`.
Strict horizon acceptance applies to the N96 reference; the coarse runs do not
claim horizon acceptance.

```sh
python3 tst/regression/scalar_ems_rn_strong.py \
  --pangu build-cuda-nr/src/pangu --reference-run /tmp/ems-rn-hold \
  --workdir /tmp/ems-rn-strong-convergence
```

Cartesian initial data apply `problem/constraint_mask_radius` and
`numerical_relativity/constraint_mask_outer_radius` to the native NR diagnostic
mask. These select a fixed coordinate shell about the origin; they affect
constraint statistics, not the evolved fields or excision. The mask is included
in checkpoints. The standard NR tracker option can refresh a moving mask when
puncture tracking is enabled.

`examples/ems_cartesian_rn.in` and the following local regression check the
unperturbed electrovac RN data on three uniform grids, Hamiltonian and electric
Gauss convergence before and after a short evolution, the analytic electric
field and energy normalization, history L1/L2/Linf against the actual masked
field, and restart from one to two MPI ranks:

```sh
python3 tst/regression/scalar_ems_rn.py \
  --pangu build-cuda-nr/src/pangu --workdir /tmp/ems-rn-check
```

The example's grids leave the horizon unresolved. Its convergence shell is
`1.5 < r <= 3` at `t=0` and `t=0.25M`; passing this test establishes short
exterior electrovac checks, not a long-time black-hole or scalar-hair result.

The nonlinear homogeneous check uses nonzero scalar momentum and electric
density together, with `f=exp(5 phi^2)` and a polynomial potential. An explicit
synthetic restart fixture supplies homogeneous, Hamiltonian-satisfying Bianchi I
initial data. The test compares the evolved physical metric, extrinsic curvature,
scalar and momentum against an independent Python integration of the diagonal
ADM equations in proper time, checking temporal convergence with three CFLs:

```sh
python3 tst/regression/scalar_ems_homogeneous.py \
  --pangu build-cuda-nr/src/pangu --workdir /tmp/ems-homogeneous-check
```

This exercises nonlinear scalar–Maxwell energy exchange and anisotropic
Einstein backreaction. It has no spatial gradients or black hole, and does not
validate black-hole scalar hair, AMR, or long-time strong-field evolution.

`tst/regression/scalar_ems_puncture_constraints.py` also constructs an independent
time-symmetric, spherical scalar-pulse initial condition on an isotropic ADM
slice. With `F=r psi`, zero potential, `P=K_ij=B=0`, and
`D^i=Q x^i/r^3`, the Hamiltonian equation is

    F'' = -F (phi')^2/4 - Q^2/[4 F^3 exp(alpha phi^2)].

The fixture matches analytic electrovac inner and outer regions and shoots for
`psi(infinity)=1`. It uses a native restart file solely as a container for these
explicitly constructed initial fields. Its radial constraint solve is separate
from the subsequent three-dimensional scalar/Maxwell/Z4c evolution; no PG data
are imported. An RN limit, radial integration refinement, three-dimensional
constraint convergence and an unsolved-RN-plus-scalar control check the initial
condition. The added scalar energy changes the ADM mass; the seed mass is not
silently treated as the final ADM mass.

```sh
python3 tst/regression/scalar_ems_puncture_constraints.py \
  --pangu build-cuda-nr/src/pangu --workdir /tmp/ems-scalar-puncture
python3 tst/regression/scalar_ems_puncture_evolution.py \
  --pangu build-cuda-nr/src/pangu --initial-run /tmp/ems-scalar-puncture \
  --workdir /tmp/ems-scalar-puncture-evolution
```

These fixed-parameter checks cover spherical initial data and short coupled
evolution to0.1M. The fixture checks that its native input uses
the required psi^-4 representation. The checks do not provide a general
nonspherical initial-data solver or validate long-time scalarized black holes.

Nonfinite Cartesian EMS states abort with the minimum conformal variable and
spatial-metric determinant recorded in the error. The check preserves the
failure rather than replacing invalid geometry with a floor.
`tst/regression/scalar_ems_invalid_geometry.py` checks this path using an
explicitly invalid synthetic checkpoint; it is an error-handling test.
Nonpositive conformal variables and non-positive-definite spatial metrics are
also rejected.
