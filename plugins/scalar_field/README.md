# Scalar-field plugin: test fields and coupled spherical EMS

Plugin version: **1.0.0**. This release covers the documented test-field and
coupled spherical EMS interfaces, including qualified profile import and
controlled perturbations. Collapse from an initially regular, untrapped center
is a separate development task and is not provided by this release.

This package evolves actual Klein–Gordon test fields with Parthenon MeshData,
CUDA kernels, fourth-order finite differences and classical four-stage RK4.
The `cartesian_flat` and `radial_tortoise` backends retain the fixed-background
test-field problems. The `spherical_ems` backend evolves the fully nonlinear
spherically symmetric Einstein–Maxwell–scalar system, including scalar
self-interaction, a dynamical metric, and the scalar-dependent electric field.
All backends currently require a uniform mesh. The coupled backend uses one
radial MeshBlock and one MPI rank. It is not a three-dimensional Z4c matter
solver and does not implement gravitational backreaction of a GB coupling.

## Configuration and extension

Enable with `problem_id=scalar_field` and `<plugins> enabled=scalar_field` in a
fixed-background build (`METRIC=cks`, `MODE=dynamic`).

The `<plugin/scalar_field>` namespace contains:

| Parameter | Meaning |
|---|---|
| `backend` | `radial_tortoise` or `cartesian_flat` |
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
These actions use different Einstein-source normalizations; a future full
backreaction interface must keep that normalization explicit.

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

## Fully coupled spherical EMS

Use `backend=spherical_ems` with `examples/ems_coupled_rn.in` or
`examples/ems_coupled_quartic.in`. The coordinate is now **areal radius r**, not
tortoise radius. The action is the unified EMS action above, with `gb_eta=0`.
The build remains `METRIC=cks MODE=dynamic` to select the application without
the unrelated 3D Z4c driver; the PG geometry in `scalar.state`/`ems.geometry`
is dynamical and is not supplied by the fixed CKS geometry package.
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
