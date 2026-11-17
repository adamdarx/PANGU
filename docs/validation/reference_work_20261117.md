# Independent-reference work, 2026-11-17

All new evolution runs use CUDA GPUs. Local short pilots are allowed; CPU
resources are used for compilation and analysis. Production single-black-hole
500M and TOV/BNS N128 jobs retain their frozen programs and inputs.

## Einstein Toolkit on WM2

CarpetX is the AMReX grid driver, SpacetimeX/Z4c supplies spacetime evolution,
and AsterX supplies GRMHD. AsterX is retained for the matter comparisons; it
does not need to be active in a vacuum run. See the upstream
[CarpetX](https://github.com/EinsteinToolkit/CarpetX),
[SpacetimeX](https://github.com/EinsteinToolkit/SpacetimeX), and
[AsterX](https://github.com/EinsteinToolkit/AsterX) repositories.

The isolated WM2 build uses DATA for sources, LOGIN for CUDA compilation,
and Slurm GPU compute allocations for execution. The build recipes and source
pins are in `hpc/et_gpu_wm2`. The source selection follows AsterX's current
`scripts/asterx.th` CI recipe, with a compatible API pin:
CarpetX `f5ef96c1109e12af4275937ab10c06b8076f7026`,
SpacetimeX `e162dd1111b149618cd5ed964656bfa41e17e6f3`, and AsterX
`b046e1d9f6a3026fc1f40a0e883d587d7decc45c`. The official CarpetX main checkout
lacked AsterX dev's subcycling API. The latest maintainer CarpetX dev HEAD
`29c2901` had already refactored the private communication API used by AsterX;
the selected `f5ef96c` retains the required headers, function signatures,
subcycling parameter and PreRestrict schedule group. No missing API was
stubbed or disabled, and no evolution equations were modified. The failed
layout builds were preserved. SpacetimeX is pinned to the same-period
pre-restructure revision using CarpetX's original patch-data/symmetry API.
The complete 64-thorn CUDA executable compiled successfully on WM2 at
2026-11-17 05:38:47 UTC. Its frozen SHA256 is
`7379fd2b9b5edff39c4a384a617439058e401bc646b652d7dae9db471aae18c6`.

The build includes TwoPuncturesX for spectral BBH initial data. The separate
PETSc-dependent Punctures initial-data provider is omitted. CUDA compilation
targets A100/A800 `sm_80`, C++20, GCC12.2, CUDA12.6, CUDA-aware MPI, and the
existing parallel HDF5. CUDA device images are checked before freezing the
executable. Official WaveToyX/standing and AsterX/magTOV_Z4c_AMR_order4
regressions then run on one Slurm GPU and compare against upstream reference
outputs. Their success is technical validation, not the final research
comparison or module acceptance. GPU device images passed inspection;
allocated runtime regression job17370919 passed both selected official
tests on an A10040GB at06:08:03UTC. The suite compared144 magnetic-TOV
files and58 WaveToy files successfully within its stated tolerances;
an actual GPU process was observed. This is a technical validation gate.
Short acceptance allocations may use
high QOS, as authorized by the user; formal long runs retain low QOS.

## Formal GRMHD references

AthenaK and ET direct TOV/BNS comparisons are restricted to N64 and N96.
Existing PANGU N128 runs continue separately. The completed stable-star
pilots below are auxiliary evidence and do not replace the original
magnetized TOV collapse to time100 or the time-symmetric BNS to time200.

The isolated WM2 AthenaK CUDA build uses upstream commit
`8a5d398cb50c7621d03127008aad52c80d579c9c` and Kokkos
`6739bc623081648af9e752b616d9671527922cbf`. The source snapshot SHA256 is
`2b570de2379209b362ba6077dfd836007808bcbb4770233654baacb5f991a5ac`.
Inherited changes check I/O errors and initialize mesh-only destructor
pointers; evolution kernels and the native TOV initializer are unchanged.
The user requested retaining the native finite-difference implementation:
SMR uses four ghost layers and native sixth-order spacetime differences,
while PANGU's corresponding cases use fourth order. This difference is
reported explicitly, and these runs do not establish a matched-condition
performance advantage.

The TOV N64/N96 input candidates reproduce the original456-block physical
refinement coverage and stellar/magnetic parameters. Their native mesh
checks passed locally. High-QOS allocated probes must verify actual GPU
execution, memory reserve, finite fields and initial-state agreement with
the archived PANGU time-zero fields before automatic low-QOS formal
submission. Both probes passed: N64 job17371496 used about19.55GiB on
one A10040GB; N96 job17371505 used about21.30GiB per GPU on two A10040GB.
Initial density and magnetic-field volume L1 differences are approximately
2.2e-8, and initial baryon-mass relative biases are below8e-11.
Formal time100 jobs17371502(N64) and17371507(N96) were observed running
at06:24UTC. N96's first probe17371497 failed in shared restart MPI-I/O;
the successful retry uses native per-rank restart files through an input
option. No evolution/derivative code changed. See
`evidence/athenak_tov_submission_20261117.json`.
These resource probes are not physical acceptance. BNS needs
the same time-symmetric initial-data table before a direct comparison;
unrelated inspiral data are not a substitute.

## Existing single-BH resolution labels

Actual frozen time-zero PHDF headers show that N96 and N192 share all
3,498,736 leaf cells in8<r<30, despite different finest-grid spacings.
The integer global-cell identity hash is
`47ff535da25ce1ecdf57cbceb3bb48caba2d15c7c706c4bb63a0f4538d0954b5`.
This is an initial-mesh result; the production mesh mode is adaptive,
so it must not be generalized to later outputs without checking.
The previous single-BH constraint analysis used2<r<=10, requiring its
own audit. See `evidence/single_bh_initial_mesh_20261117.json`.
The authorized stationary/spinning N32/N48/N64 supplement must demonstrate
actual physical refinement before an order curve is fitted. All original
N96/N128/N192500M runs remain unchanged.

## BBH resolution candidates

`tst/benchmark/nr_bbh_resolution_plan.py` derives N32/48/64 root grids from
the archived quasi-circular BBH input, preserving the physical refinement
regions and all physics and time settings. Blocks are 16/24/32 respectively;
all three static layouts contain 904 blocks and eight physical refinement
levels. Finest spacings are 0.03125, 0.0208333333, and 0.015625.

Mesh-only checks passed against both repositories' different Parthenon
dependencies. WM2 job 17369148 requests four GPUs for a bounded memory and
step-cost admission test. Full 300M runs are not approved by a mesh-only
check; their launch remains contingent on measured admission results.
These BBH root counts do not change the single-black-hole N96/128/192 runs.

## Matched stable TOV pilot

`validation_tov_pair_pangu.in`, `validation_tov_pair_athenak.athinput`, and
`tst/reference/tov_matched_pilot_v1.json` define a B=0 stable TOV pilot with
rho_c=1.28e-3, K=100, Gamma=2, an isotropic uniform N64 grid on [-20,20]^3,
RK2, and CFL0.1, ending at 50 code time units. This extends the existing short comparisons;
it is separate from magnetized collapse and the longer ET stable-star test.

PANGU's GPU trajectory completed 800 steps to time 50. The initial reader and
block-geometry check gives rho_max relative bias -2.58713e-8 and integrated
baryon-mass relative bias -9.31342e-9. AthenaK stores these fields as float32,
whereas PANGU uses float64 here. Those initial differences are not evolution
errors. An inadvertently started CPU reference was stopped; its partial
trajectory is excluded. The AthenaK CUDA trajectory also completed time50
/800steps. Its frozen executable SHA256 is
`c914ab7541144ed8e65803d43ba91a75e3abcc1127e07735cd1e40e138af3199`.
A separate eight-step probe with that same executable recorded CUDA
dispatches for the fluid fluxes, primitive recovery, constrained transport,
Z4c RHS, and Runge–Kutta update. The WSL nvidia-smi process-name observer
could not see processes; its historical false flag is retained, and the
direct Kokkos execution-space proof supplements it.

At eleven shared times from0 to50, the maximum rho_max relative difference
is 6.09275e-7, and the maximum initial-core density relative L1 difference is
9.27123e-7. Both codes' sampled rho_max decrease by about2% during this pilot;
their close agreement does not bound their shared discretization or boundary
error. The initial core uses the reference time-zero rho>1e-4 mask.

Native double-precision history reductions at the same eleven times give
maximum relative baryon drifts of1.49925e-12 (PANGU) and9.96980e-14 (AthenaK).
The AthenaK float32 snapshot integral fluctuates by1.32625e-8 instead; this
storage effect must not be used as evidence that its evolution conserves
mass less accurately. The machine-readable measurements and GPU proof are
in `evidence/tov_matched_gpu_20261117.json`.

`tst/reference/compare_tov_trajectory.py` requires a completed GPU execution
receipt, matching complete sample sets, matching block geometry and times,
finite density, and native mass histories with the matching exact times.
It reports initial bias, normalized density excursions,
core density differences, and each code's snapshot and native-history baryon drift. Both conserved
densities already include sqrt(gamma); no second metric factor is applied.
No convergence, efficiency, or full physical acceptance is inferred from
this single-resolution, finite-domain pilot.

## BNS initial ADM normalization

The actual closed N64 and N96 initial fields were analyzed separately with
`tst/reference/initial_adm_monopole.py`; N96's missing initial field was added
to D and its full decompression SHA256 matched the freshly read server field.
The independently integrated baryon masses reproduce the existing per-grid
initial values. Three analytic/coverage regression tests pass.

| Root grid | Initial ADM fit range | Late mean AH mass excess over that range |
| --- | --- | --- |
| N64 | 2.6548779902–2.6550666443 | +0.06961% to +0.07672% |
| N96 | 2.6548526621–2.6548998063 | +0.01732% to +0.01909% |

The complete fits, field hashes, archive proofs, and late-window evidence
are recorded in `evidence/bns_initial_normalization_20261117.json`.
Fit spread is a diagnostic, not a certified uncertainty interval. The cause
of the residual mass excess remains unresolved; neither two resolutions nor
this normalization check establishes convergence or closes the horizon,
finite-radius, flux, and accumulated recovery-source error budgets.
