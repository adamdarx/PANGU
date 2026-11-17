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
`scripts/asterx.th` CI recipe: CarpetX `29c2901522b595c24d9e1a918fd6600d6050a0cf`,
SpacetimeX `a7d497402abebdfb9b6dd968e085ebdabe9154f8`, and AsterX
`b046e1d9f6a3026fc1f40a0e883d587d7decc45c`. The official CarpetX main checkout
lacked AsterX dev's subcycling API; no missing API was stubbed or disabled.

The build includes TwoPuncturesX for spectral BBH initial data. The separate
PETSc-dependent Punctures initial-data provider is omitted. CUDA compilation
targets A100/A800 `sm_80`, C++20, GCC12.2, CUDA12.6, CUDA-aware MPI, and the
existing parallel HDF5. CUDA device images are checked before freezing the
executable. Official WaveToyX/standing and AsterX/magTOV_Z4c_AMR_order4
regressions then run on one Slurm GPU and compare against upstream reference
outputs. Their success is technical validation, not the final research
comparison or module acceptance. Build and GPU-regression completion remain
pending until the corresponding receipts say otherwise.

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
trajectory is excluded. AthenaK CUDA compilation and the full GPU trajectory
comparison remain pending.

`tst/reference/compare_tov_trajectory.py` requires a completed GPU execution
receipt, matching complete sample sets, matching block geometry and times,
and finite density. It reports initial bias, normalized density excursions,
core density differences, and each code's baryon drift. Both conserved
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
