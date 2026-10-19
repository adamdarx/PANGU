# Stationary GPU execution benchmark

These inputs compare primary evolution settings: root 32³, MeshBlock 16³,
232 blocks, four refinement levels, FD2, RK4, dt=0.025. Refinement is static;
outputs, trackers, horizons and waveforms are disabled. AthenaK still runs
its ordinary-step ADM constraints. This is not an end-to-end server AMR test.
The AthenaK built-in boosted-puncture generator uses **zero velocity** here.

Run from separate output directories, with one GPU process at a time:

```bash
/path/to/pangu -i /path/to/stationary_multilevel_pangu.in
/path/to/athena -i /path/to/stationary_multilevel_athenak.in
```

Alternate order and compare cycles 4–39. PANGU reports `wsec_step`; AthenaK
uses differences of consecutive `elapsed` values. Retain stalls in cumulative
time, report medians separately, and record GPU clock/temperature. Do not
equate profiler timings with uninstrumented performance.

For Nsight Compute 2025.1, select the RHS and K-O kernels:

```bash
ncu --kernel-name-base demangled \
  --kernel-name 'regex:.*CalculateRHSMeshImpl.*' --launch-count 4 \
  --metrics gpu__time_duration.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv /path/to/pangu -i /path/to/stationary_multilevel_pangu.in \
  parthenon/time/nlim=4 > pangu-ncu.csv
```

For AthenaK replace the filter with `regex:.*CalcRHS.*`, the binary/input
paths, and the override with `time/nlim=4`. Samples 0 and 2 are RHS; 1 and 3
are K-O. For local-memory counters use `--launch-skip 2 --launch-count 1`
and metrics `gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum`.

`stationary_amr_derefine_pangu.in` is a separate field-validation fixture,
not a speed benchmark. It enables HDF5 and tests tracker-controlled AMR.
Its claimed 120→64 block transition must be confirmed from the actual run.

Recorded results are in `results/2026-10-14/`. The `bounded`/`after` timing
files refer to the withdrawn vacuum-RHS `LaunchBounds<128,3>` experiment,
not the current production policy. See the execution-comparison report for
scope, numerical checks and the decision not to enable it.

To check diagnostic freshness, dump `nr.z4c,nr.constraints` together and run:

```bash
python3 check_output_constraints.py final.phdf --require-evolved-theta
```

The constraint theta component must exactly match the evolved Z4c theta.
`Metadata::Derived`/`FillDerived` alone does not ensure field-output refresh;
the NR package explicitly uses Parthenon's `UserWorkBeforeOutputMesh` hook.
It does not run on ordinary no-output steps. The optional evolved-theta check
is specific to the nontrivial stationary fixture, not exact Minkowski states.

`results/2026-10-14/power-one/` records a withdrawn exponent-one fast-path
experiment: NCU measured less RHS work, but normal-run GPU clocks varied and
the fields were not bitwise identical. `output-refresh/` contains logs for
the stale-output regression and its fix. Historical constraints output from
before this fix may reflect initialization or the last remesh, not file time.
