# WM2 Einstein Toolkit CUDA build

Target workspace:
`/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu`.

1. DATA fetches the official CarpetX component list and GetComponents sources.
   `SOURCE_PINS.json` records all fetched repository revisions and the explicit
   CarpetX/SpacetimeX overrides from AsterX's CI recipe. `et-gpu.th` selects
   the 64 required framework, vacuum, and GRMHD thorns. Required bundled
   dependency archives must already exist; LOGIN performs no source download.
2. In the isolated workspace, LOGIN runs `build_controller.py`, which invokes
   `build.sh` with `wm2-cuda.cfg`. The original production binaries are not
   used as build outputs. `cuobjdump --list-elf` must find CUDA images before
   a SHA-named executable is frozen read-only under `bin`.
3. Start `post_build_submit.py` once on LOGIN while the build is running.
   It stops on a failed build and otherwise submits `gpu-smoke.slurm` once.
   The allocation runs two official regressions with the frozen executable,
   verifies both ran and passed, and records an actual GPU process. Read
   `GPU_SMOKE_SUBMISSION.json` and `runs/JOBID/GPU_SMOKE_STATUS.json` for
   submission and execution separately. Neither is a physical acceptance
   claim for PANGU.

Required module paths and existing user MPI/HDF5 paths are in `environment.sh`.
The supplied files are specific to WM2 and this isolated workspace. Source
manifests `FETCH_RESULT.json`, `COMPATIBLE_SOURCE_SELECTION.json`, and
`DEPENDENCY_ARCHIVES.json` must accompany an actual server build. Initial
component retrieval and compatibility overrides are preserved separately.
The current state is recorded by server receipts, not by this README.
