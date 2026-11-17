# Native AthenaK TOV GPU reference on WM2

The original magnetized TOV collapse is compared at N64 and N96 to code
time100. Stable TOV pilots to50 are auxiliary references. Existing PANGU
N128 runs continue separately. BNS direct comparison requires the same
time-symmetric initial-data table and is not submitted by these scripts.

DATA stages the pinned AthenaK/Kokkos source, input source records and
compact original time-zero fields into
`/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-athenak-grmhd-reference`.
`SOURCE_SNAPSHOT.json` identifies the exact previously CUDA-validated
snapshot. The accompanying inherited patch checks I/O and initializes
mesh-only destructor pointers; native evolution and the TOV initializer
are unchanged. LOGIN runs `build_controller.py` to build sm80 and freeze
the binary by SHA256. Its environment comes from the separately validated
ET build in the sibling operation directory. No evolution runs on LOGIN.

`prepare_tov.py` translates the original source decks. Four ghost layers
are needed for native SMR, selecting native FD6. PANGU uses FD4; retain
and disclose this difference as requested, without changing derivative
code or making a matched-condition performance claim. Atmosphere and
boundary implementations also differ and are recorded in `TOV_INPUTS.json`.
N96 uses the native `output9/single_file_per_rank=true` restart option:
the first two-rank shared-restart probe failed with MPI_ERR_IO; the
rank-separated retry passed with the same frozen program. Only restart
layout changes. Binary field output remains shared.

Start `auto_submit_tov.py` once on LOGIN. It waits for the frozen binary,
submits eight-step GPU40G high-QOS probes (one GPU forN64, two forN96),
then submits the corresponding low-QOS five-day formal job only after
its resource and initial-state gates pass. The probes compare original
PANGU density, integrated baryon mass and magnetic field at timezero,
check all emitted fields for finiteness, observe actual GPU execution,
and require at least20% and8GiB of sampled device-memory reserve.
They do not validate horizon accuracy or long-time physics. The temporary
probe defers AH search while retaining AH objects and full output/restart
allocation. Original formal AH settings run with native FastFlow and
need independent quality screening. Wall cost from the probe omits AH
search and cannot certify formal run duration.

Both phases refuse to start with less than200GiB quota free and stop
only their own new process group if free space drops below150GiB.
Existing production programs, jobs and archive journals are untouched.
Read `TOV_BUILD_STATUS.json`, `TOV_SUBMISSIONS.json` and the per-resolution
admission/formal receipts independently; submission is not completion.
