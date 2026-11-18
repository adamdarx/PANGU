# PANGU head-on BNS diagnostics

Jobs 17133234 (N64) and 17254476 (N96) reached 200M. The N96 series joins the original trajectory to the archived continuation at checkpoint cycle 32001, approximately 100.003M. The N128 history is a running snapshot from job 17355477. All 41 N96 frames and the final checkpoint are present in the existing D archive. The completion receipt, current histories, and source hashes accompany this report.

| Observable | N64 | N96 |
|---|---:|---:|
| Last failed and first qualified common horizon times (M) | 138.080–140.081 | 138.137–140.137 |
| Last qualified horizon sample time (M) | 198.127 | 198.138 |
| Last Christodoulou mass | 2.657957 | 2.655735 |
| Excess above solved initial ADM mass 2.654889 | 0.1156% | 0.0319% |
| Proper-volume Hamiltonian RMS at 200M, coordinate shell 8–30 | 9.88e-7 | 2.89e-7 |
| Whole-domain baryon change at 200M | +0.0861% | +0.0804% |

The first-horizon brackets overlap at the approximately 2M search cadence. The latest qualified horizon lies inside radius 5.48 in both runs. The positive AH mass excess is about 72% smaller at N96. Additional resolutions and an independent horizon check would help distinguish evolution error from horizon measurement error.

Whole-domain baryon change includes the interior. The two runs and the live N128 snapshot show maximum relative baryon variation below 3.6e-10 in their sampled 0–80M histories. Run-specific time-zero baryon masses are used. N128 has no qualified common horizon through its recorded searches to about 84M. Its later evolution awaits additional output.

Reproduce the history and horizon tables with python3 summarize.py --source source --output results. The source snapshots are stored at Operations/20261118-bns-series/source on the D archive. Archive coverage lists observed file presence and historical checksums when present in the active index.
