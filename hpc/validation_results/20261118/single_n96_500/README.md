# PANGU N96 single black holes through 500M

| Case | Original job | Qualified horizon samples | Median horizon interval (M) | Last horizon time (M) | Max relative irreducible mass change | Max absolute angular momentum change | Hamiltonian RMS at 500M | Momentum RMS at 500M |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stationary | 17313453 | 998 | 0.501 | 499.801 | 9.527e-05 | 1.584e-10 | 1.318e-06 | 8.227e-07 |
| spinning | 17313451 | 499 | 1.003 | 499.300 | 4.632e-04 | 4.758e-05 | 1.412e-06 | 8.574e-07 |

Each mass and angular momentum change is measured relative to the same run's first qualified horizon. The CSV column named spin is the recorded angular momentum magnitude J. The field cadence can be reconstructed from the 21 constraint samples at nominal intervals of 25M from t=0 through t=500M. Constraints use cell-count RMS over active physical cells defined by the native constraint mask. Ghost cells are excluded.

The field series uses prior archive validation receipts through 475M. The two final fields, completion receipts, and horizon CSVs were decoded and matched by SHA256 to the original archive index. The field and horizon diagnostics retain their original, separate time stamps. The result covers the two completed N96 PANGU runs. Multi-resolution convergence and the AthenaK 500M comparisons remain open. The earlier AthenaK matched data cover 0–25M.
