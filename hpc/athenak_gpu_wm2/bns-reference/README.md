AthenaK BNS references N64/N96 target200M

The initial adapter used mixed integer/Real math calls that compiled with CUDA host-function warnings. Use Real bounds in fmin/fmax, synchronize initialization before dependent conversion, and validate source interpolation directly on GPU. The native GRMHD/Z4c evolution and finite differences are unchanged.

N64 GPU job17373408 passed8steps, full field/restart I/O, finite snapshots, actual CUDA process and memory reserve. Initial density volume L1 difference from independently archived PANGU fields is2.12597e-8; baryon mass relative bias-1.63263e-9. Same736 physical leaf boxes. Peak31569MiB on1GPU80G. N96 GPU admission17373410 remains queued; both200M formal jobs17373409/17373411 depend on their own successful admission. Short-run success is not long-time physical acceptance.

Superseded failed test raw output was directly deleted after diagnosis; logs and receipts remain. Source table and PANGU data are unchanged.

FastFlow input correction: native hmean_tol is the bound that aborts an iteration when abs(integral(H dA)) is too large; it is not an expansion convergence target. Frozen TOV/BNS decks inherited an incorrect1e-5 value. Current new matter wrappers explicitly record/pass fastflow/hmean_tol_0=100 (native default and existing single-BH setting), preserving frozen deck/binary SHA. Native mass_tol1e-5 remains unchanged. Residual quality must be reviewed independently: exported hrms is mean(H^2), so its true RMS is sqrt(hrms), not hrms. This diagnostic correction does not by itself repair nonfinite evolution. See FASTFLOW_INPUT_AUDIT.json under tov-repair.
