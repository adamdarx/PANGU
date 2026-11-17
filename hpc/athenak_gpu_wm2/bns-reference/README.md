AthenaK BNS references N64/N96 target200M

The initial adapter used mixed integer/Real math calls that compiled with CUDA host-function warnings. Use Real bounds in fmin/fmax, synchronize initialization before dependent conversion, and validate source interpolation directly on GPU. The native GRMHD/Z4c evolution and finite differences are unchanged.

N64 GPU job17373408 passed8steps, full field/restart I/O, finite snapshots, actual CUDA process and memory reserve. Initial density volume L1 difference from independently archived PANGU fields is2.12597e-8; baryon mass relative bias-1.63263e-9. Same736 physical leaf boxes. Peak31569MiB on1GPU80G. N96 GPU admission17373410 remains queued; both200M formal jobs17373409/17373411 depend on their own successful admission. Short-run success is not long-time physical acceptance.

Superseded failed test raw output was directly deleted after diagnosis; logs and receipts remain. Source table and PANGU data are unchanged.
