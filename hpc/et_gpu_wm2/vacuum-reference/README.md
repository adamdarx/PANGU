Native ET GPU vacuum comparisons

Frozen existing executable64f3b5e9b7ed; no thorn/evolution code rebuild. Stationary/spinning isolated bare-mass1 Bowen-York cases use N64, domain400, nine levels, target500M. Spinning Jz0.5. Zero-mass second spectral focus plus center_offset place the physical puncture at the origin. Outer coordinates shift by half a finest cell to avoid punctures on vertices.

QC uses the PANGU physical case: b3.257, bare masses0.483, Py_plus=-0.133, Py_minus=+0.133, no spin, domain256, N32, nine levels, target300M. This is not GW150914. BoxInBox follows PunctureTracker; AHFinderDirect finds horizons; native GPU Weyl and Multipole extract l2m2 at30/40/50.

Stationary/spinning64 GPU admissions17373901/17373902 passed64steps to0.3125, both with initial AH diagnostics, finite global constraint norms, observed CUDA process, and peak32883MiB/81920MiB. Early500M projections89.1/90.4h are estimates, not completion promises. Formal17373867/17373869 are pending resource allocation.

First QC GPU admission17373903 failed at step4 after finding both individual horizons: interpolated shift diverged and PunctureTracker aborted. Formal child17373871 never started. Retry17373933 restores native upstream QC chi_floor1e-6, alphaG_floor1e-8, epsdiss0.32 and offsets the vertex grid by half a finest cell; formal child17373934 remains dependent on success. Root cause and retry outcome are not yet established. PunctureTracker guard and finite checks are retained. This changes only the independent ET reference input, never PANGU production.

Initial full-spectral interpolation trials17373866/68/70 were stopped before evolution after gdb confirmed serial PunctEvalAtArbitPositionFast initialization. Current inputs use upstream default Taylor expansion; quantify this initial interpolation error separately. No CPU evolution fallback. Strict Cactus parameter validation is enabled in current runners.

GPU gates check64steps, actual GPU use, memory reserve, finite norms, initial horizons, three QC wave extraction radii and early throughput. A short gate is technical acceptance only. Native ET gauge omits PANGU slow-start lapse; independently generated spectral initial data and AMR layouts require explicit error budgets and waveform alignment. No identical-grid performance claim. Global norms and axis profiles cannot provide a full3D exterior-only constraint norm. AH diagnostic presence cannot validate final mass/spin precision. Quantitative long-time analysis remains pending.

Existing build outputs diagnostics/profiles/waves and has no full-volume checkpoint backend enabled. Inputs, scripts, SHA receipts and queue IDs are preserved here. The separate ET formal archiver owns completed formal runs only.
