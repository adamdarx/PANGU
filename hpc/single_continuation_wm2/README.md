# Current checkpoint continuation
The user explicitly requests practical acceptance (“不需要过分严格”). Original long jobs remain unchanged. StationaryN192 continuation17379019 waits for original17260150; spinningN192 continuation17379020 waits for original17260147. Both retain original four80GB GPUs, fourMPI ranks, frozen executable, original mesh and physics; target absolute500M.

Actual fourGPU checks already validated checkpoint data, recovered the exact evolved state (zero scaled difference) and advanced8finite GPU steps. Only the original wrapper's arbitrary8GiB reserve gate failed, with7575MiB actually free. Model reviews preserve those original SlurmFAILED states and failed receipts while accepting their successful numerical trajectories. The operational margin is4GiB; no repeated GPU pilot is required. Superseded eightGPU jobs were canceled while pending.

Formal startup still verifies binary/input/checkpoint hashes and actual GPU execution, respects original completion/cancellation, validates latest closed checkpoints, and protects production quota. Original500M completion skips continuation. Original TIMEOUT/NODE_FAIL/PREEMPTED permit continuation; user cancellation does not.

Existing AthenaK single-BH data remain used in197 matched0–25M diagnostic points. Historical25M restart archive bytes remain unlocated. Numerical/physical results and native finder quality are documented separately. Dedicated completed-run archive ownership remains unchanged.
