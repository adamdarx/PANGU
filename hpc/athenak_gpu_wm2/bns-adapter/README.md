# BNS table initial-data adapter — pending validation

Reads the existing PANGU_BNS_TIME_SYMMETRIC_V1 constraint-solved table into native AthenaK primitives and ADM fields. The interpolation and initial-data equations follow plugins/compact_objects/bns_initial_data.cc. Native GRMHD, Z4c and finite differences are unchanged. Static refinement and dynamically evolved Z4c are required; prescribed-metric and adaptive-mesh configurations are rejected.

Candidate N64/N96 inputs target200. CUDA sm80 compilation is running on LOGIN, reusing the existing build directory; previously frozen TOV executables and running jobs are unaffected. No BNS runtime or physical validation is claimed. Do not submit formal runs before checking actual GPU execution, resources, initial rho/D/metric matching against archived PANGU fields, and constraint consistency. Short GPU acceptance may use high QOS. Formal runs use low QOS. The full CUDA build and frozen-binary receipt must precede execution.
