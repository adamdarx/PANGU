# FM torus magnetic topology

The existing `gr_torus_sane` problem identifier remains the magnetized FM torus
entry point for backward compatibility. The name no longer determines the seed:

```ini
<problem>
fm_torus = true
magnetic_topology = mad  # sane (default) or mad
potential_beta_min = 100.0
```

`sane` preserves the previous potential and its configurable weights exactly:

`A_phi = max[(rho/rho_max)^q ((r/r_edge) sin(theta))^p exp(-r/R) - cutoff, 0]`.

Defaults are `p=0`, `q=1`, `cutoff=0.2`, and no exponential taper (`R=0`).
`mad` selects the large poloidal-loop seed with `p=3`, `q=1`, `R=400`,
`cutoff=0.2`. These are the KHARMA `seed_a<BSeedType::mad>` weights,
with density normalized by the torus peak (identical for `rho_max=1`).
This uses the existing staggered potential, discrete curl, and global MPI
normalization; it does not introduce a uniform vertical field or magnetic
monopole. A closed loop has return flux, not a net magnetic monopole flux.

MAD mode rejects conflicting `potential_*` weights left over from SANE inputs.
Remove those four parameters to accept the defaults, or set them explicitly.
Only the untilted FM branch supports the MAD selector. Invalid topology strings
are rejected. The existing `prograde=true` specifies the positive orbital branch;
with a negative black-hole spin it describes a counterrotating disk.

Normalization is `max(pgas) / max(b^2/2) = potential_beta_min`, reduced globally
across MPI ranks. Despite the legacy parameter name, it is **not** the minimum
of the pointwise plasma beta. Logs report topology, this ratio and max|divB|.

A MAD seed is not proof of a magnetically arrested evolved state. Confirm
saturation of dimensionless horizon magnetic flux and interruption of inflow
over a statistically useful interval. Neither a seed label nor a short smoke
test proves saturation by 30000M. Keep the magnetic topology and flow state
distinct when publishing the SANE/MAD comparison.

## Seven-case campaign

Production inputs are in `input/relativity/electron_mks_7tasks/`:
SANE spins 0, 0.5, 0.9, 0.9375 and MAD spins -0.9375, 0, +0.9375.
All use 384x192x192, two radial 192x192x192 blocks, six electron models,
tlim=30000M and 100M output/restart intervals.

All seven use r_edge=20, r_peak=41 and radial range [1.2,1000]M. The former
[6,12] torus has invalid peak enthalpy for a=-0.9375 with positive disk rotation;
the larger common torus avoids changing fluid initial conditions only for that
case and provides a larger magnetic-flux reservoir. These are fresh runs and
must not reuse the earlier [6,12] / r_out=50 checkpoints.

## Verification

With an MKS/static/wave MPI/HDF5 executable, run:

```bash
python3 tst/regression/fm_magnetic_topology.py \
  --executable build-electron-mks/src/pangu \
  --cases input/relativity/electron_mks_7tasks \
  --workdir /tmp/fm-topology-regression
```

The Python regression uses h5py/numpy locally; the production submission and
restart scripts require only Python's standard library, not h5py or h5dump.
It checks all spins on small two-block meshes, finite fields, six electron
components plus Ktot, divergence, normalization, topology distinction, unchanged
default SANE behavior, default MAD controls, positive disk rotation at negative
spin, one/two-rank initialization and split-vs-continuous evolution.

Reference implementation inspected at KHARMA commit
`df6684740de9db6ed44d77d0161b6726f1c038d1`:

- https://github.com/parthenon-hpc-lab/kharma/blob/df6684740de9db6ed44d77d0161b6726f1c038d1/kharma/prob/seed_B.hpp
- https://github.com/parthenon-hpc-lab/kharma/blob/df6684740de9db6ed44d77d0161b6726f1c038d1/pars/tori_3d/mad.par
