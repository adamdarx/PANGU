Regular-origin Einstein-scalar collapse (development version 0.1.0)

This independent spherical backend starts with m(0)=0 and no charged or
puncture black-hole seed. It shares the public scalar potential policy with
scalar_field, but the two packages must not be enabled together: they own
different evolution formulations of scalar.state.

Action: [R - 2 (d phi)^2 - 4 V]/(16 pi), hence G_ab = 2 T_ab in the
canonical scalar normalization. V = mu^2 phi^2/2 + lambda phi^4/2 + nu phi^6/6.
Metric: ds^2 = -a^2 dt^2 + (dr + a z dt)^2 + r^2 dOmega^2,
z = sqrt(2 m/r), P = (d_t phi)/a - z d_r phi, X = d_r phi.

The equations are
  phi_t = a (P + z X)
  P_t = d_r[r^2 a (X + z P)]/r^2 - a V'(phi)
  m_t = r^2 a (P + z X) (z P + X)
with the radial constraints reconstructed at each classical RK4 substage:
  m_r = r^2 [(P^2 + X^2)/2 + z P X + V]
  a_r/a = -r P X/z,       a(R)=1.
The temporal Einstein equation remains an independent recorded residual.
The origin uses parity conditions and a fourth-order central mass integral.
The initial even shell is
  phi = A {exp[-((r-c)/w)^2] + exp[-((r+c)/w)^2]}, P=0.
It is initially standing with respect to the hypersurface normal, not an
assertion that the coordinate derivative phi_t vanishes in a nonzero shift.

Initial trapped spheres are rejected. After a trapped region develops, the
inner boundary may move only into cells with 1-z^2 < -0.1. Characteristics
-a(z+1) and a(1-z) must both point inward there. collapse.active checkpoints
the moving excision mask. Excised scalar.state cells are frozen; their derived
fields are zero and masked, not physical interior data. The outermost root
of 1-2m/r=0 gives the spherical apparent horizon and M_irr=r_h/2. No detected
horizon is reported with horizon_found=0, not a reused earlier measurement.

Build a small standalone executable without rebuilding the GRMHD main target:
  cmake -S . -B build-cuda-scalar [the existing CUDA configuration options]
  cmake --build build-cuda-scalar --target pangu_collapse -j2
The full application plugin is available for PHYSICS=gr METRIC=cks MODE=dynamic.
Use one uniform radial MeshBlock and one rank; no AMR or 3D claims are made.
  build-cuda-scalar/src/pangu_collapse -i plugins/scalar_collapse/examples/dispersion_admission.in
  python3 tst/regression/scalar_collapse.py --pangu build-cuda-scalar/src/pangu_collapse --workdir /new/qualification/path

Additional controls reuse a completed N512 R40 T20 collapse run with the same
binary hash, instead of repeating that reference:
  python3 tst/regression/scalar_collapse.py --pangu /frozen/cuda/binary --workdir /new/controls/path --cases weak_massive controls --reference /completed/n512/qualification
The massive weak-field reference solves U_tt=U_rr-mu^2*U, U=r*phi, by Fourier
evolution on a distant odd periodic extension. It is independent of the finite
difference implementation. Controls halve the CFL and double outer radius at
fixed spacing; the horizon-shift project tolerance is 2e-6 relative.

The regression checks exact Minkowski, independent analytic spherical-wave
convergence through the origin, subcritical dispersion, apparent-horizon
formation, causal excision and restart. It requires the supplied binary to
use CUDA. Retain failed runs and their inputs; exit code zero is insufficient.
Development status persists until those checks and strong-field convergence
are qualified. Example runs are short admission tests, not production evidence
of critical exponents or universal scaling.

Reference formulation: Ziprick and Kunstatter, arXiv:0812.0993, equations
(2.21)-(2.24) and (4.2). Their scalar normalization, initial pulse and lapse
normalization differ; our initial-data family is specified above. This is a
formation benchmark inspired by that work, not a reproduction of its numerical
critical threshold or time coordinate.
