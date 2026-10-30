# Native puncture initial data

This directory contains PANGU's native, constraint-solving puncture
initial-data implementation. It has no dependency on TwoPuncturesC, GSL, or
another evolution framework.

The intended module boundary is:

- `puncture_data.h`: puncture parameters and pointwise solver results;
- `bowen_york.h`: arbitrary-momentum/spin Bowen--York free data and the
  Hamiltonian source;
- `spectral_solver.h/.cc`: rational-Chebyshev compactification and spectral
  differential operators, followed by the Hamiltonian nonlinear solve;
- `interpolation.h/.cc`: interpolation from the spectral representation to
  Parthenon mesh blocks;
- `initializer.h/.cc`: physical ADM construction, puncture-end ADM-mass
  measurement, and target-mass iteration.

The implementation supports one or two Bowen--York punctures with arbitrary
linear momentum and spin. It solves the vacuum Hamiltonian constraint for the
regular conformal-factor correction on a compactified Cartesian
rational-Chebyshev grid, interpolates the result locally onto Parthenon mesh
blocks, and constructs maximal conformally flat ADM/Z4c data. Both direct bare
masses and internal-end target ADM masses are supported. `nr_puncture` is the
native problem name; `nr_two_punctures` and its historical parameter names are
accepted as a compatibility interface.

`problem/puncture_interpolation_points` selects 4 (the compatibility default),
6, or 8 points per axis for mapping the regular correction onto the evolution
mesh. For spinning data, increase this together with
`puncture_spectral_points`: refining only the evolution mesh cannot remove
the fixed spectral-solve and interpolation error. The nonlinear tolerance
measures the residual at collocation points, not the continuum constraint
residual between them. The wider stencil affects initialization, not the
evolution kernels.
