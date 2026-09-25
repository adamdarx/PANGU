# Native puncture initial data

This directory is reserved for PANGU's native, constraint-solving puncture
initial-data implementation. It must not depend on TwoPuncturesC or another
evolution framework.

The intended module boundary is:

- `puncture_data.h`: puncture parameters, Bowen--York free data, and results;
- `spectral_solver.h/.cc`: compactified spectral grid, Hamiltonian operator,
  nonlinear solve, and convergence diagnostics;
- `interpolation.h/.cc`: interpolation from the spectral representation to
  Parthenon mesh blocks;
- `initializer.h/.cc`: input parsing, ADM construction, and ADM-to-Z4c setup.

The first implementation should support one or two punctures with arbitrary
linear momentum and spin. The solver must report its Hamiltonian residual and
must be validated independently before replacing any existing initializer.
