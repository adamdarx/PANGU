# Viscosity plugin

This plugin provides isotropic kinematic viscosity for hydro and MHD fluids. It depends on the
diffusion foundation and registers an unsplit diffusive-flux operator plus its timestep constraint.

```text
<plugins>
enabled = viscosity

<plugin/viscosity>
nu = 0.25
cfl = 0.5
```
