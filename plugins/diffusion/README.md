# Diffusion foundation plugin

This plugin owns diffusion integration policy and the public operator contract in
`pangu/plugin_api/diffusion.h`. Independent Ohmic and viscosity plugins depend on it and register
their own statically compiled operators. The first migrated integration path is `unsplit`; RKL2 is
kept unavailable until its stage recurrence is connected through a dedicated driver extension.
