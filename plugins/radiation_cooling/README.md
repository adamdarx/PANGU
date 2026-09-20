# Radiation cooling plugin

This plugin implements target-thickness cooling for GRMHD. It owns the cooling source, energy
ledger, diagnostics, parameters, and restart fields. It conflicts with `radiation_transport` and
with electron thermodynamics until their energy ledgers have an explicit coupling contract.
