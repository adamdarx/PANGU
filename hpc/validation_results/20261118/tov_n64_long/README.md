# N64 TOV: comparison using existing 100-time-unit GPU data

PANGU and AthenaK now have a direct N64 comparison using21 paired field snapshots from t=0 to100. They share physical cells and the stellar model. Nine pre-horizon density samples (t<=40) agree within0.92%. AthenaK first reports a horizon at42.4625, within PANGU's sampling interval42.025–44.0281.

The maximum difference in the baryon inventory of the fixed3<=r<=30 shell is0.0943% of the initial baryon mass. At100, proper-volume Hamiltonian RMS in that shell is1.08e-4(PANGU) and6.06e-6(AthenaK). Native FD order, damping, regularization and reset settings differ; no equal-order accuracy or timing ranking is inferred.

More than99.99999% of the endpoint global baryon difference lies at r<1.8, inside both latest reported horizon radii. Outside r=1.8, the absolute integral of the densitized-density difference is1.71e-10. Post-collapse global accounting is separate from exterior accuracy; this spatial decomposition does not replace a complete flux budget.

At the last matched AH sample98.1125, raw irreducible masses differ by1.17%. PANGU requires mass and expansion convergence; native AthenaK stops on a small mass change. This diagnostic difference is recorded without blocking remaining physical tests.

This is a useful single-resolution comparison for the report. N96 and ET long runs will add independent resolution/code evidence. No extra evolution was run for these figures. Times are code units; no unverified total-system-mass normalization is used.

The figures directory provides PNG, PDF and editable SVG for the physical/exterior comparison and separate AH diagnostics. PHYSICAL_NUMERICAL_SETTINGS.json records actual numerical settings, including PANGU's archived zero damping/dissipation versus the stabilized native reference.

Replot the published reductions:

    python analyze_plot.py --plot-only

This needs NumPy/Matplotlib and adjacent JSON/CSV files, without large raw fields. Full extraction also uses the verified D archive, the recorded native archive index/reader and raw source companion at Operations/20261118-tov-long-comparison/source. All105 field archives were SHA256 verified while decoding. Native field dumps are float32 and PANGU dumps float64; reductions use float64 and conservation curves use high-precision histories.
