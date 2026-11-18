# PANGU BNS late baryon mass budget

Six archived fields cover 175M, 180M and 200M for the two completed head-on runs. The field hashes match the archive index. Spatial integration uses the archived densitized baryon density and coordinate cell volumes. Integrals over all bins match the original run validation.

The table gives the signed inventory change from the 175M snapshot to the 200M final field. All quantities are in the simulation's mass units.

| Resolution | Whole-domain change | Change at r < 4 | Change at 4 <= r < 8 | Change at 8 <= r < 30 | Change at r >= 30 |
|---|---:|---:|---:|---:|---:|
| N64 | +2.520274e-03 | +4.134058e-03 | -8.445517e-04 | -7.897804e-04 | +2.054859e-05 |
| N96 | +2.354641e-03 | +2.847974e-03 | -3.092797e-04 | -1.874416e-04 | +3.388407e-06 |

The qualified horizon samples over this interval have radii between 4 and 8 and their recorded centers are at the origin. The last horizon samples precede 200M by about 1.9M. The r < 4 and r >= 8 regions provide conservative coordinate buffers around the sampled horizons.

The central r < 4 inventory rises in both runs, while the combined inventory at r >= 4 falls. More than 99.99999% of the absolute last-stage recovery correction at 180M and 200M is inside r < 4.

The mass growth and an increase in the signed last-stage primitive recovery diagnostic appear in the same history interval, beginning at the 177.084M sample for N64 and the 178.941M sample for N96. The per-frame table records the recovery correction by radius. That field is reset each stage. A full source budget would require the corrections from every stage and the flux through the chosen boundaries, including the moving horizon if integrating the physical exterior.

See SUMMARY.json for original field times, archive hashes and radius bounds. The source histories and horizon data used here are saved in the sibling 20261118-bns-series archive.
