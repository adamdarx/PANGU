"""Locate the late BNS baryon change using six archived PANGU fields."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np


REGIONS = ("r_lt_4", "r_4_to_8", "r_8_to_30", "r_ge_30")


def reduce_mass(path):
    records = {key: dict(baryon_mass=0., recovery_mass_signed=0., recovery_mass_absolute=0.) for key in REGIONS}
    with h5py.File(path, "r") as field:
        info = field["Info"].attrs
        time = float(info["Time"])
        if int(info.get("IncludesGhost", 0)):
            raise ValueError("Expected a field file with physical cells only")
        conserved, recovery = field["mhd.cons"], field["mhd.recovery"]
        for block in range(conserved.shape[0]):
            faces = [field[f"Locations/{axis}"][block, :] for axis in "xyz"]
            x, y, z = [(a[1:] + a[:-1]) * .5 for a in faces]
            dx, dy, dz = [np.diff(a) for a in faces]
            weights = dz[:, None, None] * dy[None, :, None] * dx[None, None, :]
            radius2 = z[:, None, None]**2 + y[None, :, None]**2 + x[None, None, :]**2
            group = np.searchsorted([16., 64., 900.], radius2, side="right").ravel()
            dens = np.asarray(conserved[block, 0, :], dtype=np.float64)
            adds = np.asarray(recovery[block, 0, :], dtype=np.float64)
            if not np.isfinite(dens).all() or not np.isfinite(adds).all() or np.any(weights <= 0):
                raise ValueError(f"Invalid mass data or cell volumes in block {block}")
            reductions = {
                "baryon_mass": dens,
                "recovery_mass_signed": adds,
                "recovery_mass_absolute": abs(adds),
            }
            for observable, values in reductions.items():
                totals = np.bincount(group, weights=(values * weights).ravel(), minlength=len(REGIONS))
                for region, value in zip(REGIONS, totals):
                    records[region][observable] += float(value)
    return time, records


def decode_reduce(entry, zstd, scratch_dir):
    sha = hashlib.sha256()
    total = 0
    with tempfile.NamedTemporaryFile(dir=scratch_dir, suffix=".phdf") as scratch:
        with subprocess.Popen([zstd, "-dc", entry["archive"]], stdout=subprocess.PIPE) as stream:
            for block in iter(lambda: stream.stdout.read(8 * 1024**2), b""):
                sha.update(block)
                scratch.write(block)
                total += len(block)
            if stream.wait() != 0:
                raise RuntimeError(f"Could not decode {entry['archive']}")
        if total != entry["bytes"] or sha.hexdigest() != entry["sha256"]:
            raise ValueError(f"Digest mismatch for {entry['archive']}")
        scratch.flush()
        return reduce_mass(scratch.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--bns-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zstd", default="zstd")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = json.loads(args.index.read_text())["files"]
    all_records = {}
    for resolution, job in ((64, "17133234"), (96, "17254476")):
        run = args.bns_source / f"n{resolution}"
        baseline = json.loads((run / "summary.json").read_text())
        with (run / "nr_horizon.csv").open() as stream:
            rows = csv.DictReader(line for line in stream if not line.startswith("#"))
            horizons = [
                row for row in rows if row["found"] == "1" and float(row["time"]) >= 174
                and float(row["scaled_rms_expansion"]) <= float(row["expansion_tolerance"])
            ]
        horizon_min = min(float(row["min_radius"]) for row in horizons)
        horizon_max = max(float(row["max_radius"]) for row in horizons)
        if not (horizon_min > 4 and horizon_max < 8
                and all(float(row[f"center_{axis}"]) == 0 for row in horizons for axis in "xyz")):
            raise ValueError("Coordinate bins require centered archived horizons between radii 4 and 8")
        frames = []
        for nominal_time, label in ((175, "00035"), (180, "00036"), (200, "final")):
            entry = next(row for path, row in index.items()
                         if f"/{job}/formal/" in path and path.endswith(f".bns.{label}.phdf"))
            time, regions = decode_reduce(entry, args.zstd, args.output)
            measured = sum(d["baryon_mass"] for d in regions.values())
            existing = min(baseline["samples"], key=lambda row: abs(row["time"] - time))
            if abs(time - nominal_time) >= .01 or abs(existing["time"] - time) > 1e-8:
                raise ValueError(f"Unexpected field timestamp {time}")
            if abs(existing["baryon_mass"] - measured) > 1e-9:
                raise ValueError(f"Mass integral disagrees with original validation for {job} t={time}")
            recovered = sum(row["recovery_mass_absolute"] for row in regions.values())
            if abs(existing["last_stage_recovery_mass"] - recovered) > 1e-12:
                raise ValueError(f"Recovery diagnostic disagrees with original validation for {job} t={time}")
            frames.append(dict(nominal_time=nominal_time, time=time, archive=entry["archive"],
                               sha256=entry["sha256"], baryon_mass_total=measured, regions=regions))
            print(f"N{resolution} t={time:.4f}: {measured:.12g}", flush=True)
        start, end = frames[0], frames[-1]
        all_records[f"n{resolution}"] = dict(
            job=job, frames=frames, min_measured_horizon_radius=horizon_min,
            max_measured_horizon_radius=horizon_max, last_measured_horizon_time=float(horizons[-1]["time"]),
            baryon_mass_change_175_to_200=end["baryon_mass_total"] - start["baryon_mass_total"],
            regional_baryon_mass_change_175_to_200={
                region: end["regions"][region]["baryon_mass"] - start["regions"][region]["baryon_mass"]
                for region in REGIONS
            },
        )
    results = dict(
        method="Integrate densitized D and last-stage recovery component 0 with coordinate cell volumes.",
        scope="Fixed coordinate radial bins. The recovery values are per-output last-stage diagnostics.",
        cases=all_records,
    )
    (args.output / "SUMMARY.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
