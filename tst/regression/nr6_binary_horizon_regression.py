#!/usr/bin/env python3
"""Validate the short equal-mass TwoPunctures dual-horizon NR-6 gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_rows(path: Path) -> list[dict[str, float | str]]:
    if not path.is_file():
        raise RuntimeError(f"missing NR-6 product: {path}")
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    values = [
        {key: value if key == "basis" else float(value) for key, value in row.items()}
        for row in rows
    ]
    numeric = [value for row in values for value in row.values() if isinstance(value, float)]
    if not values or not all(math.isfinite(value) for value in numeric):
        raise RuntimeError(f"empty or non-finite NR-6 product: {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    diagnostics = run / "diagnostics"
    log_path = run / "run.log"
    if not log_path.is_file() or "Driver completed." not in log_path.read_text(encoding="utf-8"):
        raise RuntimeError(f"binary horizon run did not complete: {log_path}")

    horizon_paths = [diagnostics / f"nr_horizon_{index}.csv" for index in range(2)]
    horizons = [numeric_rows(path) for path in horizon_paths]
    if any(len(rows) != 1 for rows in horizons):
        raise RuntimeError("the one-step binary gate must contain one row per horizon")
    left, right = horizons[0][0], horizons[1][0]
    if left["found"] != 1.0 or right["found"] != 1.0:
        raise RuntimeError("at least one binary apparent horizon was not found")
    if left["iterations"] > 10.0 or right["iterations"] > 10.0:
        raise RuntimeError("binary apparent horizon exceeded its iteration budget")

    odd_pairs = (("center_x", 0.0), ("center_y", 0.0), ("center_z", 0.0))
    center_symmetry = max(abs(left[name] + right[name] - offset) for name, offset in odd_pairs)
    even_fields = (
        "radius",
        "area",
        "coordinate_area",
        "irreducible_mass",
        "christodoulou_mass",
        "spin",
        "hmean_integral",
        "hrms_mean_square",
        "rms_expansion",
        "min_radius",
        "max_radius",
    )
    even_symmetry = max(abs(left[name] - right[name]) for name in even_fields)
    if center_symmetry > 1.0e-10 or even_symmetry > 1.0e-10:
        raise RuntimeError(
            f"binary inversion symmetry failed: center={center_symmetry}, even={even_symmetry}"
        )
    center_error = max(abs(abs(left["center_x"]) - 3.257), abs(abs(right["center_x"]) - 3.257))
    if center_error > 1.0e-8:
        raise RuntimeError(f"binary tracker/horizon center error {center_error}")
    if max(abs(left["irreducible_mass"] - 0.505),
           abs(right["irreducible_mass"] - 0.505)) > 1.0e-3:
        raise RuntimeError("binary irreducible masses are inconsistent with the target masses")

    auxiliary: dict[str, list[dict[str, float | str]]] = {}
    for kind in ("coefficients", "iterations", "surface"):
        for index in range(2):
            path = diagnostics / f"nr_horizon_{kind}_{index}.csv"
            auxiliary[f"{kind}_{index}"] = numeric_rows(path)
    if any(len(auxiliary[f"iterations_{index}"]) != int(horizons[index][0]["iterations"])
           for index in range(2)):
        raise RuntimeError("horizon iteration traces do not match the accepted iteration counts")
    if any(len(auxiliary[f"coefficients_{index}"]) != 49 for index in range(2)):
        raise RuntimeError("lmax=6 horizon coefficient output must contain 49 real modes")
    if any(len(auxiliary[f"surface_{index}"]) != 200 for index in range(2)):
        raise RuntimeError("ntheta=10 horizon grid must contain 10x20 surface points")

    report = {
        "schema": "pangu.nr6.binary-horizon-regression.v1",
        "pass": True,
        "run": str(run),
        "horizons_found": 2,
        "iterations": [int(left["iterations"]), int(right["iterations"])],
        "centers": [[left["center_x"], left["center_y"], left["center_z"]],
                    [right["center_x"], right["center_y"], right["center_z"]]],
        "irreducible_masses": [left["irreducible_mass"], right["irreducible_mass"]],
        "center_inversion_symmetry_linf": center_symmetry,
        "even_quantity_symmetry_linf": even_symmetry,
        "center_target_error_linf": center_error,
        "surface_points_per_horizon": 200,
        "real_harmonic_modes_per_horizon": 49,
        "horizon_sha256": [sha256(path) for path in horizon_paths],
        "run_log_sha256": sha256(log_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
