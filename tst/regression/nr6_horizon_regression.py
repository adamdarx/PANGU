#!/usr/bin/env python3
"""Audit the formal stationary and boosted NR-6 apparent-horizon runs."""

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


def read_horizon(run: Path) -> tuple[Path, list[dict[str, float]]]:
    path = run / "diagnostics" / "nr_horizon.csv"
    if not path.is_file():
        raise RuntimeError(f"missing horizon table: {path}")
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    if not rows:
        raise RuntimeError(f"empty horizon table: {path}")
    numeric = [{key: float(value) for key, value in row.items()} for row in rows]
    for row in numeric:
        if not all(math.isfinite(value) for value in row.values()):
            raise RuntimeError(f"non-finite horizon row in {path}")
    return path, numeric


def read_history(run: Path) -> tuple[Path, list[list[float]]]:
    candidates = sorted(run.glob("*.hst"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one HST file in {run}, found {len(candidates)}")
    rows: list[list[float]] = []
    with candidates[0].open(encoding="utf-8") as stream:
        for line in stream:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            values = [float(value) for value in line.split()]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"non-finite HST row in {candidates[0]}")
            rows.append(values)
    if not rows:
        raise RuntimeError(f"empty HST file: {candidates[0]}")
    return candidates[0], rows


def summarize(
    run: Path,
    expected_final_cycle: int | None,
    maximum_rms_limit: float | None,
    mass_span_limit: float | None,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    horizon_path, horizon = read_horizon(run)
    history_path, history = read_history(run)
    log_path = run / "run.log"
    if not log_path.is_file() or "Driver completed." not in log_path.read_text(encoding="utf-8"):
        raise RuntimeError(f"run did not complete cleanly: {log_path}")
    if any(row["found"] != 1.0 for row in horizon):
        raise RuntimeError(f"at least one horizon search failed in {horizon_path}")
    if any(row["min_radius"] <= 0.0 or row["max_radius"] < row["min_radius"] for row in horizon):
        raise RuntimeError(f"invalid horizon radius in {horizon_path}")
    maximum_rms = max(row["rms_expansion"] for row in horizon)
    if maximum_rms_limit is not None and maximum_rms > maximum_rms_limit:
        raise RuntimeError(
            f"horizon RMS {maximum_rms:.17g} exceeds {maximum_rms_limit:.17g}"
        )
    mass = [row["irreducible_mass"] for row in horizon]
    mass_span = max(mass) - min(mass)
    if mass_span_limit is not None and mass_span > mass_span_limit:
        raise RuntimeError(
            f"irreducible-mass span {mass_span:.17g} exceeds {mass_span_limit:.17g}"
        )
    final_cycle = int(round(history[-1][2]))
    if expected_final_cycle is not None and final_cycle != expected_final_cycle:
        raise RuntimeError(f"expected final cycle {expected_final_cycle}, found {final_cycle}")
    return {
        "run": str(run),
        "horizon_rows": len(horizon),
        "all_found": True,
        "initial_cycle": int(round(horizon[0]["cycle"])),
        "final_cycle": final_cycle,
        "final_time": history[-1][0],
        "maximum_rms_expansion": maximum_rms,
        "minimum_irreducible_mass": min(mass),
        "maximum_irreducible_mass": max(mass),
        "irreducible_mass_span": mass_span,
        "maximum_iterations": int(max(row["iterations"] for row in horizon)),
        "minimum_radius": min(row["min_radius"] for row in horizon),
        "maximum_radius": max(row["max_radius"] for row in horizon),
        "horizon_sha256": sha256(horizon_path),
        "history_sha256": sha256(history_path),
        "log_sha256": sha256(log_path),
    }, horizon


def read_athenak_summary(path: Path) -> list[dict[str, float]]:
    columns = (
        "cycle",
        "time",
        "christodoulou_mass",
        "spin_x",
        "spin_y",
        "spin_z",
        "spin",
        "area",
        "hrms_mean_square",
        "hmean_integral",
        "radius",
        "min_radius",
    )
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            values = [float(value) for value in line.split()]
            if len(values) != len(columns) or not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"invalid AthenaK horizon row in {path}")
            rows.append(dict(zip(columns, values, strict=True)))
    if not rows:
        raise RuntimeError(f"empty AthenaK horizon summary: {path}")
    return rows


def compare_boosted_to_athenak(
    pangu: list[dict[str, float]], reference_path: Path
) -> dict[str, object]:
    reference = read_athenak_summary(reference_path)
    if len(pangu) != len(reference):
        raise RuntimeError(
            f"PANGU/AthenaK boosted row-count mismatch: {len(pangu)} != {len(reference)}"
        )
    fields = (
        "christodoulou_mass",
        "area",
        "hrms_mean_square",
        "hmean_integral",
        "radius",
        "min_radius",
    )
    maximum_scaled: dict[str, float] = {}
    for field in fields:
        errors = [
            abs(pangu_row[field] - reference_row[field])
            / max(abs(pangu_row[field]), abs(reference_row[field]), 1.0e-12)
            for pangu_row, reference_row in zip(pangu, reference, strict=True)
        ]
        maximum_scaled[field] = max(errors)
    threshold = 3.0e-3
    failures = {field: error for field, error in maximum_scaled.items() if error > threshold}
    if failures:
        raise RuntimeError(f"boosted-horizon AthenaK scaled-error gate failed: {failures}")
    spin_absolute = max(
        abs(pangu_row["spin"] - reference_row["spin"])
        for pangu_row, reference_row in zip(pangu, reference, strict=True)
    )
    # The continuum solution is non-spinning.  AthenaK's Cartesian interpolation
    # leaves an O(1e-5) symmetry residual, whereas PANGU's MeshData reduction
    # cancels it nearly to roundoff; compare this zero quantity in absolute units.
    if spin_absolute > 3.0e-5:
        raise RuntimeError(f"boosted-horizon spin difference {spin_absolute:.17g} exceeds 3e-5")
    return {
        "reference": str(reference_path.resolve()),
        "rows": len(reference),
        "scaled_error_threshold": threshold,
        "maximum_scaled_errors": maximum_scaled,
        "maximum_absolute_spin_difference": spin_absolute,
        "reference_sha256": sha256(reference_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stationary", type=Path, required=True)
    parser.add_argument("--boosted", type=Path, required=True)
    parser.add_argument("--athenak-boosted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stationary, _ = summarize(args.stationary.resolve(), None, 3.0e-2, 5.0e-3)
    boosted, boosted_rows = summarize(args.boosted.resolve(), 200, None, None)
    cross_code = compare_boosted_to_athenak(boosted_rows, args.athenak_boosted.resolve())
    summary = {
        "schema": "pangu.nr6.horizon-regression.v1",
        "acceptance": {
            "stationary_maximum_rms_expansion": 3.0e-2,
            "stationary_maximum_irreducible_mass_span": 5.0e-3,
            "boosted_cross_code_scaled_error": 3.0e-3,
            "boosted_maximum_absolute_spin_difference": 3.0e-5,
            "all_searches_found": True,
            "all_history_values_finite": True,
        },
        "stationary": stationary,
        "boosted": boosted,
        "boosted_athenak_comparison": cross_code,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
