#!/usr/bin/env python3
"""Audit an NR-6 equal-mass binary waveform/horizon validation run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        raise RuntimeError(f"missing diagnostic table: {path}")
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    values = [{key: float(value) for key, value in row.items()} for row in rows]
    if not values or not all(math.isfinite(value) for row in values for value in row.values()):
        raise RuntimeError(f"empty or non-finite diagnostic table: {path}")
    return values


def history(path: Path) -> tuple[list[str], np.ndarray]:
    header = ""
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("# [1]=time"):
                header = line
            elif line.lstrip().startswith("#") or not line.strip():
                continue
            rows.append([float(value) for value in line.split()])
    names = [token.split("=", 1)[1] for token in header.split() if "=" in token]
    values = np.asarray(rows, dtype=np.float64)
    if not names or values.ndim != 2 or values.shape[1] != len(names):
        raise RuntimeError(f"invalid history layout: {path}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite history output: {path}")
    return names, values


def hdf_audit(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as stream:
        time = float(stream["Info"].attrs["Time"])
        cycle = int(stream["Info"].attrs["NCycle"])
        extrema: dict[str, list[float]] = {}
        for name in ("nr.z4c", "nr.adm", "nr.constraints", "nr.weyl"):
            values = np.asarray(stream[name], dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"{path}: non-finite {name}")
            extrema[name] = [float(np.min(values)), float(np.max(values))]
        z4c = np.asarray(stream["nr.z4c"], dtype=np.float64)
        if float(np.min(z4c[:, 0])) <= 0.0 or float(np.min(z4c[:, 18])) <= 0.0:
            raise RuntimeError(f"{path}: non-positive chi or lapse")
    return {"path": str(path), "time": time, "cycle": cycle, "extrema": extrema,
            "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-final-time", type=float, default=0.5)
    args = parser.parse_args()
    run = args.run.resolve()

    log_path = run / "run.log"
    log = log_path.read_text(encoding="utf-8")
    if "Driver completed." not in log:
        raise RuntimeError(f"run did not complete: {log_path}")
    wall_matches = re.findall(r"walltime used = ([+\-0-9.eE]+)", log)
    throughput_matches = re.findall(r"zone-cycles/wallsecond = ([+\-0-9.eE]+)", log)

    horizon = [csv_rows(run / "diagnostics" / f"nr_horizon_{index}.csv") for index in range(2)]
    if len(horizon[0]) < 5 or len(horizon[1]) != len(horizon[0]):
        raise RuntimeError("binary run needs at least five matched horizon times")
    for index, rows in enumerate(horizon):
        if any(row["found"] != 1.0 for row in rows):
            raise RuntimeError(f"horizon {index} contains a failed search")
        maximum_rms = max(row["rms_expansion"] for row in rows)
        if maximum_rms > 3.0e-2:
            raise RuntimeError(f"horizon {index} RMS {maximum_rms:.17g} exceeds 3e-2")
    horizon_symmetry = max(
        abs(left[key] - right[key])
        for left, right in zip(horizon[0], horizon[1], strict=True)
        for key in ("radius", "area", "irreducible_mass", "rms_expansion", "min_radius",
                    "max_radius")
    )
    if horizon_symmetry > 2.0e-10:
        raise RuntimeError(f"dual-horizon inversion symmetry error {horizon_symmetry:.17g}")

    waveform_path = run / "diagnostics" / "nr_waveforms.csv"
    waveform = csv_rows(waveform_path)
    radii = sorted({row["radius"] for row in waveform})
    times = sorted({row["time"] for row in waveform})
    if radii != [5.0, 7.0, 9.0] or len(times) < 5 or len(waveform) != 3 * len(times):
        raise RuntimeError(f"invalid multi-radius waveform grid: radii={radii}, times={times}")
    mode_symmetry = max(
        abs(row["re_l2_m-2"] - row["re_l2_m2"])
        for row in waveform
    )
    mode_symmetry = max(
        mode_symmetry,
        max(abs(row["im_l2_m-2"] + row["im_l2_m2"]) for row in waveform),
        max(abs(row["im_l2_m0"]) for row in waveform),
    )
    if mode_symmetry > 5.0e-10:
        raise RuntimeError(f"equal-mass waveform symmetry error {mode_symmetry:.17g}")

    history_paths = sorted(run.glob("*.hst"))
    if len(history_paths) != 1:
        raise RuntimeError("binary run must contain exactly one history file")
    names, values = history(history_paths[0])
    tracker_columns = [index for index, name in enumerate(names) if "puncture_tracker" in name]
    if len(tracker_columns) != 12:
        raise RuntimeError("binary history does not contain two complete trackers")
    tracker = values[:, tracker_columns]
    tracker_symmetry = float(
        max(
            np.max(np.abs(tracker[:, :3] + tracker[:, 6:9])),
            np.max(np.abs(tracker[:, 3:6] + tracker[:, 9:12])),
        )
    )
    if tracker_symmetry > 5.0e-10:
        raise RuntimeError(f"two-tracker inversion symmetry error {tracker_symmetry:.17g}")

    final_paths = sorted(run.glob("*.out2.final.phdf"))
    if len(final_paths) != 1:
        raise RuntimeError("binary run is missing its unique final HDF5 snapshot")
    final = hdf_audit(final_paths[0])
    if abs(float(final["time"]) - args.expected_final_time) > 5.0e-12:
        raise RuntimeError(f"unexpected final time {final['time']}")

    report = {
        "schema": "pangu.nr6.binary-diagnostics.v1",
        "pass": True,
        "run": str(run),
        "final": final,
        "horizon_samples_per_object": len(horizon[0]),
        "maximum_horizon_rms": [max(row["rms_expansion"] for row in rows) for rows in horizon],
        "horizon_irreducible_mass_span": [
            max(row["irreducible_mass"] for row in rows)
            - min(row["irreducible_mass"] for row in rows)
            for rows in horizon
        ],
        "dual_horizon_symmetry_linf": horizon_symmetry,
        "waveform_samples_per_radius": len(times),
        "waveform_radii": radii,
        "waveform_mode_symmetry_linf": mode_symmetry,
        "two_tracker_symmetry_linf": tracker_symmetry,
        "wall_seconds": float(wall_matches[-1]) if wall_matches else None,
        "zone_cycles_per_second": float(throughput_matches[-1]) if throughput_matches else None,
        "waveform_sha256": sha256(waveform_path),
        "horizon_sha256": [
            sha256(run / "diagnostics" / f"nr_horizon_{index}.csv") for index in range(2)
        ],
        "history_sha256": sha256(history_paths[0]),
        "run_log_sha256": sha256(log_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
