#!/usr/bin/env python3
"""Run and audit the optional NR-6 TwoPunctures initial-data path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


DATASETS = {"nr.z4c": 22, "nr.adm": 13, "nr.constraints": 8, "nr.weyl": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_snapshot(path: Path) -> tuple[float, int, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as stream:
        time = float(stream["Info"].attrs["Time"])
        cycle = int(stream["Info"].attrs["NCycle"])
        fields: dict[str, np.ndarray] = {}
        for name, components in DATASETS.items():
            if name not in stream or stream[name].shape[1] != components:
                raise RuntimeError(f"{path}: invalid or missing {name}")
            values = np.asarray(stream[name], dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"{path}: non-finite values in {name}")
            fields[name] = values
    return time, cycle, fields


def metric_determinant(z4c: np.ndarray) -> np.ndarray:
    gxx, gxy, gxz = z4c[:, 1], z4c[:, 2], z4c[:, 3]
    gyy, gyz, gzz = z4c[:, 4], z4c[:, 5], z4c[:, 6]
    return (
        gxx * (gyy * gzz - gyz * gyz)
        - gxy * (gxy * gzz - gxz * gyz)
        + gxz * (gxy * gyz - gxz * gyy)
    )


def read_history(path: Path) -> tuple[list[str], np.ndarray]:
    header = ""
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("# [1]=time"):
                header = line
                continue
            elif line.lstrip().startswith("#") or not line.strip():
                continue
            rows.append([float(value) for value in line.split()])
    if not header or not rows:
        raise RuntimeError(f"{path}: incomplete history output")
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path}: non-finite history value")
    names = [token.split("=", 1)[1] for token in header.split() if "=" in token]
    return names, values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    log_path = workdir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(args.executable.resolve()), "-i", str(args.input.resolve()), "-d", "."],
            cwd=workdir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if "Driver completed." not in log_path.read_text(encoding="utf-8"):
        raise RuntimeError("TwoPunctures smoke evolution did not complete")

    initial_paths = sorted(workdir.glob("*.out2.00000.phdf"))
    final_paths = sorted(workdir.glob("*.out2.final.phdf"))
    history_paths = sorted(workdir.glob("*.hst"))
    if len(initial_paths) != 1 or len(final_paths) != 1 or len(history_paths) != 1:
        raise RuntimeError("TwoPunctures smoke output set is incomplete")
    initial_time, initial_cycle, initial = read_snapshot(initial_paths[0])
    final_time, final_cycle, final = read_snapshot(final_paths[0])
    if initial_time != 0.0 or initial_cycle != 0:
        raise RuntimeError("TwoPunctures initial snapshot is not cycle zero")
    if abs(final_time - 0.025) > 2.0e-14 or final_cycle < 1 or final_cycle > 2:
        raise RuntimeError(f"unexpected final state t={final_time}, cycle={final_cycle}")

    for label, fields in (("initial", initial), ("final", final)):
        z4c = fields["nr.z4c"]
        if float(np.min(z4c[:, 0])) <= 0.0 or float(np.min(z4c[:, 18])) <= 0.0:
            raise RuntimeError(f"{label}: chi and lapse must remain positive")
        determinant_error = float(np.max(np.abs(metric_determinant(z4c) - 1.0)))
        if determinant_error > 2.0e-12:
            raise RuntimeError(
                f"{label}: conformal-metric determinant error {determinant_error:.17g}"
            )

    names, history = read_history(history_paths[0])
    tracker_columns = [index for index, name in enumerate(names) if "puncture_tracker" in name]
    if len(tracker_columns) != 12:
        raise RuntimeError(f"expected 12 two-tracker history columns, found {len(tracker_columns)}")
    tracker = history[-1, tracker_columns]
    position_0, velocity_0 = tracker[:3], tracker[3:6]
    position_1, velocity_1 = tracker[6:9], tracker[9:12]
    symmetry_error = float(
        max(
            np.max(np.abs(position_0 + position_1)),
            np.max(np.abs(velocity_0 + velocity_1)),
        )
    )
    if symmetry_error > 2.0e-12:
        raise RuntimeError(f"two-tracker inversion symmetry error {symmetry_error:.17g}")

    restart_paths = sorted(workdir.glob("*.out3.final.rhdf"))
    if len(restart_paths) != 1:
        raise RuntimeError("TwoPunctures smoke run is missing its final restart")
    restart_dir = workdir / "restart"
    restart_dir.mkdir()
    restart_log = restart_dir / "run.log"
    with restart_log.open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                str(args.executable.resolve()),
                "-r",
                str(restart_paths[0].resolve()),
                "-d",
                ".",
                "parthenon/time/tlim=0.05",
            ],
            cwd=restart_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if "Driver completed." not in restart_log.read_text(encoding="utf-8"):
        raise RuntimeError("TwoPunctures restart evolution did not complete")
    restarted_final_paths = sorted(restart_dir.glob("*.out2.final.phdf"))
    if len(restarted_final_paths) != 1:
        raise RuntimeError("TwoPunctures restart run is missing its final HDF5 output")
    restarted_time, restarted_cycle, restarted = read_snapshot(restarted_final_paths[0])
    if abs(restarted_time - 0.05) > 2.0e-14 or restarted_cycle <= final_cycle:
        raise RuntimeError(
            f"unexpected restarted state t={restarted_time}, cycle={restarted_cycle}"
        )
    if float(np.min(restarted["nr.z4c"][:, 0])) <= 0.0:
        raise RuntimeError("restarted TwoPunctures state has non-positive chi")

    report = {
        "schema": "pangu.nr6.two-punctures-regression.v1",
        "pass": True,
        "initial_time": initial_time,
        "final_time": final_time,
        "final_cycle": final_cycle,
        "restarted_time": restarted_time,
        "restarted_cycle": restarted_cycle,
        "initial_conformal_determinant_linf": float(
            np.max(np.abs(metric_determinant(initial["nr.z4c"]) - 1.0))
        ),
        "final_conformal_determinant_linf": float(
            np.max(np.abs(metric_determinant(final["nr.z4c"]) - 1.0))
        ),
        "two_tracker_symmetry_linf": symmetry_error,
        "initial_constraint_linf": float(np.max(np.abs(initial["nr.constraints"]))),
        "final_constraint_linf": float(np.max(np.abs(final["nr.constraints"]))),
        "initial_sha256": sha256(initial_paths[0]),
        "final_sha256": sha256(final_paths[0]),
        "history_sha256": sha256(history_paths[0]),
        "run_log_sha256": sha256(log_path),
        "restart_sha256": sha256(restart_paths[0]),
        "restarted_final_sha256": sha256(restarted_final_paths[0]),
        "restart_log_sha256": sha256(restart_log),
    }
    summary_path = workdir / "nr6_two_punctures_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
