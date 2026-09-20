#!/usr/bin/env python3
"""Validate SYNC-3 independent-state restart and derived-state rebuilding."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


INDEPENDENT_FIELDS = (
    "mhd.cons",
    "mhd.b_face",
    "nr.z4c",
    "nr.constraint_mask",
)
DERIVED_FIELDS = (
    "mhd.prim",
    "mhd.b_cell",
    "mhd.divb",
    "mhd.fofc",
    "mhd.recovery",
    "nr.adm",
    "nr.tmunu",
    "nr.constraints",
)
TRACKER_PARAMS = (
    "numerical_relativity/tracker_position",
    "numerical_relativity/tracker_velocity",
)


def run(command: list[str], directory: Path, log_name: str) -> None:
    environment = os.environ.copy()
    completed = subprocess.run(
        command,
        cwd=directory,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (directory / log_name).write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or "Driver completed." not in completed.stdout:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-8000:]}"
        )


def latest_phdf(directory: Path) -> Path:
    timed: list[tuple[float, Path]] = []
    for path in directory.glob("*.phdf"):
        with h5py.File(path, "r") as stream:
            timed.append((float(stream["Info"].attrs["Time"]), path))
    if not timed:
        raise RuntimeError(f"no PHDF output in {directory}")
    return max(timed)[1]


def checkpoint_at(directory: Path, target: float) -> Path:
    matches: list[Path] = []
    for path in directory.glob("*.rhdf"):
        with h5py.File(path, "r") as stream:
            if float(stream["Info"].attrs["Time"]) == target:
                matches.append(path)
    if not matches:
        raise RuntimeError(f"no t={target} restart checkpoint in {directory}")
    return sorted(matches)[0]


def compare(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    report: dict[str, object] = {}
    with h5py.File(reference_path, "r") as reference, h5py.File(
        candidate_path, "r"
    ) as candidate:
        reference_time = float(reference["Info"].attrs["Time"])
        candidate_time = float(candidate["Info"].attrs["Time"])
        if reference_time != candidate_time or reference_time != 0.15:
            raise RuntimeError(
                f"restart comparison time mismatch: {reference_time} != {candidate_time}"
            )
        for name in ("Levels", "LogicalLocations"):
            if not np.array_equal(reference[name][...], candidate[name][...]):
                raise RuntimeError(f"restart changed AMR topology field {name}")

        field_differences: dict[str, float] = {}
        for name in (*INDEPENDENT_FIELDS, *DERIVED_FIELDS):
            left = np.asarray(reference[name], dtype=np.float64)
            right = np.asarray(candidate[name], dtype=np.float64)
            if left.shape != right.shape or not np.isfinite(right).all():
                raise RuntimeError(f"invalid restarted field {name}")
            field_differences[name] = float(np.max(np.abs(right - left)))

        tracker_differences: dict[str, float] = {}
        for name in TRACKER_PARAMS:
            left = np.asarray(reference["Params"].attrs[name], dtype=np.float64)
            right = np.asarray(candidate["Params"].attrs[name], dtype=np.float64)
            tracker_differences[name] = float(np.max(np.abs(right - left)))

        report["field_maximum_absolute_difference"] = field_differences
        report["tracker_maximum_absolute_difference"] = tracker_differences
        report["maximum_abs_divb"] = float(
            max(
                np.max(np.abs(reference["mhd.divb"][...])),
                np.max(np.abs(candidate["mhd.divb"][...])),
            )
        )
        report["meshblocks"] = int(reference["Info"].attrs["NumMeshBlocks"])

    maximum = max(
        max(report["field_maximum_absolute_difference"].values()),
        max(report["tracker_maximum_absolute_difference"].values()),
    )
    if maximum > 2.0e-14:
        raise RuntimeError(f"restart changed the synchronized state: {report}")
    if report["maximum_abs_divb"] > 1.0e-12:
        raise RuntimeError(f"restart divB gate failed: {report}")
    report["maximum_absolute_difference"] = maximum
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()

    executable = str(args.pangu.resolve())
    input_path = str(args.input.resolve())
    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    continuous = workdir / "continuous"
    split = workdir / "split"
    continued = workdir / "continued"
    for directory in (continuous, split, continued):
        directory.mkdir(parents=True, exist_ok=True)

    common = [
        executable,
        "-i",
        input_path,
        "parthenon/time/dt_force=0.0375",
        "parthenon/output1/dt=0.0375",
        "parthenon/output2/dt=0.0375",
        "parthenon/output3/dt=0.075",
    ]
    run(
        [*common, "parthenon/time/tlim=0.15", "parthenon/time/nlim=4"],
        continuous,
        "run.log",
    )
    run(
        [*common, "parthenon/time/tlim=0.075", "parthenon/time/nlim=2"],
        split,
        "initial.log",
    )
    checkpoint = checkpoint_at(split, 0.075)
    run(
        [executable, "-r", str(checkpoint.resolve()), "parthenon/time/tlim=0.15",
         "parthenon/time/nlim=4"],
        continued,
        "restart.log",
    )

    comparison = compare(latest_phdf(continuous), latest_phdf(continued))
    report = {
        "schema": "pangu.sync3.grmhd-restart.v1",
        "pass": True,
        "comparison_time": 0.15,
        **comparison,
    }
    (workdir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
