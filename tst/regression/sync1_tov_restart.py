#!/usr/bin/env python3
"""Check uninterrupted/restarted SYNC-1 TOV evolution for state continuity."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


STATE_FIELDS = (
    "hydro.cons",
    "hydro.prim",
    "nr.z4c",
    "nr.adm",
    "nr.tmunu",
    "nr.constraints",
    "nr.constraint_mask",
)


def run(command: list[str], directory: Path, log_name: str) -> None:
    completed = subprocess.run(
        command,
        cwd=directory,
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
    candidates = list(directory.glob("*.phdf"))
    if not candidates:
        raise RuntimeError(f"no PHDF output in {directory}")
    timed: list[tuple[float, Path]] = []
    for path in candidates:
        with h5py.File(path, "r") as stream:
            timed.append((float(stream["Info"].attrs["Time"]), path))
    return max(timed)[1]


def compare(reference_path: Path, candidate_path: Path) -> dict[str, float]:
    report: dict[str, float] = {}
    with h5py.File(reference_path, "r") as reference, h5py.File(
        candidate_path, "r"
    ) as candidate:
        reference_time = float(reference["Info"].attrs["Time"])
        candidate_time = float(candidate["Info"].attrs["Time"])
        if reference_time != candidate_time or reference_time != 0.2:
            raise RuntimeError(
                f"restart comparison time mismatch: {reference_time} != {candidate_time}"
            )
        for name in STATE_FIELDS:
            left = np.asarray(reference[name], dtype=np.float64)
            right = np.asarray(candidate[name], dtype=np.float64)
            if left.shape != right.shape or not np.isfinite(right).all():
                raise RuntimeError(f"invalid restarted field {name}")
            report[name] = float(np.max(np.abs(right - left)))
    if max(report.values()) > 2.0e-14:
        raise RuntimeError(f"restart changed the synchronized state: {report}")
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
    continuous.mkdir(parents=True)
    split.mkdir(parents=True)
    continued.mkdir(parents=True)
    common = [
        executable,
        "-i",
        input_path,
        "parthenon/time/nlim=20",
        "parthenon/time/dt_force=0.05",
        "parthenon/output1/dt=0.05",
        "parthenon/output2/file_type=rst",
        "parthenon/output2/id=restart",
        "parthenon/output2/dt=0.1",
    ]
    run([*common, "parthenon/time/tlim=0.2"], continuous, "run.log")
    run([*common, "parthenon/time/tlim=0.1"], split, "initial.log")
    restart_files = sorted(split.glob("*.rhdf"))
    if not restart_files:
        raise RuntimeError("split TOV run produced no restart checkpoint")
    timed_restarts: list[tuple[float, Path]] = []
    for path in restart_files:
        with h5py.File(path, "r") as stream:
            timed_restarts.append((float(stream["Info"].attrs["Time"]), path))
    restart_time, restart_path = max(timed_restarts)
    if restart_time != 0.1:
        raise RuntimeError(f"expected a t=0.1 restart checkpoint, found t={restart_time}")
    run(
        [executable, "-r", str(restart_path.resolve()), "parthenon/time/tlim=0.2"],
        continued,
        "restart.log",
    )
    differences = compare(latest_phdf(continuous), latest_phdf(continued))
    report = {
        "schema": "pangu.sync1.tov-restart.v1",
        "pass": True,
        "comparison_time": 0.2,
        "maximum_absolute_difference": differences,
    }
    (workdir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for pattern in ("*.phdf", "*.xdmf", "*.rhdf"):
        for path in workdir.rglob(pattern):
            path.unlink()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
