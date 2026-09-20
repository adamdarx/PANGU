#!/usr/bin/env python3
"""Prove that the explicit zero-Tmunu path reduces exactly to vacuum Z4c."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess

import h5py
import numpy as np


def run(command: list[str], directory: pathlib.Path) -> None:
    directory.mkdir(parents=True)
    completed = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    (directory / "run.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"run failed with {completed.returncode}: {' '.join(command)}")


def snapshots(directory: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(directory.glob("nr_minkowski.geometry.*.phdf"))
    if len(paths) != 3:
        raise RuntimeError(f"expected three snapshots in {directory}, found {len(paths)}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    vacuum = workdir / "vacuum"
    zero = workdir / "zero"
    executable = str(args.pangu.resolve())
    input_path = str(args.input.resolve())
    run([executable, "-i", input_path], vacuum)
    run(
        [
            executable,
            "-i",
            input_path,
            "numerical_relativity/matter_source=zero",
            "parthenon/output2/variables=nr.z4c,nr.adm,nr.tmunu",
        ],
        zero,
    )

    maximum = 0.0
    times: list[float] = []
    for vacuum_path, zero_path in zip(snapshots(vacuum), snapshots(zero)):
        with h5py.File(vacuum_path, "r") as lhs, h5py.File(zero_path, "r") as rhs:
            if "nr.tmunu" in lhs:
                raise RuntimeError("vacuum path unexpectedly allocated/output nr.tmunu")
            if "nr.tmunu" not in rhs:
                raise RuntimeError("explicit zero-source path did not output nr.tmunu")
            time_lhs = float(lhs["Info"].attrs["Time"])
            time_rhs = float(rhs["Info"].attrs["Time"])
            if time_lhs != time_rhs:
                raise RuntimeError(f"snapshot time mismatch {time_lhs} != {time_rhs}")
            times.append(time_lhs)
            tmunu = np.asarray(rhs["nr.tmunu"], dtype=np.float64)
            maximum = max(maximum, float(np.max(np.abs(tmunu))))
            for field in ("nr.z4c", "nr.adm"):
                difference = np.asarray(lhs[field], dtype=np.float64) - np.asarray(
                    rhs[field], dtype=np.float64
                )
                maximum = max(maximum, float(np.max(np.abs(difference))))

    if maximum != 0.0:
        raise RuntimeError(f"zero source did not reduce bitwise to vacuum: {maximum:.17e}")
    report = {
        "schema": "pangu.sync0.zero-source.v1",
        "status": "pass",
        "times": times,
        "maximum_absolute_difference": maximum,
        "vacuum_tmunu_allocated": False,
    }
    (workdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"SYNC-0 zero-source PASS: outputs={len(times)} maximum difference={maximum:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
