#!/usr/bin/env python3
"""Validate stage-time updates of the SYNC-0 analytic Tmunu field."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess

import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    completed = subprocess.run(
        [str(args.pangu.resolve()), "-i", str(args.input.resolve())],
        cwd=workdir,
        check=False,
        capture_output=True,
        text=True,
    )
    (workdir / "run.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"PANGU analytic Tmunu run failed with {completed.returncode}")

    files = sorted(workdir.glob("nr_minkowski.sync0.*.phdf"))
    if len(files) < 5:
        raise RuntimeError(f"expected the initial and four evolved outputs, found {len(files)}")

    maximum = 0.0
    times: list[float] = []
    for path in files:
        with h5py.File(path, "r") as handle:
            time = float(handle["Info"].attrs["Time"])
            values = np.asarray(handle["nr.tmunu"], dtype=np.float64)
            x = np.asarray(handle["VolumeLocations/x"], dtype=np.float64)
            y = np.asarray(handle["VolumeLocations/y"], dtype=np.float64)
            z = np.asarray(handle["VolumeLocations/z"], dtype=np.float64)
            if not np.isfinite(np.asarray(handle["nr.z4c"])).all():
                raise RuntimeError(f"non-finite Z4c state in {path.name}")
        phase = (
            1.25 * x[:, None, None, :]
            - 0.75 * y[:, None, :, None]
            + 0.5 * z[:, :, None, None]
            - 1.75 * time
        )
        expected = np.stack(
            [0.0025 * (component + 1) * np.sin(phase + 0.125 * component) for component in range(10)],
            axis=1,
        )
        maximum = max(maximum, float(np.max(np.abs(values - expected))))
        times.append(time)

    tolerance = 128.0 * np.finfo(np.float64).eps
    if not math.isfinite(maximum) or maximum > tolerance:
        raise RuntimeError(f"analytic Tmunu mismatch {maximum:.17e} > {tolerance:.17e}")
    report = {
        "schema": "pangu.sync0.analytic-source.v1",
        "status": "pass",
        "times": times,
        "maximum_absolute_error": maximum,
        "tolerance": tolerance,
    }
    (workdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"SYNC-0 analytic source PASS: outputs={len(times)} including t=0 "
        f"maximum error={maximum:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
