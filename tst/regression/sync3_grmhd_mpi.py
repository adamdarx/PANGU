#!/usr/bin/env python3
"""Compare one- and two-rank CUDA-aware MPI SYNC-3 evolution."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


FIELDS = (
    "mhd.cons",
    "mhd.b_face",
    "mhd.prim",
    "mhd.b_cell",
    "mhd.divb",
    "mhd.fofc",
    "mhd.recovery",
    "nr.z4c",
    "nr.adm",
    "nr.tmunu",
    "nr.constraints",
    "nr.constraint_mask",
)


def run(command: list[str], directory: Path) -> None:
    directory.mkdir(parents=True)
    completed = subprocess.run(
        command,
        cwd=directory,
        env=os.environ.copy(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (directory / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or "Driver completed." not in completed.stdout:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-8000:]}"
        )


def latest(directory: Path) -> Path:
    timed: list[tuple[float, Path]] = []
    for path in directory.glob("*.phdf"):
        with h5py.File(path, "r") as stream:
            timed.append((float(stream["Info"].attrs["Time"]), path))
    if not timed:
        raise RuntimeError(f"no PHDF files in {directory}")
    return max(timed)[1]


def logical_fields(path: Path) -> tuple[dict[str, np.ndarray], float, float]:
    with h5py.File(path, "r") as stream:
        keys = [
            (int(stream["Levels"][index]), *map(int, stream["LogicalLocations"][index]))
            for index in range(len(stream["Levels"]))
        ]
        order = np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)
        fields = {name: np.asarray(stream[name], dtype=np.float64)[order] for name in FIELDS}
        time = float(stream["Info"].attrs["Time"])
        maximum_divb = float(np.max(np.abs(stream["mhd.divb"])))
    return fields, time, maximum_divb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--mpi-launcher", default="mpirun")
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    executable = str(args.pangu.resolve())
    input_path = str(args.input.resolve())
    common = [
        executable,
        "-i",
        input_path,
        "parthenon/time/dt_force=0.0375",
        "parthenon/time/tlim=0.15",
        "parthenon/time/nlim=4",
        "parthenon/output1/dt=0.15",
        "parthenon/output2/dt=0.15",
        "parthenon/output3/include_in_final=false",
    ]
    for ranks in (1, 2):
        run(
            [args.mpi_launcher, "--bind-to", "none", "-np", str(ranks), *common],
            workdir / f"rank{ranks}",
        )

    one, one_time, one_divb = logical_fields(latest(workdir / "rank1"))
    two, two_time, two_divb = logical_fields(latest(workdir / "rank2"))
    if one_time != 0.15 or two_time != one_time:
        raise RuntimeError(f"MPI comparison time mismatch: {one_time}, {two_time}")
    differences: dict[str, float] = {}
    for name in FIELDS:
        if one[name].shape != two[name].shape:
            raise RuntimeError(f"MPI decomposition changed {name} shape")
        differences[name] = float(np.max(np.abs(two[name] - one[name])))
    maximum = max(differences.values())
    maximum_divb = max(one_divb, two_divb)
    if maximum > 2.0e-14:
        raise RuntimeError(f"MPI decomposition changed fields: {differences}")
    if maximum_divb > 1.0e-12:
        raise RuntimeError(f"MPI divergence gate failed: {maximum_divb}")
    report = {
        "schema": "pangu.sync3.grmhd-mpi.v1",
        "pass": True,
        "comparison_time": one_time,
        "maximum_absolute_difference": maximum,
        "field_maximum_absolute_difference": differences,
        "maximum_abs_divb": maximum_divb,
    }
    (workdir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
