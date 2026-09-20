#!/usr/bin/env python3
"""Compare PANGU and AthenaK vacuum-Z4c fields at every RK cycle."""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import re
import shutil
import subprocess

import h5py
import numpy as np


def reset(path: pathlib.Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def execute(command: list[str], cwd: pathlib.Path) -> None:
    environment = dict(os.environ)
    environment["CUDA_LAUNCH_BLOCKING"] = "1"
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")


def pangu_snapshots(directory: pathlib.Path) -> dict[int, tuple[float, np.ndarray]]:
    snapshots: dict[int, tuple[float, np.ndarray]] = {}
    for path in directory.glob("*.phdf"):
        with h5py.File(path, "r") as data:
            cycle = int(data["Info"].attrs["NCycle"])
            time = float(data["Info"].attrs["Time"])
            field = np.asarray(data["nr.z4c"][0])
        snapshots[cycle] = (time, field)
    return snapshots


def athenak_snapshots(directory: pathlib.Path) -> dict[int, tuple[float, np.ndarray]]:
    snapshots: dict[int, tuple[float, np.ndarray]] = {}
    for path in (directory / "tab").glob("*.tab"):
        header = path.read_text(encoding="utf-8").splitlines()[0]
        cycle_match = re.search(r"cycle=(\d+)", header)
        time_match = re.search(r"time=([0-9.eE+-]+)", header)
        if cycle_match is None or time_match is None:
            raise RuntimeError(f"cannot parse AthenaK tab header: {header}")
        cycle = int(cycle_match.group(1))
        time = float(time_match.group(1))
        values = np.loadtxt(path, comments="#")[:, 3:25]
        snapshots[cycle] = (time, values)
    return snapshots


def compare_case(
    name: str,
    pangu: pathlib.Path,
    athenak: pathlib.Path,
    pangu_input: pathlib.Path,
    athenak_input: pathlib.Path,
    workdir: pathlib.Path,
    resolution: int,
    transverse: int,
    integrator: str,
) -> tuple[float, int, int, float, float]:
    pangu_dir = workdir / f"pangu-{name}"
    athenak_dir = workdir / f"athenak-{name}"
    reset(pangu_dir)
    reset(athenak_dir)
    execute(
        [
            str(pangu),
            "-i",
            str(pangu_input),
            f"parthenon/time/integrator={integrator}",
            f"parthenon/mesh/nx1={resolution}",
            f"parthenon/mesh/nx2={transverse}",
            f"parthenon/mesh/nx3={transverse}",
            f"parthenon/meshblock/nx1={resolution}",
            f"parthenon/meshblock/nx2={transverse}",
            f"parthenon/meshblock/nx3={transverse}",
            "parthenon/mesh/nghost=2",
            "numerical_relativity/finite_difference_order=2",
            "parthenon/output1/dt=-1",
            "parthenon/output2/dt=-1",
            "parthenon/output2/dn=1",
        ],
        pangu_dir,
    )
    execute([str(athenak), "-i", str(athenak_input)], athenak_dir)
    measured = pangu_snapshots(pangu_dir)
    reference = athenak_snapshots(athenak_dir)
    if measured.keys() != reference.keys():
        raise RuntimeError(
            f"{name}: cycle sets differ: PANGU={sorted(measured)} AthenaK={sorted(reference)}"
        )
    maximum = 0.0
    maximum_cycle = -1
    initial = math.inf
    first_cycle = math.inf
    for cycle in sorted(measured):
        pangu_time, field = measured[cycle]
        athenak_time, line = reference[cycle]
        # AthenaK prints header times with six decimal digits. Cycle identity is
        # exact; the relaxed time check only accounts for that text formatting.
        if abs(pangu_time - athenak_time) > 5.1e-7:
            raise RuntimeError(
                f"{name}: time differs at cycle {cycle}: {pangu_time} versus {athenak_time}"
            )
        if field.shape != (22, transverse, transverse, resolution):
            raise RuntimeError(f"{name}: unexpected PANGU field shape {field.shape}")
        pangu_line = field[:, transverse // 2, transverse // 2, :].T
        transverse_error = max(
            float(np.max(np.abs(field - field[:, :1, :1, :]))),
            0.0,
        )
        if transverse_error > 64.0 * np.finfo(np.float64).eps:
            raise RuntimeError(f"{name}: transverse invariance failed: {transverse_error}")
        difference = np.abs(pangu_line - line)
        scaled = difference / np.maximum(1.0, np.maximum(np.abs(pangu_line), np.abs(line)))
        local = float(np.max(scaled))
        if local > maximum:
            maximum = local
            maximum_cycle = cycle
        if cycle == 0:
            initial = local
        elif cycle == 1:
            first_cycle = local
    return maximum, maximum_cycle, len(measured), initial, first_cycle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", type=pathlib.Path, required=True)
    parser.add_argument("--athenak-linear", type=pathlib.Path, required=True)
    parser.add_argument("--athenak-gauge", type=pathlib.Path, required=True)
    parser.add_argument("--pangu-linear-input", type=pathlib.Path, required=True)
    parser.add_argument("--athenak-linear-input", type=pathlib.Path, required=True)
    parser.add_argument("--pangu-gauge-input", type=pathlib.Path, required=True)
    parser.add_argument("--athenak-gauge-input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    reset(args.workdir)

    linear = compare_case(
        "linear",
        args.pangu,
        args.athenak_linear,
        args.pangu_linear_input,
        args.athenak_linear_input,
        args.workdir,
        resolution=16,
        transverse=16,
        integrator="rk2",
    )
    gauge = compare_case(
        "gauge",
        args.pangu,
        args.athenak_gauge,
        args.pangu_gauge_input,
        args.athenak_gauge_input,
        args.workdir,
        resolution=32,
        transverse=4,
        integrator="rk4",
    )
    initial_tolerance = 128.0 * np.finfo(np.float64).eps
    evolution_tolerance = 1.0e-12
    for name, result in (("linear", linear), ("gauge", gauge)):
        if not math.isfinite(result[3]) or result[3] > initial_tolerance:
            raise RuntimeError(
                f"{name}: initial scaled PANGU-AthenaK difference {result[3]} "
                f"exceeds machine-roundoff bound {initial_tolerance}"
            )
        if not math.isfinite(result[0]) or result[0] > evolution_tolerance:
            raise RuntimeError(
                f"{name}: maximum scaled PANGU-AthenaK difference {result[0]} "
                f"at cycle {result[1]} exceeds short-evolution bound {evolution_tolerance}"
            )
    print(
        "NR-2 AthenaK multitime PASS: "
        f"linear_initial={linear[3]} linear_first={linear[4]} "
        f"linear_max={linear[0]} cycles={linear[2]}; "
        f"gauge_initial={gauge[3]} gauge_first={gauge[4]} "
        f"gauge_max={gauge[0]} cycles={gauge[2]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
