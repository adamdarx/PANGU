#!/usr/bin/env python3
"""Compare seed-independent robust-stability growth envelopes with AthenaK."""

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


def pangu_curve(directory: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    samples: dict[float, float] = {}
    target = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    for path in directory.glob("*.phdf"):
        with h5py.File(path, "r") as data:
            time = float(data["Info"].attrs["Time"])
            metric = np.asarray(data["nr.z4c"][:, 1:7])
        component_l1 = np.mean(
            np.abs(metric - target.reshape(1, 6, 1, 1, 1)), axis=(0, 2, 3, 4)
        )
        samples[time] = float(np.sqrt(np.sum(component_l1 * component_l1)))
    times = np.asarray(sorted(samples))
    return times, np.asarray([samples[time] for time in times])


def athenak_curve(directory: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    samples: dict[float, float] = {}
    target = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    for path in (directory / "tab").glob("*.tab"):
        header = path.read_text(encoding="utf-8").splitlines()[0]
        match = re.search(r"time=([0-9.eE+-]+)", header)
        if match is None:
            raise RuntimeError(f"cannot parse AthenaK tab time: {header}")
        state = np.loadtxt(path, comments="#", ndmin=2)[:, 3:25]
        component_l1 = np.mean(np.abs(state[:, 1:7] - target.reshape(1, 6)), axis=0)
        samples[float(match.group(1))] = float(np.sqrt(np.sum(component_l1 * component_l1)))
    times = np.asarray(sorted(samples))
    return times, np.asarray([samples[time] for time in times])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", type=pathlib.Path, required=True)
    parser.add_argument("--athenak", type=pathlib.Path, required=True)
    parser.add_argument("--pangu-input", type=pathlib.Path, required=True)
    parser.add_argument("--athenak-input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    reset(args.workdir)
    pangu_dir = args.workdir / "pangu"
    athenak_dir = args.workdir / "athenak"
    pangu_dir.mkdir()
    athenak_dir.mkdir()
    execute([str(args.pangu), "-i", str(args.pangu_input), "parthenon/output1/dt=-1"], pangu_dir)
    execute([str(args.athenak), "-i", str(args.athenak_input)], athenak_dir)
    pangu_time, pangu_error = pangu_curve(pangu_dir)
    athenak_time, athenak_error = athenak_curve(athenak_dir)
    if pangu_time.shape != athenak_time.shape or not np.allclose(
        pangu_time, athenak_time, rtol=0.0, atol=1.01e-2
    ):
        raise RuntimeError(f"robust-stability output times differ: {pangu_time} versus {athenak_time}")
    if not np.all(np.isfinite(pangu_error)) or not np.all(np.isfinite(athenak_error)):
        raise RuntimeError("robust-stability curve contains a non-finite value")
    if np.any(pangu_error <= 0.0) or np.any(athenak_error <= 0.0):
        raise RuntimeError("robust-stability curve contains a non-positive error norm")
    pangu_growth = pangu_error / pangu_error[0]
    athenak_growth = athenak_error / athenak_error[0]
    maximum_growth = max(float(np.max(pangu_growth)), float(np.max(athenak_growth)))
    final_growth_ratio = float(pangu_growth[-1] / athenak_growth[-1])
    if maximum_growth > 100.0:
        raise RuntimeError(f"robust-stability perturbation grew too rapidly: {maximum_growth}")
    if not 0.1 <= final_growth_ratio <= 10.0:
        raise RuntimeError(
            f"PANGU/AthenaK normalized final growth differs by more than one decade: "
            f"{final_growth_ratio}"
        )
    print(
        "NR-2 AthenaK robust PASS: "
        f"samples={len(pangu_time)} pangu_initial={pangu_error[0]} "
        f"pangu_final={pangu_error[-1]} athenak_initial={athenak_error[0]} "
        f"athenak_final={athenak_error[-1]} max_growth={maximum_growth} "
        f"final_growth_ratio={final_growth_ratio}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
