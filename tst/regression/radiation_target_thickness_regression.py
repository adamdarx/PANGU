#!/usr/bin/env python3
"""Run the RC-4 MKS target-thickness convergence and parameter gates."""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import shutil
import subprocess
import tempfile

import h5py
import numpy as np


SPIN = 0.9375
HSLOPE = 0.3
TARGET_HEIGHT = 0.1


def run_case(
    executable: pathlib.Path,
    input_path: pathlib.Path,
    directory: pathlib.Path,
    nx1: int,
    final_time: float,
    beta_cool: float,
) -> None:
    nx2 = nx1 // 2
    command = [
        str(executable.resolve()),
        "-i",
        str(input_path.resolve()),
        f"parthenon/mesh/nx1={nx1}",
        f"parthenon/mesh/nx2={nx2}",
        f"parthenon/meshblock/nx1={nx1}",
        f"parthenon/meshblock/nx2={nx2}",
        f"parthenon/time/tlim={final_time:.17g}",
        f"radiation/beta_cool={beta_cool:.17g}",
        f"parthenon/output2/dt={final_time:.17g}",
    ]
    completed = subprocess.run(
        command,
        cwd=directory,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (directory / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"RC-4 run failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-8000:]}"
        )


def final_snapshot(directory: pathlib.Path) -> pathlib.Path:
    matches = sorted(directory.glob("*.prim.final.phdf"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one final snapshot in {directory}, found {matches}")
    return matches[0]


def disk_height(path: pathlib.Path) -> float:
    weighted_angle2 = 0.0
    mass = 0.0
    with h5py.File(path, "r") as stream:
        primitive = np.asarray(stream["mhd.prim"], dtype=np.float64)
        if not np.isfinite(primitive).all():
            raise RuntimeError(f"non-finite primitive state in {path}")
        for block in range(primitive.shape[0]):
            density = primitive[block, 0, 0]
            v1, v2, v3 = primitive[block, 1:4, 0]
            x1 = np.asarray(stream["VolumeLocations/x"][block], dtype=np.float64)
            x2 = np.asarray(stream["VolumeLocations/y"][block], dtype=np.float64)
            xx1, xx2 = np.meshgrid(x1, x2)
            radius = np.exp(xx1)
            theta = math.pi * xx2 + 0.5 * (1.0 - HSLOPE) * np.sin(2.0 * math.pi * xx2)
            polar_jacobian = math.pi * (
                1.0 + (1.0 - HSLOPE) * np.cos(2.0 * math.pi * xx2)
            )
            sine = np.sin(theta)
            cosine = np.cos(theta)
            sigma = radius * radius + SPIN * SPIN * cosine * cosine
            factor = 2.0 * radius / sigma
            g11 = (1.0 + factor) * radius * radius
            g13 = -SPIN * (1.0 + factor) * radius * sine * sine
            g22 = sigma * polar_jacobian * polar_jacobian
            g33 = sine * sine * (
                sigma + SPIN * SPIN * sine * sine * (1.0 + factor)
            )
            spatial_norm = (
                g11 * v1 * v1
                + g22 * v2 * v2
                + g33 * v3 * v3
                + 2.0 * g13 * v1 * v3
            )
            lapse = 1.0 / np.sqrt(1.0 + factor)
            u0 = np.sqrt(np.maximum(1.0 + spatial_norm, 1.0)) / lapse
            weight = density * u0 * sigma * np.abs(sine) * radius * polar_jacobian
            region = (
                (radius >= 4.0)
                & (radius <= 30.0)
                & (np.abs(theta - 0.5 * math.pi) <= math.pi / 3.0)
            )
            mass += float(np.sum(weight[region]))
            weighted_angle2 += float(
                np.sum(weight[region] * (theta[region] - 0.5 * math.pi) ** 2)
            )
    if not mass > 0.0:
        raise RuntimeError(f"empty disk-height diagnostic region in {path}")
    return math.sqrt(weighted_angle2 / mass)


def history(directory: pathlib.Path) -> dict[str, np.ndarray]:
    matches = sorted(directory.glob("*.hst"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one history file in {directory}, found {matches}")
    names: dict[int, str] = {}
    rows: list[list[float]] = []
    for line in matches[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            for index, name in re.findall(r"\[(\d+)\]=([^\s]+)", line):
                names[int(index) - 1] = name
        elif line.strip():
            rows.append([float(value) for value in line.split()])
    values = np.asarray(rows)
    return {name: values[:, index] for index, name in names.items()}


def validate_resolution(directory: pathlib.Path) -> tuple[float, float]:
    record = history(directory)
    required = (
        "mhd_energy",
        "mhd_max_abs_divb",
        "radiation_removed_energy",
        "radiation_boundary_energy",
    )
    if any(name not in record for name in required):
        raise RuntimeError(f"incomplete RC-4 history columns in {directory}")
    removed = record["radiation_removed_energy"]
    if np.any(np.diff(removed) < -1.0e-12):
        raise RuntimeError(f"non-monotone removed energy in {directory}")
    delta_energy = record["mhd_energy"][-1] - record["mhd_energy"][0]
    delta_removed = removed[-1] - removed[0]
    delta_boundary = (
        record["radiation_boundary_energy"][-1]
        - record["radiation_boundary_energy"][0]
    )
    residual = delta_energy - delta_removed + delta_boundary
    scale = max(abs(delta_energy), abs(delta_removed), abs(delta_boundary), 1.0)
    relative_closure = abs(residual) / scale
    if float(np.max(record["mhd_max_abs_divb"])) > 1.0e-12:
        raise RuntimeError(f"divergence gate failed in {directory}")
    return disk_height(final_snapshot(directory)), relative_closure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    case = pathlib.Path(tempfile.mkdtemp(prefix="radiation-rc4-", dir=args.workdir))
    try:
        heights: list[float] = []
        closures: list[float] = []
        for nx1 in (32, 64, 128):
            directory = case / f"resolution_{nx1}"
            directory.mkdir()
            run_case(args.executable, args.input, directory, nx1, 200.0, 0.2 * math.pi)
            height, closure = validate_resolution(directory)
            heights.append(height)
            closures.append(closure)

        if abs(heights[1] - TARGET_HEIGHT) / TARGET_HEIGHT > 0.12:
            raise RuntimeError(f"medium-resolution H/R={heights[1]:.8e} misses the target")
        if abs(heights[2] - TARGET_HEIGHT) / TARGET_HEIGHT > 0.12:
            raise RuntimeError(f"high-resolution H/R={heights[2]:.8e} misses the target")
        if abs(heights[2] - heights[1]) > 0.01:
            raise RuntimeError(f"medium/high disk heights do not converge: {heights[1:]}")
        if not (closures[0] > 5.0 * closures[1] > 25.0 * closures[2]):
            raise RuntimeError(f"energy residual does not converge strongly: {closures}")
        if closures[2] > 1.0e-5:
            raise RuntimeError(f"high-resolution energy closure={closures[2]:.8e}")

        beta_heights: list[float] = []
        beta_removed: list[float] = []
        for label, beta in (("fast", 0.2 * math.pi), ("medium", 2.0 * math.pi),
                            ("slow", 20.0 * math.pi)):
            directory = case / f"beta_{label}"
            directory.mkdir()
            run_case(args.executable, args.input, directory, 64, 100.0, beta)
            beta_heights.append(disk_height(final_snapshot(directory)))
            record = history(directory)
            beta_removed.append(float(record["radiation_removed_energy"][-1]))
        if not (beta_heights[0] < beta_heights[1] < beta_heights[2]):
            raise RuntimeError(f"disk thickness is not monotone in beta_cool: {beta_heights}")
        if not (beta_removed[0] > beta_removed[1] > beta_removed[2]):
            raise RuntimeError(f"removed energy is not monotone in beta_cool: {beta_removed}")

        print(
            "PANGU radiation RC-4 PASS: "
            f"H/R={heights} closure={closures} "
            f"beta_H/R={beta_heights} beta_removed={beta_removed}"
        )
        if args.keep:
            print(f"RC-4 artifacts: {case}")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(case, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
