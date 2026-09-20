#!/usr/bin/env python3
"""Validate SYNC-1 TOV initialization convergence and short-time stability."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


TOV_RADIUS = 9.5856924301473612
CENTRAL_DENSITY = 1.28e-3


def run(executable: Path, input_path: Path, directory: Path, overrides: list[str]) -> None:
    directory.mkdir(parents=True)
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(executable), "-i", str(input_path), "-d", ".", *overrides],
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if "Driver completed." not in (directory / "run.log").read_text(encoding="utf-8"):
        raise RuntimeError(f"TOV run did not complete in {directory}")


def unique(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def cell_radius(stream: h5py.File, block: int) -> np.ndarray:
    centers = []
    for axis in ("x", "y", "z"):
        faces = np.asarray(stream[f"Locations/{axis}"][block], dtype=np.float64)
        centers.append(0.5 * (faces[:-1] + faces[1:]))
    z, y, x = np.meshgrid(centers[2], centers[1], centers[0], indexing="ij")
    return np.sqrt(x * x + y * y + z * z)


def audit(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as stream:
        primitive = np.asarray(stream["hydro.prim"], dtype=np.float64)
        z4c = np.asarray(stream["nr.z4c"], dtype=np.float64)
        constraints = np.asarray(stream["nr.constraints"], dtype=np.float64)
        for name, values in (
            ("hydro.prim", primitive),
            ("nr.z4c", z4c),
            ("nr.constraints", constraints),
        ):
            if not np.isfinite(values).all():
                raise RuntimeError(f"{path}: non-finite {name}")
        gxx, gxy, gxz, gyy, gyz, gzz = [z4c[:, component] for component in range(1, 7)]
        determinant = (
            gxx * (gyy * gzz - gyz * gyz)
            - gxy * (gxy * gzz - gxz * gyz)
            + gxz * (gxy * gyz - gxz * gyy)
        )
        interior_constraints = []
        core_speed = []
        speed = np.sqrt(np.sum(primitive[:, 1:4] ** 2, axis=1))
        for block in range(primitive.shape[0]):
            radius = cell_radius(stream, block)
            interior_constraints.append(
                constraints[block, :, radius < 0.7 * TOV_RADIUS].reshape(-1)
            )
            core_speed.append(speed[block, primitive[block, 0] > 1.0e-5].reshape(-1))
        smooth = np.concatenate(interior_constraints)
        dense_speed = np.concatenate(core_speed)
        return {
            "time": float(stream["Info"].attrs["Time"]),
            "density_max": float(np.max(primitive[:, 0])),
            "pressure_min": float(np.min(primitive[:, 4])),
            "lapse_min": float(np.min(z4c[:, 18])),
            "core_speed_max": float(np.max(dense_speed)),
            "inner_constraint_l2": float(np.sqrt(np.mean(smooth * smooth))),
            "constraint_linf": float(np.max(np.abs(constraints))),
            "conformal_determinant_linf": float(np.max(np.abs(determinant - 1.0))),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()
    executable = args.pangu.resolve()
    input_path = args.input.resolve()
    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    initialization: dict[int, dict[str, float]] = {}
    for resolution in (32, 48, 64):
        block = resolution // 2
        directory = workdir / f"n{resolution}"
        overrides = [
            f"parthenon/mesh/nx1={resolution}",
            f"parthenon/mesh/nx2={resolution}",
            f"parthenon/mesh/nx3={resolution}",
            f"parthenon/meshblock/nx1={block}",
            f"parthenon/meshblock/nx2={block}",
            f"parthenon/meshblock/nx3={block}",
        ]
        run(executable, input_path, directory, overrides)
        initialization[resolution] = audit(unique(directory, "*.00000.phdf"))

    orders = []
    for coarse, fine in ((32, 48), (48, 64)):
        coarse_error = initialization[coarse]["inner_constraint_l2"]
        fine_error = initialization[fine]["inner_constraint_l2"]
        orders.append(math.log(coarse_error / fine_error) / math.log(fine / coarse))
    if min(orders) < 1.8:
        raise RuntimeError(f"TOV smooth-interior constraint convergence is below second order: {orders}")
    if max(item["conformal_determinant_linf"] for item in initialization.values()) > 2.0e-14:
        raise RuntimeError("TOV conformal metric is not unit determinant")
    density_errors = [
        abs(initialization[resolution]["density_max"] - CENTRAL_DENSITY)
        for resolution in (32, 48, 64)
    ]
    if not density_errors[2] < density_errors[1] < density_errors[0]:
        raise RuntimeError(f"TOV sampled central density does not converge: {density_errors}")

    evolution_dir = workdir / "evolution"
    run(
        executable,
        input_path,
        evolution_dir,
        ["parthenon/time/tlim=0.1", "parthenon/time/nlim=4", "parthenon/output1/dt=0.05"],
    )
    initial = audit(unique(evolution_dir, "*.00000.phdf"))
    final = audit(unique(evolution_dir, "*.final.phdf"))
    relative_density_drift = abs(final["density_max"] - initial["density_max"]) / initial[
        "density_max"
    ]
    if final["time"] != 0.1 or relative_density_drift > 2.0e-3:
        raise RuntimeError(
            f"TOV short evolution drift: t={final['time']} density={relative_density_drift}"
        )
    if final["core_speed_max"] > 1.0e-3:
        raise RuntimeError(f"TOV dense core developed excessive velocity {final['core_speed_max']}")
    if final["constraint_linf"] > 1.05 * initial["constraint_linf"]:
        raise RuntimeError("TOV short evolution amplified the maximum constraint")

    report = {
        "schema": "pangu.sync1.tov-regression.v1",
        "pass": True,
        "initialization": {str(key): value for key, value in initialization.items()},
        "observed_constraint_orders": orders,
        "evolution_initial": initial,
        "evolution_final": final,
        "relative_central_density_drift": relative_density_drift,
    }
    (workdir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for path in workdir.rglob("*.phdf"):
        path.unlink()
    for path in workdir.rglob("*.xdmf"):
        path.unlink()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
