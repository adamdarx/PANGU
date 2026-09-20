#!/usr/bin/env python3
"""Measure short-time Sommerfeld/outflow convergence on uniform GPU grids."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import h5py
import numpy as np


def metric(path: pathlib.Path, amplitude: float = 1.0e-6) -> dict[str, float | int]:
    with h5py.File(path, "r") as data:
        info = data["Info"].attrs
        time = float(info["Time"])
        root = np.asarray(info["RootGridDomain"])
        x = np.asarray(data["VolumeLocations"]["x"])
        y = np.asarray(data["VolumeLocations"]["y"])
        z = np.asarray(data["VolumeLocations"]["z"])
        z4c = np.asarray(data["nr.z4c"])
        constraints = np.asarray(data["nr.constraints"])
        num_blocks = int(info["NumMeshBlocks"])

    x_min, x_max = float(root[0]), float(root[1])
    # Parthenon's RootGridDomain stores (xmin, xmax, ratio) for x, then y and z.
    y_min, y_max = float(root[3]), float(root[4])
    z_min, z_max = float(root[6]), float(root[7])
    kx1 = 1.0 / (x_max - x_min)
    kx2 = 1.0 / (y_max - y_min)
    kx3 = 0.0 / (z_max - z_min)
    wave_number = np.sqrt(kx1 * kx1 + kx2 * kx2 + kx3 * kx3)
    theta = np.arctan2(np.sqrt(kx2 * kx2 + kx1 * kx1), kx3)
    phi = np.arctan2(kx1, kx2)
    weights = np.array([
        -np.cos(theta) ** 2 * np.cos(2.0 * phi)
        - np.cos(phi) ** 2 * np.sin(theta) ** 2,
        -0.25 * (3.0 + np.cos(2.0 * theta)) * np.sin(2.0 * phi),
        -np.cos(theta) * np.sin(theta) * np.sin(phi),
        np.cos(theta) ** 2 * np.cos(2.0 * phi)
        - np.sin(theta) ** 2 * np.sin(phi) ** 2,
        np.cos(theta) * np.sin(theta) * np.cos(phi),
        np.sin(theta) ** 2,
    ])
    phase = 2.0 * np.pi * (
        kx1 * x[:, None, None, :] + kx2 * y[:, None, :, None]
        + kx3 * z[:, :, None, None] - wave_number * time
    )
    expected = np.empty((6,) + phase.shape)
    for component in range(6):
        diagonal = component in (0, 3, 5)
        expected[component] = (1.0 if diagonal else 0.0) + weights[component] * amplitude * np.sin(phase)
    observed = z4c[:, 1:7]
    field_error = np.abs(observed - expected.transpose(1, 0, 2, 3, 4))
    return {
        "time": time,
        "blocks": num_blocks,
        "phase_linf": float(np.max(field_error)),
        "phase_l1": float(np.mean(field_error)),
        "constraint_linf": float(np.max(np.abs(constraints))),
        "constraint_l1": float(np.mean(np.abs(constraints))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    input_file = args.input.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    resolutions = (16, 32, 64)
    results: dict[str, dict[str, float | int]] = {}
    for resolution in resolutions:
        run_dir = args.workdir / f"n{resolution}"
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(executable), "-i", str(input_file),
            f"parthenon/mesh/nx1={resolution}",
            f"parthenon/mesh/nx2={resolution}",
            f"parthenon/mesh/nx3={resolution}",
        ]
        completed = subprocess.run(
            command, cwd=run_dir, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, timeout=300,
        )
        (run_dir / "run.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"outflow convergence run N={resolution} failed\n{completed.stdout}")
        snapshots = sorted(run_dir.glob("*.phdf"))
        if not snapshots:
            raise RuntimeError(f"outflow convergence run N={resolution} wrote no PHDF")
        results[str(resolution)] = metric(snapshots[-1])

    linf_errors = [results[str(n)]["phase_linf"] for n in resolutions]
    l1_errors = [results[str(n)]["phase_l1"] for n in resolutions]
    linf_orders = [float(np.log(linf_errors[i] / linf_errors[i + 1]) / np.log(2.0))
                   for i in range(len(linf_errors) - 1)]
    l1_orders = [float(np.log(l1_errors[i] / l1_errors[i + 1]) / np.log(2.0))
                 for i in range(len(l1_errors) - 1)]
    if not all(order > 0.5 for order in l1_orders):
        raise RuntimeError(f"bulk phase L1 did not converge at second order: {l1_orders}")
    if max(linf_errors) > 0.2 * 1.0e-6:
        raise RuntimeError(f"Sommerfeld boundary error is unbounded: {linf_errors}")
    if max(results[str(n)]["constraint_linf"] for n in resolutions) > 1.0e-2:
        raise RuntimeError("Sommerfeld constraint norm exceeded the short-time stability gate")
    summary = {
        "resolutions": list(resolutions), "results": results,
        "phase_linf_orders": linf_orders, "phase_l1_orders": l1_orders,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "NR-4 Sommerfeld stability/convergence PASS: "
        f"phase_L1={','.join(f'{e:.6e}' for e in l1_errors)} "
        f"L1_orders={','.join(f'{p:.3f}' for p in l1_orders)} "
        f"boundary_Linf={','.join(f'{e:.6e}' for e in linf_errors)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
