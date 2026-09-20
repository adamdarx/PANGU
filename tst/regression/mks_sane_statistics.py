#!/usr/bin/env python3
"""Compare short MKS SANE trajectories produced by PANGU and KHARMA.

The comparison intentionally uses integral and distributional diagnostics.  The
four-percent pressure perturbation and limiter branches make evolved pointwise
agreement neither expected nor useful once the two integrators take different
time steps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=Path)
    parser.add_argument("--kharma", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spin", type=float, default=0.9375)
    parser.add_argument("--hslope", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=4.0 / 3.0)
    parser.add_argument("--disk-density", type=float, default=1.0e-3)
    parser.add_argument("--time-tolerance", type=float, default=5.0e-2)
    return parser.parse_args()


def snapshots(directory: Path, pattern: str) -> list[tuple[float, Path]]:
    found: list[tuple[float, Path]] = []
    for path in directory.glob(pattern):
        with h5py.File(path, "r") as stream:
            found.append((float(stream["Info"].attrs["Time"]), path))
    return sorted(found, key=lambda item: (item[0], item[1].name))


def mks_geometry(
    stream: h5py.File, spin: float, hslope: float
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    x1 = np.asarray(stream["VolumeLocations/x"], dtype=np.float64)[:, None, None, :]
    x2 = np.asarray(stream["VolumeLocations/y"], dtype=np.float64)[:, None, :, None]
    radius = np.exp(x1)
    theta = np.pi * x2 + 0.5 * (1.0 - hslope) * np.sin(2.0 * np.pi * x2)
    theta_jacobian = np.pi * (1.0 + (1.0 - hslope) * np.cos(2.0 * np.pi * x2))
    sine = np.sin(theta)
    cosine = np.cos(theta)
    sine2 = sine * sine
    rho2 = radius * radius + spin * spin * cosine * cosine
    factor = 2.0 * radius / rho2
    one_plus_factor = 1.0 + factor
    g11 = one_plus_factor * radius * radius
    g13 = -spin * sine2 * one_plus_factor * radius
    g22 = rho2 * theta_jacobian * theta_jacobian
    g33 = sine2 * (rho2 + spin * spin * sine2 * one_plus_factor)
    lapse = np.sqrt(1.0 / one_plus_factor)
    gdet = np.abs(rho2 * sine * radius * theta_jacobian)
    return gdet / lapse, (g11, g13, g22, g33)


def diagnostic(path: Path, code: str, args: argparse.Namespace) -> dict[str, float | int]:
    with h5py.File(path, "r") as stream:
        info = stream["Info"].attrs
        proper_volume, spatial_metric = mks_geometry(stream, args.spin, args.hslope)
        if code == "pangu":
            primitive = np.asarray(stream["mhd.prim"], dtype=np.float64)
            density = primitive[:, 0]
            internal_energy = primitive[:, 4]
            magnetic = np.asarray(stream["mhd.b_cell"], dtype=np.float64)
            repair = np.asarray(stream["mhd.fofc"]) != 0
            floor = np.zeros_like(repair)
            divergence = np.asarray(stream["mhd.divb"], dtype=np.float64)
        else:
            density = np.asarray(stream["prims.rho"], dtype=np.float64)
            internal_energy = np.asarray(stream["prims.u"], dtype=np.float64)
            magnetic = np.asarray(stream["prims.B"], dtype=np.float64)
            repair = np.asarray(stream["fofcflag"]) != 0
            floor = (np.asarray(stream["fflag"]) != 0) | (np.asarray(stream["pflag"]) != 0)
            divergence = np.asarray(stream["divB"], dtype=np.float64)

        g11, g13, g22, g33 = spatial_metric
        magnetic_squared = (
            g11 * magnetic[:, 0] ** 2
            + 2.0 * g13 * magnetic[:, 0] * magnetic[:, 2]
            + g22 * magnetic[:, 1] ** 2
            + g33 * magnetic[:, 2] ** 2
        )
        gas_pressure = (args.gamma - 1.0) * internal_energy
        disk = density > args.disk_density
        beta_mask = disk & (magnetic_squared > 0.0) & np.isfinite(magnetic_squared)
        beta = gas_pressure[beta_mask] / (0.5 * magnetic_squared[beta_mask])
        arrays = (density, internal_energy, magnetic, magnetic_squared, divergence)
        return {
            "time": float(info["Time"]),
            "cycle": int(info["NCycle"]),
            "density_max": float(np.max(density)),
            "pressure_max": float(np.max(gas_pressure)),
            "proper_density_integral": float(np.sum(density * proper_volume)),
            "proper_internal_energy_integral": float(
                np.sum(internal_energy * proper_volume)
            ),
            "eulerian_magnetic_energy": float(
                np.sum(0.5 * magnetic_squared * proper_volume)
            ),
            "disk_beta_median": float(np.median(beta)),
            "max_abs_divb": float(np.max(np.abs(divergence))),
            "repair_cells": int(np.count_nonzero(repair)),
            "repair_cells_in_disk": int(np.count_nonzero(repair & disk)),
            "floor_cells": int(np.count_nonzero(floor)),
            "floor_cells_in_disk": int(np.count_nonzero(floor & disk)),
            "nonfinite_values": int(
                sum(array.size - np.count_nonzero(np.isfinite(array)) for array in arrays)
            ),
        }


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), np.finfo(np.float64).tiny)


def main() -> int:
    args = parse_args()
    pangu = snapshots(args.pangu, "gr_torus_sane.prim.*.phdf")
    kharma = snapshots(args.kharma, "torus.out0.*.phdf")
    if not pangu or len(pangu) != len(kharma):
        raise RuntimeError(
            f"snapshot count mismatch: PANGU={len(pangu)}, KHARMA={len(kharma)}"
        )

    rows: list[dict[str, object]] = []
    compared = (
        "density_max",
        "pressure_max",
        "proper_density_integral",
        "proper_internal_energy_integral",
        "eulerian_magnetic_energy",
        "disk_beta_median",
    )
    for (pangu_time, pangu_path), (kharma_time, kharma_path) in zip(pangu, kharma):
        if abs(pangu_time - kharma_time) > args.time_tolerance:
            raise RuntimeError(
                f"output-time mismatch: PANGU={pangu_time}, KHARMA={kharma_time}"
            )
        left = diagnostic(pangu_path, "pangu", args)
        right = diagnostic(kharma_path, "kharma", args)
        rows.append(
            {
                "pangu": left,
                "kharma": right,
                "relative_difference": {
                    name: relative_difference(float(left[name]), float(right[name]))
                    for name in compared
                },
            }
        )

    report = {
        "schema_version": 1,
        "case": "MKS SANE Fishbone-Moncrief torus",
        "comparison": "short-time integral and distributional diagnostics",
        "parameters": {
            "spin": args.spin,
            "hslope": args.hslope,
            "gamma": args.gamma,
            "disk_density_threshold": args.disk_density,
            "time_tolerance": args.time_tolerance,
        },
        "snapshots": rows,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if any(
        int(row[code]["nonfinite_values"]) != 0
        for row in rows
        for code in ("pangu", "kharma")
    ):
        return 1
    if not all(math.isfinite(value) for row in rows for value in row["relative_difference"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
