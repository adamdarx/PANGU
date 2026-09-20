#!/usr/bin/env python3
"""Measure one-period SR/fixed-GR Hydro and MHD linear-wave convergence."""

from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib

import h5py
import numpy as np


def snapshot(path: pathlib.Path, physics: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["VolumeLocations/x"]).reshape(-1)
        values = np.asarray(handle[f"{physics}.prim"])[:, :, :, 0, :]
        values = values.transpose(1, 0, 2, 3).reshape(5, -1)
        if physics == "mhd":
            magnetic = np.asarray(handle["mhd.b_cell"])[:, :, :, 0, :]
            magnetic = magnetic.transpose(1, 0, 2, 3).reshape(3, -1)
            values = np.vstack((values, magnetic))
    order = np.argsort(x)
    return x[order], values[:, order]


def error(directory: pathlib.Path, physics: str) -> dict[str, object]:
    initial_paths = [pathlib.Path(name) for name in glob.glob(
        str(directory / "*.00000.phdf"))]
    final_paths = [pathlib.Path(name) for name in glob.glob(
        str(directory / "*.final.phdf"))]
    if len(initial_paths) != 1 or len(final_paths) != 1:
        raise RuntimeError(f"{directory}: expected one initial and one final PHDF")
    initial_x, initial = snapshot(initial_paths[0], physics)
    final_x, final = snapshot(final_paths[0], physics)
    if not np.array_equal(initial_x, final_x):
        raise RuntimeError(f"{directory}: initial/final coordinate mismatch")
    delta = np.abs(final - initial)
    return {
        "directory": str(directory),
        "resolution": int(initial_x.size),
        "l1": float(np.mean(delta)),
        "l2": float(np.sqrt(np.mean(delta * delta))),
        "linf": float(np.max(delta)),
    }


def series(name: str, directories: list[pathlib.Path]) -> dict[str, object]:
    physics = "mhd" if name.endswith("mhd") else "hydro"
    errors = [error(directory, physics) for directory in directories]
    errors.sort(key=lambda record: record["resolution"])
    orders = [
        math.log(coarse["l1"] / fine["l1"]) /
        math.log(fine["resolution"] / coarse["resolution"])
        for coarse, fine in zip(errors[:-1], errors[1:])
    ]
    return {
        "case": name,
        "errors": errors,
        "orders_l1": orders,
        "minimum_order_l1": min(orders),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("sr-hydro", "gr-hydro", "sr-mhd", "gr-mhd"):
        parser.add_argument(f"--{name}", nargs=3, required=True,
                            type=pathlib.Path, metavar=("N32", "N64", "N128"))
    parser.add_argument("--minimum-order", type=float, default=1.8)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    records = [
        series("sr_hydro", args.sr_hydro),
        series("gr_hydro", args.gr_hydro),
        series("sr_mhd", args.sr_mhd),
        series("gr_mhd", args.gr_mhd),
    ]
    passed = all(record["minimum_order_l1"] >= args.minimum_order
                 for record in records)
    report = {
        "passed": passed,
        "minimum_required_order_l1": args.minimum_order,
        "minimum_observed_order_l1": min(
            record["minimum_order_l1"] for record in records),
        "series": records,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    for record in records:
        print(f"{record['case']}: minimum L1 order="
              f"{record['minimum_order_l1']:.12f}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
