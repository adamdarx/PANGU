#!/usr/bin/env python3
"""Compare five double-precision fixed-GR monopole snapshots with AthenaK."""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re

import h5py
import numpy as np


FIELDS = ("density", "u1", "u2", "u3", "internal_energy_density",
          "B1", "B2", "B3")


def pangu_snapshots(directory: pathlib.Path, y_slice: float,
                    z_slice: float) -> dict[int, dict[str, object]]:
    paths = [pathlib.Path(name) for name in sorted(
        glob.glob(str(directory / "*.phdf")))]
    snapshots: dict[int, dict[str, object]] = {}
    for path in paths:
        with h5py.File(path, "r") as handle:
            cycle = int(handle["Info"].attrs["NCycle"])
            if cycle in snapshots:
                continue
            x = np.asarray(handle["VolumeLocations/x"])[0]
            y = np.asarray(handle["VolumeLocations/y"])[0]
            z = np.asarray(handle["VolumeLocations/z"])[0]
            j = int(np.argmin(np.abs(y - y_slice)))
            k = int(np.argmin(np.abs(z - z_slice)))
            values = np.concatenate((
                np.asarray(handle["mhd.prim"])[0, :, k, j, :],
                np.asarray(handle["mhd.b_cell"])[0, :, k, j, :]), axis=0)
            snapshots[cycle] = {
                "cycle": cycle,
                "time": float(handle["Info"].attrs["Time"]),
                "x": x,
                "y": float(y[j]),
                "z": float(z[k]),
                "values": values,
                "path": path.name,
            }
    return snapshots


def athenak_snapshots(directory: pathlib.Path) -> dict[int, dict[str, object]]:
    histories = sorted(directory.glob("*.hst"))
    if len(histories) != 1:
        raise RuntimeError(f"{directory}: expected exactly one history file")
    times = np.loadtxt(histories[0], comments="#", ndmin=2)[:, 0]
    snapshots: dict[int, dict[str, object]] = {}
    paths = sorted((directory / "tab").glob("*.tab"))
    if len(paths) != len(times):
        raise RuntimeError(f"{directory}: history/table count mismatch")
    for time, path in zip(times, paths):
        header = path.read_text(encoding="utf-8").splitlines()[0]
        metadata = re.search(r"cycle=(\d+)", header)
        if metadata is None:
            raise RuntimeError(f"{path}: missing cycle metadata")
        cycle = int(metadata.group(1))
        if cycle in snapshots:
            continue
        rows = np.loadtxt(path, comments="#", ndmin=2)
        snapshots[cycle] = {
            "cycle": cycle,
            "time": float(time),
            "x": rows[:, 2],
            "values": rows[:, 3:11].T,
            "path": path.name,
        }
    return snapshots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=pathlib.Path)
    parser.add_argument("--athenak", required=True, type=pathlib.Path)
    parser.add_argument("--slice-y", type=float, default=-0.9375)
    parser.add_argument("--slice-z", type=float, default=0.3125)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    pangu = pangu_snapshots(args.pangu, args.slice_y, args.slice_z)
    athenak = athenak_snapshots(args.athenak)
    cycles = sorted(set(pangu) & set(athenak))
    if len(cycles) < 5:
        raise RuntimeError(f"expected at least five common cycles, got {cycles}")

    records: list[dict[str, object]] = []
    for cycle in cycles:
        lhs = pangu[cycle]
        rhs = athenak[cycle]
        indices = np.asarray([
            int(np.argmin(np.abs(lhs["x"] - coordinate)))
            for coordinate in rhs["x"]
        ])
        x = lhs["x"][indices]
        values = lhs["values"][:, indices]
        delta = np.abs(values - rhs["values"])
        unequal = values.view(np.uint64) != rhs["values"].view(np.uint64)
        fields = {
            field: {
                "l1": float(np.mean(component)),
                "l2": float(np.sqrt(np.mean(component * component))),
                "linf": float(np.max(component)),
                "bitwise_unequal": int(np.count_nonzero(bitwise)),
            }
            for field, component, bitwise in zip(FIELDS, delta, unequal)
        }
        records.append({
            "cycle": cycle,
            "time": lhs["time"],
            "time_delta": abs(lhs["time"] - rhs["time"]),
            "x_linf": float(np.max(np.abs(x - rhs["x"]))),
            "y_delta": abs(lhs["y"] - args.slice_y),
            "z_delta": abs(lhs["z"] - args.slice_z),
            "pangu_file": lhs["path"],
            "athenak_file": rhs["path"],
            "maximum_linf": float(np.max(delta)),
            "bitwise_unequal": int(np.count_nonzero(unequal)),
            "fields": fields,
        })

    tolerance = 1.0e-13
    metadata_tolerance = 1.0e-14
    passed = all(
        record["maximum_linf"] <= tolerance
        and record["time_delta"] <= metadata_tolerance
        and record["x_linf"] <= metadata_tolerance
        and record["y_delta"] <= metadata_tolerance
        and record["z_delta"] <= metadata_tolerance
        for record in records)
    report = {
        "passed": passed,
        "case": "fixed-Kerr GRMHD split monopole",
        "absolute_tolerance": tolerance,
        "metadata_tolerance": metadata_tolerance,
        "snapshot_pairs": len(records),
        "maximum_linf": max(record["maximum_linf"] for record in records),
        "bitwise_unequal": sum(record["bitwise_unequal"] for record in records),
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"GR monopole: snapshots={len(records)} passed={passed} "
          f"max_Linf={report['maximum_linf']:.17e} "
          f"bitwise_unequal={report['bitwise_unequal']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
