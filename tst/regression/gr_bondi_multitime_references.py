#!/usr/bin/env python3
"""Compare five double-precision fixed-GR Bondi snapshots with AthenaK."""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re

import h5py
import numpy as np


FIELDS = ("density", "u1", "u2", "u3", "pressure")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=pathlib.Path)
    parser.add_argument("--athenak", required=True, type=pathlib.Path)
    parser.add_argument("--slice-y", type=float, default=0.625)
    parser.add_argument("--slice-z", type=float, default=0.625)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    pangu_paths = [pathlib.Path(name) for name in sorted(
        glob.glob(str(args.pangu / "*.phdf")))]
    athenak_paths = sorted((args.athenak / "tab").glob("*.tab"))
    histories = sorted(args.athenak.glob("*.hst"))
    if len(histories) != 1:
        raise RuntimeError(f"{args.athenak}: expected exactly one history file")
    history_times = np.loadtxt(histories[0], comments="#", ndmin=2)[:, 0]
    if len(athenak_paths) != len(history_times):
        raise RuntimeError("AthenaK history/table count mismatch")

    reference: dict[int, dict[str, object]] = {}
    gamma = 4.0 / 3.0
    for time, path in zip(history_times, athenak_paths):
        header = path.read_text(encoding="utf-8").splitlines()[0]
        match = re.search(r"cycle=(\d+)", header)
        if match is None:
            raise RuntimeError(f"{path}: missing cycle metadata")
        cycle = int(match.group(1))
        if cycle in reference:
            continue
        rows = np.loadtxt(path, comments="#", ndmin=2)
        values = rows[:, 3:8].T
        values[4] *= gamma - 1.0
        reference[cycle] = {
            "time": float(time), "x": rows[:, 2], "values": values,
            "path": path.name,
        }

    records: list[dict[str, object]] = []
    seen: set[int] = set()
    for path in pangu_paths:
        with h5py.File(path, "r") as handle:
            cycle = int(handle["Info"].attrs["NCycle"])
            if cycle in seen or cycle not in reference:
                continue
            seen.add(cycle)
            time = float(handle["Info"].attrs["Time"])
            x = np.asarray(handle["VolumeLocations/x"])[0]
            y = np.asarray(handle["VolumeLocations/y"])[0]
            z = np.asarray(handle["VolumeLocations/z"])[0]
            j = int(np.argmin(np.abs(y - args.slice_y)))
            k = int(np.argmin(np.abs(z - args.slice_z)))
            values = np.asarray(handle["hydro.prim"])[0, :, k, j, :]
        rhs = reference[cycle]
        indices = np.asarray([int(np.argmin(np.abs(x - coordinate)))
                              for coordinate in rhs["x"]])
        values = values[:, indices]
        delta = np.abs(values - rhs["values"])
        fields = {
            field: {
                "l1": float(np.mean(component)),
                "l2": float(np.sqrt(np.mean(component * component))),
                "linf": float(np.max(component)),
            }
            for field, component in zip(FIELDS, delta)
        }
        records.append({
            "cycle": cycle,
            "time": time,
            "time_delta": abs(time - rhs["time"]),
            "x_linf": float(np.max(np.abs(x[indices] - rhs["x"]))),
            "pangu_file": path.name,
            "athenak_file": rhs["path"],
            "maximum_linf": float(np.max(delta)),
            "fields": fields,
        })

    if len(records) < 5:
        raise RuntimeError(f"expected at least five snapshot pairs, got {len(records)}")
    tolerance = 1.0e-13
    metadata_tolerance = 1.0e-14
    passed = all(
        record["maximum_linf"] <= tolerance
        and record["time_delta"] <= metadata_tolerance
        and record["x_linf"] <= metadata_tolerance
        for record in records)
    report = {
        "passed": passed,
        "case": "fixed-Schwarzschild GR hydrodynamic Bondi accretion",
        "snapshot_pairs": len(records),
        "absolute_tolerance": tolerance,
        "metadata_tolerance": metadata_tolerance,
        "maximum_linf": max(record["maximum_linf"] for record in records),
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"GR Bondi: snapshots={len(records)} passed={passed} "
          f"max_Linf={report['maximum_linf']:.17e}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
