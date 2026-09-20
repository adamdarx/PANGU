#!/usr/bin/env python3
"""Compare matched SR Hydro/SRMHD evolution snapshots with AthenaK."""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re

import h5py
import numpy as np


FIELDS = {
    "hydro": ("density", "u1", "u2", "u3", "pressure"),
    "mhd": ("density", "u1", "u2", "u3", "internal_energy_density",
            "B1", "B2", "B3"),
}


def pangu_snapshots(directory: pathlib.Path, physics: str) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    for name in sorted(glob.glob(str(directory / "*.phdf"))):
        path = pathlib.Path(name)
        with h5py.File(path, "r") as handle:
            primitive = np.asarray(handle[f"{physics}.prim"])[:, :, 0, 0, :]
            primitive = primitive.transpose(1, 0, 2).reshape(5, -1)
            x = np.asarray(handle["VolumeLocations/x"]).reshape(-1)
            order = np.argsort(x)
            values = primitive[:, order]
            if physics == "mhd":
                magnetic = np.asarray(handle["mhd.b_cell"])[:, :, 0, 0, :]
                magnetic = magnetic.transpose(1, 0, 2).reshape(3, -1)
                values = np.vstack((values, magnetic[:, order]))
            snapshots.append({
                "cycle": int(handle["Info"].attrs["NCycle"]),
                "time": float(handle["Info"].attrs["Time"]),
                "values": values,
                "x": x[order],
                "path": path.name,
            })
    return snapshots


def athenak_snapshots(directory: pathlib.Path, physics: str,
                      gamma: float) -> list[dict[str, object]]:
    history_paths = sorted(directory.glob("*.hst"))
    if len(history_paths) != 1:
        raise RuntimeError(
            f"{directory}: expected exactly one high-precision history file, got "
            f"{len(history_paths)}")
    history_times = np.loadtxt(history_paths[0], comments="#", ndmin=2)[:, 0]
    snapshots: list[dict[str, object]] = []
    for snapshot_index, path in enumerate(sorted((directory / "tab").glob("*.tab"))):
        lines = path.read_text(encoding="utf-8").splitlines()
        metadata = re.search(r"time=([^ ]+)\s+cycle=(\d+)", lines[0])
        if metadata is None:
            raise RuntimeError(f"missing time/cycle metadata in {path}")
        rows = np.asarray([
            [float(value) for value in line.split()[2:]]
            for line in lines if line.strip() and not line.startswith("#")
        ]).T
        # AthenaK writes its primitive energy component as internal-energy
        # density.  PANGU uses pressure for Hydro but now deliberately retains
        # AthenaK's native internal-energy representation for SR/GR MHD.
        primitive_energy = ((gamma - 1.0) * rows[5]
                            if physics == "hydro" else rows[5])
        values = np.vstack((rows[1:5], primitive_energy))
        if physics == "mhd":
            values = np.vstack((values, rows[6:9]))
        snapshots.append({
            "cycle": int(metadata.group(2)),
            # AthenaK's tab header is hard-coded to six decimal places.  Its
            # history output honors data_format, so pair the full-precision
            # time with the tab payload and retain the tab cycle metadata.
            "time": float(history_times[snapshot_index]),
            "values": values,
            "x": rows[0],
            "path": path.name,
        })
    if len(history_times) != len(snapshots):
        raise RuntimeError(
            f"{directory}: history/tab snapshot mismatch "
            f"({len(history_times)} != {len(snapshots)})")
    return snapshots


def compare(case: str, left: list[dict[str, object]],
            right: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(left) < 5 or len(left) != len(right):
        raise RuntimeError(
            f"{case}: expected equal snapshot counts of at least five, got "
            f"{len(left)} and {len(right)}")
    records: list[dict[str, object]] = []
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        difference = np.abs(lhs["values"] - rhs["values"])
        metrics = {}
        for field, delta in zip(FIELDS[case], difference):
            metrics[field] = {
                "l1": float(np.mean(delta)),
                "l2": float(np.sqrt(np.mean(delta * delta))),
                "linf": float(np.max(delta)),
            }
        records.append({
            "case": case,
            "snapshot": index,
            "pangu_file": lhs["path"],
            "athenak_file": rhs["path"],
            "cycle": int(lhs["cycle"]),
            "cycle_delta": abs(int(lhs["cycle"]) - int(rhs["cycle"])),
            "time": float(lhs["time"]),
            "time_delta": abs(float(lhs["time"]) - float(rhs["time"])),
            "x_linf": float(np.max(np.abs(lhs["x"] - rhs["x"]))),
            "maximum_linf": float(np.max(difference)),
            "fields": metrics,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu-hydro", required=True, type=pathlib.Path)
    parser.add_argument("--athenak-hydro", required=True, type=pathlib.Path)
    parser.add_argument("--pangu-mhd", required=True, type=pathlib.Path)
    parser.add_argument("--athenak-mhd", required=True, type=pathlib.Path)
    parser.add_argument("--hydro-gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--mhd-gamma", type=float, default=2.0)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    records = compare(
        "hydro", pangu_snapshots(args.pangu_hydro, "hydro"),
        athenak_snapshots(args.athenak_hydro, "hydro", args.hydro_gamma))
    records += compare(
        "mhd", pangu_snapshots(args.pangu_mhd, "mhd"),
        athenak_snapshots(args.athenak_mhd, "mhd", args.mhd_gamma))
    tolerance = 1.0e-11
    metadata_tolerance = 1.0e-12
    passed = all(
        record["maximum_linf"] <= tolerance
        and record["time_delta"] <= metadata_tolerance
        and record["x_linf"] <= metadata_tolerance
        and record["cycle_delta"] == 0
        for record in records)
    report = {
        "passed": passed,
        "absolute_tolerance": tolerance,
        "metadata_tolerance": metadata_tolerance,
        "snapshot_pairs": len(records),
        "maximum_linf": max(record["maximum_linf"] for record in records),
        "records": records,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    for case in FIELDS:
        selected = [record for record in records if record["case"] == case]
        print(f"SR {case}: snapshots={len(selected)} "
              f"max_Linf={max(record['maximum_linf'] for record in selected):.17e} "
              f"max_dt={max(record['time_delta'] for record in selected):.17e}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
