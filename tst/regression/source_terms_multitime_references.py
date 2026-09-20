#!/usr/bin/env python3
"""Strict multi-time source-term comparisons against AthenaK table output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np


FIELDS = ("density", "velocity1", "velocity2", "velocity3", "pressure")


def _pangu(directory: Path) -> list[dict[str, object]]:
    result = []
    for path in directory.glob("*.phdf"):
        with h5py.File(path, "r") as handle:
            result.append({
                "path": path.name,
                "time": float(handle["Info"].attrs["Time"]),
                "cycle": int(handle["Info"].attrs["NCycle"]),
                "x": np.asarray(handle["VolumeLocations/x"]).ravel(),
                "values": np.asarray(handle["hydro.prim"])[0, :, 0, 0, :].T,
            })
    return sorted(result, key=lambda item: (item["cycle"], item["time"]))


def _athenak(directory: Path, gamma: float) -> list[dict[str, object]]:
    result = []
    for path in (directory / "tab").glob("*.tab"):
        lines = path.read_text(encoding="utf-8").splitlines()
        metadata = re.search(r"time=([^ ]+)\s+cycle=(\d+)", lines[0])
        if metadata is None:
            raise RuntimeError(f"{path}: time/cycle metadata missing")
        table = np.loadtxt(path, comments="#", ndmin=2)
        order = np.argsort(table[:, 2])
        table = table[order]
        values = table[:, 3:8].copy()
        values[:, 4] *= gamma - 1.0
        result.append({
            "path": path.name,
            "time": float(metadata.group(1)),
            "cycle": int(metadata.group(2)),
            "x": table[:, 2],
            "values": values,
        })
    return sorted(result, key=lambda item: (item["cycle"], item["time"]))


def _compare(case: str, pangu_dir: Path, athenak_dir: Path,
             gamma: float) -> dict[str, object]:
    pangu = _pangu(pangu_dir)
    athenak = _athenak(athenak_dir, gamma)
    if len(pangu) < 5 or len(pangu) != len(athenak):
        raise RuntimeError(
            f"{case}: equal snapshot counts >=5 required, got {len(pangu)} and {len(athenak)}")
    records = []
    for index, (left, right) in enumerate(zip(pangu, athenak)):
        delta = np.abs(left["values"] - right["values"])
        records.append({
            "snapshot": index,
            "pangu_file": left["path"],
            "athenak_file": right["path"],
            "cycle": left["cycle"],
            "cycle_delta": abs(int(left["cycle"]) - int(right["cycle"])),
            "time": left["time"],
            "time_delta": abs(float(left["time"]) - float(right["time"])),
            "coordinate_linf": float(np.max(np.abs(left["x"] - right["x"]))),
            "maximum_linf": float(np.max(delta)),
            "fields": {
                field: {
                    "l1": float(np.mean(values)),
                    "l2": float(np.sqrt(np.mean(values * values))),
                    "linf": float(np.max(values)),
                }
                for field, values in zip(FIELDS, delta.T)
            },
        })
    return {
        "snapshot_pairs": len(records),
        "maximum_linf": max(float(record["maximum_linf"]) for record in records),
        "maximum_time_delta": max(float(record["time_delta"]) for record in records),
        "maximum_coordinate_linf": max(
            float(record["coordinate_linf"]) for record in records),
        "maximum_cycle_delta": max(int(record["cycle_delta"]) for record in records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu-acceleration", required=True, type=Path)
    parser.add_argument("--athenak-acceleration", required=True, type=Path)
    parser.add_argument("--pangu-ism", required=True, type=Path)
    parser.add_argument("--athenak-ism", required=True, type=Path)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report: dict[str, object] = {
        "absolute_tolerance": 1.0e-12,
        # AthenaK formatted-table headers store time with six digits.
        "formatted_time_tolerance": 5.0e-9,
        "constant_acceleration": _compare(
            "constant_acceleration", args.pangu_acceleration,
            args.athenak_acceleration, args.gamma),
        "ism_cooling": _compare(
            "ism_cooling", args.pangu_ism, args.athenak_ism, args.gamma),
    }
    cases = [report["constant_acceleration"], report["ism_cooling"]]
    report["passed"] = all(
        case["maximum_linf"] <= report["absolute_tolerance"]
        and case["maximum_coordinate_linf"] <= report["absolute_tolerance"]
        and case["maximum_time_delta"] <= report["formatted_time_tolerance"]
        and case["maximum_cycle_delta"] == 0
        for case in cases
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    for name, case in (("constant acceleration", cases[0]),
                       ("ISM cooling", cases[1])):
        print(f"{name}: snapshots={case['snapshot_pairs']} "
              f"max_Linf={case['maximum_linf']:.17e}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
