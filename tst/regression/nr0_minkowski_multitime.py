#!/usr/bin/env python3
"""Validate NR-0 Minkowski snapshots from PANGU and AthenaK."""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import h5py
import numpy as np


EXPECTED_TIMES = np.asarray((0.0, 0.025, 0.05))
Z4C_EXPECTED = np.zeros(22)
Z4C_EXPECTED[[0, 1, 4, 6, 18]] = 1.0
ADM_EXPECTED = np.zeros(13)
ADM_EXPECTED[[0, 3, 5, 12]] = 1.0
TOLERANCE = 128.0 * np.finfo(np.float64).eps


def field_metrics(values: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    component_axes = (0,) + tuple(range(2, values.ndim))
    component_min = np.min(values, axis=component_axes)
    component_max = np.max(values, axis=component_axes)
    return {
        "max_error": float(np.max(np.abs(values - expected[None, :, None, None, None]))),
        "component_min": component_min.tolist(),
        "component_max": component_max.tolist(),
    }


def pangu_snapshots(directory: pathlib.Path) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    for path in directory.glob("*.phdf"):
        with h5py.File(path, "r") as stream:
            z4c = np.asarray(stream["nr.z4c"])
            adm = np.asarray(stream["nr.adm"])
            snapshots.append({
                "path": path.name,
                "time": float(stream["Info"].attrs["Time"]),
                "cycle": int(stream["Info"].attrs["NCycle"]),
                "z4c": field_metrics(z4c, Z4C_EXPECTED),
                "adm": field_metrics(adm, ADM_EXPECTED),
            })
    return sorted(snapshots, key=lambda snapshot: float(snapshot["time"]))


def athenak_snapshots(directory: pathlib.Path) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    for path in (directory / "tab").glob("*.tab"):
        lines = path.read_text(encoding="utf-8").splitlines()
        time_match = re.search(r"time=([+\-0-9.eE]+)", lines[0])
        cycle_match = re.search(r"cycle=(\d+)", lines[0])
        if time_match is None or cycle_match is None:
            raise RuntimeError(f"missing time/cycle metadata in {path}")
        table = np.atleast_2d(np.loadtxt(path, comments="#"))
        values = table[:, 3:]
        component_min = np.min(values, axis=0)
        component_max = np.max(values, axis=0)
        snapshots.append({
            "path": path.name,
            "time": float(time_match.group(1)),
            "cycle": int(cycle_match.group(1)),
            "z4c": {
                "max_error": float(np.max(np.abs(values - Z4C_EXPECTED[None, :]))),
                "component_min": component_min.tolist(),
                "component_max": component_max.tolist(),
            },
        })
    return sorted(snapshots, key=lambda snapshot: float(snapshot["time"]))


def constraint_maximum(directory: pathlib.Path) -> float:
    paths = list(directory.glob("*.hst"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one AthenaK history file, found {len(paths)}")
    rows = np.atleast_2d(np.loadtxt(paths[0], comments="#"))
    return float(np.max(np.abs(rows[:, 2:10])))


def compare(pangu: list[dict[str, object]],
            athenak: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(pangu) != len(EXPECTED_TIMES) or len(athenak) != len(EXPECTED_TIMES):
        raise RuntimeError(
            f"expected three snapshots from each code, got {len(pangu)} and {len(athenak)}")
    records: list[dict[str, object]] = []
    for expected_time, lhs, rhs in zip(EXPECTED_TIMES, pangu, athenak):
        pangu_z4c = lhs["z4c"]
        pangu_adm = lhs["adm"]
        athenak_z4c = rhs["z4c"]
        cross_min = np.max(np.abs(
            np.asarray(pangu_z4c["component_min"]) -
            np.asarray(athenak_z4c["component_min"])))
        cross_max = np.max(np.abs(
            np.asarray(pangu_z4c["component_max"]) -
            np.asarray(athenak_z4c["component_max"])))
        record = {
            "time": float(expected_time),
            "time_delta": max(abs(float(lhs["time"]) - expected_time),
                              abs(float(rhs["time"]) - expected_time)),
            "pangu_z4c_linf": pangu_z4c["max_error"],
            "pangu_adm_linf": pangu_adm["max_error"],
            "athenak_z4c_linf": athenak_z4c["max_error"],
            "cross_code_component_envelope_linf": float(max(cross_min, cross_max)),
            "pangu_file": lhs["path"],
            "athenak_file": rhs["path"],
        }
        numeric_metrics = (
            record["time_delta"], record["pangu_z4c_linf"],
            record["pangu_adm_linf"], record["athenak_z4c_linf"],
            record["cross_code_component_envelope_linf"],
        )
        if max(float(metric) for metric in numeric_metrics) > TOLERANCE:
            raise RuntimeError(f"NR-0 comparison failed at t={expected_time}: {record}")
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu-directory", required=True, type=pathlib.Path)
    parser.add_argument("--athenak-directory", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    records = compare(pangu_snapshots(args.pangu_directory),
                      athenak_snapshots(args.athenak_directory))
    constraints = constraint_maximum(args.athenak_directory)
    if constraints > TOLERANCE:
        raise RuntimeError(f"AthenaK constraint maximum {constraints} exceeds {TOLERANCE}")
    result = {
        "status": "pass",
        "tolerance": TOLERANCE,
        "athenak_constraint_linf": constraints,
        "snapshots": records,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
