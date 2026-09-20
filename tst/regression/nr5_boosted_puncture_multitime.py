#!/usr/bin/env python3
"""Compare matched PANGU/AthenaK boosted-puncture evolutions at six cycles."""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np


CYCLES = (0, 10, 50, 100, 150, 200)
AXES = {
    "x": (lambda field: field[:, 32, 32, :].T, "z4c", "con"),
    "y": (lambda field: field[:, 32, :, 32].T, "yline", "ycon"),
    "z": (lambda field: field[:, :, 32, 32].T, "zline", "zcon"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=Path)
    parser.add_argument("--athenak", required=True, type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--field-tolerance", type=float, default=5.0e-12)
    parser.add_argument("--tracker-tolerance", type=float, default=5.0e-13)
    return parser.parse_args()


def pangu_files(directory):
    result = {}
    for path in directory.glob("*.phdf"):
        with h5py.File(path, "r") as data:
            cycle = int(data["Info"].attrs["NCycle"])
        # Prefer the scheduled file, but accept Parthenon's final-only cycle 200 dump.
        if cycle not in result or ".final." not in path.name:
            result[cycle] = path
    return result


def tabular(path):
    with path.open(encoding="utf-8") as stream:
        header = stream.readline()
    match = re.search(r"time=([+\-0-9.eE]+)\s+cycle=(\d+)", header)
    if not match:
        raise RuntimeError(f"cannot parse AthenaK time/cycle header in {path}")
    return float(match.group(1)), int(match.group(2)), np.loadtxt(path)[:, 3:]


def scaled_errors(measured, reference, trim=3):
    measured = measured[trim:-trim]
    reference = reference[trim:-trim]
    difference = np.abs(measured - reference)
    scale = np.maximum(1.0, np.maximum(np.abs(measured), np.abs(reference)))
    return {
        "max_abs": float(np.max(difference)),
        "scaled_max": float(np.max(difference / scale)),
        "mean_abs": float(np.mean(difference)),
    }


def comparable_constraints(values):
    # PANGU stores C, M, and Z magnitudes. AthenaK's con_C, con_M, and con_Z
    # tabular fields store their squares; signed H and M_i need no conversion.
    return np.column_stack(
        (
            values[:, 0] ** 2,
            values[:, 1],
            values[:, 2] ** 2,
            values[:, 3] ** 2,
            values[:, 4:7],
        )
    )


def tracker_errors(pangu_directory, athenak_directory):
    pangu = np.loadtxt(pangu_directory / "nr_boosted_puncture.out1.hst")
    athenak = np.loadtxt(athenak_directory / "z4c.co_0.txt")
    if pangu.shape[0] < 201 or athenak.shape[0] < 200:
        raise RuntimeError("tracker histories do not contain the complete 200-step trajectory")
    # AthenaK writes iteration n after advancing step n+1 while retaining the
    # beginning-of-step time label. PANGU history writes the same state at cycle n+1.
    difference = np.abs(pangu[1:201, 6:9] - athenak[:200, 2:5])
    return {
        "max_abs": float(np.max(difference)),
        "axis_max_abs": [float(value) for value in np.max(difference, axis=0)],
        "mean_abs": float(np.mean(difference)),
    }


def main():
    args = parse_args()
    files = pangu_files(args.pangu)
    report = {"cycles": {}, "storage_convention": "PANGU C/M/Z squared for AthenaK con_*"}
    worst = 0.0
    for cycle in CYCLES:
        output_index = cycle // 10
        if cycle not in files:
            raise RuntimeError(f"missing PANGU cycle {cycle}")
        with h5py.File(files[cycle], "r") as data:
            time = float(data["Info"].attrs["Time"])
            z4c = np.asarray(data["nr.z4c"][0], dtype=np.float64)
            adm = np.asarray(data["nr.adm"][0], dtype=np.float64)
            constraints = np.asarray(data["nr.constraints"][0], dtype=np.float64)
        cycle_report = {"time": time, "axes": {}}
        for axis, (take_line, z4c_id, constraint_id) in AXES.items():
            reference_time, reference_cycle, reference_z4c = tabular(
                args.athenak / "tab" / f"z4c.{z4c_id}.{output_index:05d}.tab"
            )
            _, constraint_cycle, reference_constraints = tabular(
                args.athenak / "tab" / f"z4c.{constraint_id}.{output_index:05d}.tab"
            )
            if reference_cycle != cycle or constraint_cycle != cycle:
                raise RuntimeError(f"AthenaK output index {output_index} is not cycle {cycle}")
            if abs(reference_time - time) > 5.0e-13:
                raise RuntimeError(f"time mismatch at cycle {cycle}: {time} vs {reference_time}")
            axis_report = {
                "z4c": scaled_errors(take_line(z4c), reference_z4c),
                "constraints": scaled_errors(
                    comparable_constraints(take_line(constraints)), reference_constraints
                ),
            }
            if axis == "x":
                _, adm_cycle, reference_adm = tabular(
                    args.athenak / "tab" / f"z4c.adm.{output_index:05d}.tab"
                )
                if adm_cycle != cycle:
                    raise RuntimeError(f"AthenaK ADM output is not cycle {cycle}")
                axis_report["adm"] = scaled_errors(take_line(adm), reference_adm)
            cycle_report["axes"][axis] = axis_report
            for values in axis_report.values():
                worst = max(worst, values["scaled_max"])
        report["cycles"][str(cycle)] = cycle_report

    report["tracker"] = tracker_errors(args.pangu, args.athenak)
    report["worst_scaled_field_error"] = worst
    report["field_tolerance"] = args.field_tolerance
    report["tracker_tolerance"] = args.tracker_tolerance
    report["pass"] = bool(
        worst <= args.field_tolerance
        and report["tracker"]["max_abs"] <= args.tracker_tolerance
    )
    serialized = json.dumps(report, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
