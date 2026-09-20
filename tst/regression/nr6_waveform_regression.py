#!/usr/bin/env python3
"""Validate NR-6 Weyl extraction across MeshBlocks and, optionally, AthenaK."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


MODE_COLUMNS = tuple(
    f"{part}_l{ell}_m{m}"
    for ell in range(2, 9)
    for m in range(-ell, ell + 1)
    for part in ("re", "im")
)


def run(command: list[str], directory: Path, log_name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / log_name).open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)


def waveform_rows(path: Path) -> list[dict[str, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or "interpolation_points=4" not in lines[0]:
        raise RuntimeError(f"missing four-point interpolation provenance in {path}")
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    if len(rows) != 6:
        raise RuntimeError(f"expected six waveform rows in {path}, found {len(rows)}")
    return [{key: float(value) for key, value in row.items()} for row in rows]


def compare_multiblock(single: list[dict[str, float]], multi: list[dict[str, float]]) -> float:
    maximum = 0.0
    for left, right in zip(single, multi, strict=True):
        for key in left:
            maximum = max(maximum, abs(left[key] - right[key]))
    return maximum


def pangu_snapshots(directory: Path) -> dict[int, tuple[float, np.ndarray]]:
    result: dict[int, tuple[float, np.ndarray]] = {}
    for path in directory.glob("*.phdf"):
        with h5py.File(path, "r") as data:
            cycle = int(data["Info"].attrs["NCycle"])
            time = float(data["Info"].attrs["Time"])
            weyl = np.asarray(data["nr.weyl"][0, :, 15, 15, :], dtype=np.float64).T
        if cycle not in result or ".final." not in path.name:
            result[cycle] = (time, weyl)
    return result


def athena_tab(path: Path) -> tuple[float, int, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    match = re.search(r"time=([+\-0-9.eE]+)\s+cycle=(\d+)", lines[0])
    if match is None:
        raise RuntimeError(f"cannot parse AthenaK header in {path}")
    return float(match.group(1)), int(match.group(2)), np.loadtxt(path)[:, 3:5]


def compare_athenak_fields(pangu_dir: Path, athena_dir: Path) -> float:
    pangu = pangu_snapshots(pangu_dir)
    maximum = 0.0
    seen: set[int] = set()
    for path in sorted((athena_dir / "tab").glob("*.tab")):
        time, cycle, reference = athena_tab(path)
        # AthenaK's cycle-0 tab is allocated but not populated; formal Weyl
        # comparison begins with the first completed evolution step.
        if cycle == 0 or cycle in seen or cycle not in pangu:
            continue
        seen.add(cycle)
        pangu_time, measured = pangu[cycle]
        if abs(time - pangu_time) > 5.0e-13:
            raise RuntimeError(f"Weyl time mismatch at cycle {cycle}: {pangu_time} vs {time}")
        maximum = max(maximum, float(np.max(np.abs(measured - reference))))
    if not {1, 2}.issubset(seen):
        raise RuntimeError(f"AthenaK Weyl tabs omit required cycles: {seen}")
    return maximum


def athena_mode_rows(directory: Path, dt: float) -> list[dict[str, float]]:
    real_path = directory / "waveforms" / "rpsi4_real_0.25.txt"
    imag_path = directory / "waveforms" / "rpsi4_imag_0.25.txt"
    real_values = np.atleast_2d(np.loadtxt(real_path))
    imag_values = np.atleast_2d(np.loadtxt(imag_path))
    if real_values.shape[0] < 2 or real_values.shape != imag_values.shape:
        raise RuntimeError(f"incomplete AthenaK waveform in {real_path}")
    # AthenaK evaluates the post-step Weyl field but writes the beginning-of-step
    # time.  Therefore the physical time represented by each row is printed_time+dt.
    result: list[dict[str, float]] = []
    for real_row, imag_row in zip(real_values, imag_values, strict=True):
        row = {"time": float(real_row[0] + dt)}
        mode = 1
        for ell in range(2, 9):
            for m in range(-ell, ell + 1):
                row[f"re_l{ell}_m{m}"] = float(real_row[mode])
                row[f"im_l{ell}_m{m}"] = float(imag_row[mode])
                mode += 1
        result.append(row)
    return result


def compare_athenak_modes(pangu: list[dict[str, float]], athena_dir: Path) -> tuple[float, float]:
    selected = [row for row in pangu if abs(row["radius"] - 0.25) < 1.0e-13]
    if len(selected) != 2:
        raise RuntimeError("PANGU waveform does not contain both r=0.25 samples")
    dt = selected[1]["time"] - selected[0]["time"]
    reference = athena_mode_rows(athena_dir, dt)
    maximum_absolute = 0.0
    maximum_scaled = 0.0
    for measured in selected:
        matches = [row for row in reference if abs(row["time"] - measured["time"]) < 5.0e-13]
        if len(matches) != 1:
            raise RuntimeError(f"no unique AthenaK physical-time row for t={measured['time']}")
        for key in MODE_COLUMNS:
            difference = abs(measured[key] - matches[0][key])
            maximum_absolute = max(maximum_absolute, difference)
            denominator = max(abs(matches[0][key]), abs(measured[key]), 1.0e-12)
            maximum_scaled = max(maximum_scaled, difference / denominator)
    return maximum_absolute, maximum_scaled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=Path)
    parser.add_argument("--pangu-input", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--athenak", type=Path)
    parser.add_argument("--athenak-input", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    if args.workdir.exists():
        shutil.rmtree(args.workdir)
    single_dir = args.workdir / "pangu_single"
    multi_dir = args.workdir / "pangu_multi"
    base = [str(args.pangu.resolve()), "-i", str(args.pangu_input.resolve()), "-d", "."]
    run(base, single_dir, "run.log")
    run(
        base
        + [
            "parthenon/meshblock/nx1=16",
            "parthenon/meshblock/nx2=16",
            "parthenon/meshblock/nx3=16",
        ],
        multi_dir,
        "run.log",
    )
    single = waveform_rows(single_dir / "diagnostics" / "nr_waveforms.csv")
    multi = waveform_rows(multi_dir / "diagnostics" / "nr_waveforms.csv")
    multiblock_error = compare_multiblock(single, multi)

    report: dict[str, object] = {
        "interpolation": "four-point tensor-product Lagrange",
        "pangu_multiblock_max_abs": multiblock_error,
        "pangu_multiblock_tolerance": 5.0e-19,
        "athenak_waveform_time_convention": "physical_time=printed_time+dt",
    }
    passed = multiblock_error <= 5.0e-19
    if args.athenak or args.athenak_input:
        if not args.athenak or not args.athenak_input:
            raise RuntimeError("--athenak and --athenak-input must be provided together")
        athena_dir = args.workdir / "athenak"
        run(
            [str(args.athenak.resolve()), "-i", str(args.athenak_input.resolve()), "-d", "."],
            athena_dir,
            "run.log",
        )
        field_error = compare_athenak_fields(single_dir, athena_dir)
        mode_absolute, mode_scaled = compare_athenak_modes(single, athena_dir)
        report.update(
            {
                "athenak_weyl_linf": field_error,
                "athenak_weyl_tolerance": 5.0e-19,
                "athenak_all_77_modes_absolute_max": mode_absolute,
                "athenak_all_77_modes_scaled_max": mode_scaled,
                "athenak_all_77_modes_scaled_tolerance": 3.0e-4,
            }
        )
        passed = passed and field_error <= 5.0e-19 and mode_scaled <= 3.0e-4
    report["pass"] = passed
    serialized = json.dumps(report, indent=2)
    summary = args.summary or args.workdir / "nr6_waveform_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
