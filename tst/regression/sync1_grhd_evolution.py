#!/usr/bin/env python3
"""Validate stage-synchronous GRHD evolution in the Minkowski limit."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess

import h5py
import numpy as np


def run(command: list[str], directory: pathlib.Path) -> None:
    directory.mkdir(parents=True)
    completed = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    (directory / "run.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"run failed with {completed.returncode}: {' '.join(command)}\n"
            f"{completed.stdout}{completed.stderr}"
        )


def final_dump(directory: pathlib.Path) -> pathlib.Path:
    paths = sorted(directory.glob("sync_grhd_linear_wave.sync_grhd_wave.final.phdf"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one final dump in {directory}, found {len(paths)}")
    return paths[0]


def finite_fields(handle: h5py.File, path: pathlib.Path) -> None:
    for field in ("hydro.cons", "hydro.prim", "nr.z4c", "nr.adm", "nr.tmunu", "nr.constraints"):
        values = np.asarray(handle[field], dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite {field} values in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    coupled = workdir / "coupled"
    reference = workdir / "test_fluid_reference"
    executable = str(args.pangu.resolve())
    input_path = str(args.input.resolve())
    run([executable, "-i", input_path], coupled)
    run(
        [
            executable,
            "-i",
            input_path,
            "numerical_relativity/matter_source=zero",
        ],
        reference,
    )

    coupled_path = final_dump(coupled)
    reference_path = final_dump(reference)
    maximum_hydro_scaled = 0.0
    maximum_spacetime_response = 0.0
    maximum_coupled_tmunu = 0.0
    maximum_reference_tmunu = 0.0
    with h5py.File(coupled_path, "r") as active, h5py.File(reference_path, "r") as passive:
        finite_fields(active, coupled_path)
        finite_fields(passive, reference_path)
        active_time = float(active["Info"].attrs["Time"])
        passive_time = float(passive["Info"].attrs["Time"])
        if active_time != passive_time:
            raise RuntimeError(f"final-time mismatch {active_time} != {passive_time}")
        for field in ("hydro.cons", "hydro.prim"):
            measured = np.asarray(active[field], dtype=np.float64)
            expected = np.asarray(passive[field], dtype=np.float64)
            scale = max(float(np.max(np.abs(expected))), np.finfo(np.float64).tiny)
            maximum_hydro_scaled = max(
                maximum_hydro_scaled, float(np.max(np.abs(measured - expected))) / scale
            )
        maximum_spacetime_response = float(
            np.max(
                np.abs(
                    np.asarray(active["nr.z4c"], dtype=np.float64)
                    - np.asarray(passive["nr.z4c"], dtype=np.float64)
                )
            )
        )
        maximum_coupled_tmunu = float(
            np.max(np.abs(np.asarray(active["nr.tmunu"], dtype=np.float64)))
        )
        maximum_reference_tmunu = float(
            np.max(np.abs(np.asarray(passive["nr.tmunu"], dtype=np.float64)))
        )

    tolerance = 512.0 * np.finfo(np.float64).eps
    if not math.isfinite(maximum_hydro_scaled) or maximum_hydro_scaled > tolerance:
        raise RuntimeError(
            f"Minkowski-limit hydro mismatch {maximum_hydro_scaled:.17e} > {tolerance:.17e}"
        )
    if not (0.0 < maximum_spacetime_response < 1.0e-15):
        raise RuntimeError(
            "coupled spacetime response was absent or too large: "
            f"{maximum_spacetime_response:.17e}"
        )
    if not (maximum_coupled_tmunu > 0.0 and maximum_reference_tmunu == 0.0):
        raise RuntimeError(
            "stress-energy paths were not distinguished: "
            f"coupled={maximum_coupled_tmunu:.17e} reference={maximum_reference_tmunu:.17e}"
        )

    report = {
        "schema": "pangu.sync1.grhd-evolution.v1",
        "status": "pass",
        "final_time": active_time,
        "maximum_hydro_scaled_difference": maximum_hydro_scaled,
        "maximum_spacetime_response": maximum_spacetime_response,
        "maximum_coupled_tmunu": maximum_coupled_tmunu,
        "maximum_reference_tmunu": maximum_reference_tmunu,
        "tolerance": tolerance,
    }
    (workdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "SYNC-1 GRHD evolution PASS: "
        f"hydro={maximum_hydro_scaled:.3e} spacetime={maximum_spacetime_response:.3e} "
        f"Tmunu={maximum_coupled_tmunu:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
