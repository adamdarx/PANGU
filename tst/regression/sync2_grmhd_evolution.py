#!/usr/bin/env python3
"""Validate stage-synchronous GRMHD evolution, CT, and matter coupling."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess

import h5py
import numpy as np


FIELDS = (
    "mhd.cons",
    "mhd.prim",
    "mhd.b_cell",
    "mhd.b_face",
    "mhd.divb",
    "nr.z4c",
    "nr.adm",
    "nr.tmunu",
    "nr.constraints",
)


def run(command: list[str], directory: pathlib.Path) -> None:
    directory.mkdir(parents=True)
    completed = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    (directory / "run.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"run failed with {completed.returncode}: {' '.join(command)}\n"
            f"{completed.stdout}{completed.stderr}"
        )


def one_dump(directory: pathlib.Path, suffix: str) -> pathlib.Path:
    paths = sorted(directory.glob(f"sync_grmhd_linear_wave.sync_grmhd_wave.{suffix}.phdf"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {suffix} dump in {directory}, found {len(paths)}")
    return paths[0]


def finite_fields(handle: h5py.File, path: pathlib.Path) -> None:
    for field in FIELDS:
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
        [executable, "-i", input_path, "numerical_relativity/matter_source=zero"],
        reference,
    )

    initial_path = one_dump(coupled, "00000")
    coupled_path = one_dump(coupled, "final")
    reference_path = one_dump(reference, "final")
    maximum_mhd_scaled = 0.0
    maximum_spacetime_response = 0.0
    maximum_coupled_tmunu = 0.0
    maximum_reference_tmunu = 0.0
    maximum_divergence = 0.0
    magnetic_evolution = 0.0
    with (
        h5py.File(initial_path, "r") as initial,
        h5py.File(coupled_path, "r") as active,
        h5py.File(reference_path, "r") as passive,
    ):
        finite_fields(initial, initial_path)
        finite_fields(active, coupled_path)
        finite_fields(passive, reference_path)
        active_time = float(active["Info"].attrs["Time"])
        passive_time = float(passive["Info"].attrs["Time"])
        if active_time != passive_time or active_time <= 0.0:
            raise RuntimeError(f"invalid final times {active_time} and {passive_time}")
        for field in ("mhd.cons", "mhd.prim", "mhd.b_cell", "mhd.b_face"):
            measured = np.asarray(active[field], dtype=np.float64)
            expected = np.asarray(passive[field], dtype=np.float64)
            scale = max(float(np.max(np.abs(expected))), np.finfo(np.float64).tiny)
            maximum_mhd_scaled = max(
                maximum_mhd_scaled, float(np.max(np.abs(measured - expected))) / scale
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
        maximum_divergence = max(
            float(np.max(np.abs(np.asarray(active["mhd.divb"], dtype=np.float64)))),
            float(np.max(np.abs(np.asarray(passive["mhd.divb"], dtype=np.float64)))),
        )
        magnetic_evolution = float(
            np.max(
                np.abs(
                    np.asarray(active["mhd.b_cell"], dtype=np.float64)
                    - np.asarray(initial["mhd.b_cell"], dtype=np.float64)
                )
            )
        )

    tolerance = 512.0 * np.finfo(np.float64).eps
    if not math.isfinite(maximum_mhd_scaled) or maximum_mhd_scaled > tolerance:
        raise RuntimeError(
            f"Minkowski-limit MHD mismatch {maximum_mhd_scaled:.17e} > {tolerance:.17e}"
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
    if maximum_divergence > tolerance:
        raise RuntimeError(f"CT divergence {maximum_divergence:.17e} > {tolerance:.17e}")
    if not magnetic_evolution > 0.0:
        raise RuntimeError("the magnetic perturbation did not evolve")

    report = {
        "schema": "pangu.sync2.grmhd-evolution.v1",
        "status": "pass",
        "final_time": active_time,
        "maximum_mhd_scaled_difference": maximum_mhd_scaled,
        "maximum_spacetime_response": maximum_spacetime_response,
        "maximum_coupled_tmunu": maximum_coupled_tmunu,
        "maximum_reference_tmunu": maximum_reference_tmunu,
        "maximum_divergence": maximum_divergence,
        "magnetic_evolution": magnetic_evolution,
        "tolerance": tolerance,
    }
    (workdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "SYNC-2 GRMHD evolution PASS: "
        f"mhd={maximum_mhd_scaled:.3e} spacetime={maximum_spacetime_response:.3e} "
        f"divB={maximum_divergence:.3e} magnetic_evolution={magnetic_evolution:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
