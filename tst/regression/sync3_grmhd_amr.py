#!/usr/bin/env python3
"""Audit the SYNC-3 moving-puncture AMR and protection diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


FLOAT_FIELDS = (
    "mhd.b_cell",
    "mhd.b_face",
    "mhd.cons",
    "mhd.divb",
    "mhd.fofc",
    "mhd.prim",
    "mhd.recovery",
    "nr.adm",
    "nr.constraint_mask",
    "nr.constraints",
    "nr.tmunu",
    "nr.z4c",
)


def history(path: Path) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# ["):
            names = [token.split("]=", 1)[1] for token in line.split() if "]=" in token]
        elif line and not line.startswith("#"):
            rows.append([float(value) for value in line.split()])
    values = np.asarray(rows, dtype=np.float64)
    if not names or values.ndim != 2 or values.shape[1] != len(names):
        raise RuntimeError(f"invalid history file {path}")
    return names, values


def interface_cells(stream: h5py.File) -> np.ndarray:
    """Return finest-level cell layers touching a coarse/fine interface."""
    levels = np.asarray(stream["Levels"], dtype=np.int64)
    face_coordinates = [np.asarray(stream[f"Locations/{axis}"]) for axis in "xyz"]
    bounds = np.column_stack(
        tuple(column for coordinates in face_coordinates for column in (coordinates[:, 0], coordinates[:, -1]))
    )
    domain = np.asarray(
        [bounds[:, 0].min(), bounds[:, 1].max(), bounds[:, 2].min(),
         bounds[:, 3].max(), bounds[:, 4].min(), bounds[:, 5].max()]
    )
    shape = np.asarray(stream["mhd.divb"]).shape
    selected = np.zeros(shape, dtype=bool)
    finest = np.flatnonzero(levels == levels.max())
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, np.max(np.abs(domain)))
    data_axis = {0: 3, 1: 2, 2: 1}

    for block in finest:
        for direction in range(3):
            transverse = tuple(axis for axis in range(3) if axis != direction)
            for side in (0, 1):
                face = bounds[block, 2 * direction + side]
                if abs(face - domain[2 * direction + side]) <= tolerance:
                    continue
                opposite = 1 - side
                same_level_neighbor = False
                for neighbor in finest:
                    if neighbor == block:
                        continue
                    if abs(bounds[neighbor, 2 * direction + opposite] - face) > tolerance:
                        continue
                    overlaps = all(
                        min(bounds[block, 2 * axis + 1], bounds[neighbor, 2 * axis + 1])
                        - max(bounds[block, 2 * axis], bounds[neighbor, 2 * axis])
                        > tolerance
                        for axis in transverse
                    )
                    if overlaps:
                        same_level_neighbor = True
                        break
                if not same_level_neighbor:
                    selection: list[object] = [block, slice(None), slice(None), slice(None)]
                    selection[data_axis[direction]] = 0 if side == 0 else shape[data_axis[direction]] - 1
                    selected[tuple(selection)] = True
    return selected


def maximum_and_rms(values: np.ndarray, selected: np.ndarray) -> tuple[float, float]:
    data = np.abs(values[selected])
    if data.size == 0:
        return 0.0, 0.0
    return float(data.max()), float(np.sqrt(np.mean(data * data)))


def analyze(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as stream:
        for name in FLOAT_FIELDS:
            values = np.asarray(stream[name])
            if not np.isfinite(values).all():
                raise RuntimeError(f"nonfinite values in {path.name}:{name}")

        levels = np.asarray(stream["Levels"], dtype=np.int64)
        interface = interface_cells(stream)
        valid = np.asarray(stream["nr.constraint_mask"])[:, 0] > 0.5
        interface &= valid
        bulk = valid & ~interface
        constraints = np.abs(np.asarray(stream["nr.constraints"]))
        divb = np.abs(np.asarray(stream["mhd.divb"]))
        fofc = np.asarray(stream["mhd.fofc"]) > 0.5
        puncture = np.asarray(stream["mhd.recovery"])[:, 2] > 0.5
        locations = [np.asarray(stream[f"VolumeLocations/{axis}"]) for axis in "xyz"]
        fine = np.flatnonzero(levels == levels.max())
        fine_centroid = [float(locations[axis][fine].mean()) for axis in range(3)]

        constraint_report: dict[str, object] = {}
        for component, name in enumerate(("combined", "hamiltonian", "momentum_norm")):
            interface_stats = maximum_and_rms(constraints[:, component], interface)
            bulk_stats = maximum_and_rms(constraints[:, component], bulk)
            if interface_stats[0] > bulk_stats[0] * (1.0 + 1.0e-12):
                raise RuntimeError(
                    f"constraint interface stripe in {path.name}:{name}: "
                    f"{interface_stats[0]} > {bulk_stats[0]}"
                )
            constraint_report[name] = {
                "interface_maximum": interface_stats[0],
                "interface_rms": interface_stats[1],
                "bulk_maximum": bulk_stats[0],
                "bulk_rms": bulk_stats[1],
            }

        maximum_divb = float(divb.max())
        interface_divb = float(divb[interface].max()) if np.any(interface) else 0.0
        if maximum_divb > 1.0e-12:
            raise RuntimeError(f"divB gate failed in {path.name}: {maximum_divb}")
        nonpuncture_fofc = int(np.count_nonzero(fofc & ~puncture))
        if nonpuncture_fofc != 0:
            raise RuntimeError(f"unexpected FOFC cells outside puncture protection in {path.name}")

        return {
            "file": path.name,
            "time": float(stream["Info"].attrs["Time"]),
            "meshblocks": int(levels.size),
            "level_counts": {str(level): int(np.count_nonzero(levels == level)) for level in np.unique(levels)},
            "fine_level_centroid": fine_centroid,
            "coarse_fine_interface_cells": int(np.count_nonzero(interface)),
            "maximum_abs_divb": maximum_divb,
            "interface_maximum_abs_divb": interface_divb,
            "fofc_cells": int(np.count_nonzero(fofc)),
            "puncture_protection_cells": int(np.count_nonzero(puncture)),
            "fofc_cells_outside_puncture": nonpuncture_fofc,
            "constraints": constraint_report,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    if (args.pangu is None) != (args.input is None):
        raise RuntimeError("--pangu and --input must be supplied together")
    if args.pangu is not None:
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.pangu.resolve()), "-i", str(args.input.resolve()),
            "parthenon/time/dt_force=0.0375", "parthenon/time/tlim=0.75",
            "parthenon/time/nlim=20", "parthenon/output3/include_in_final=false",
        ]
        completed = subprocess.run(
            command, cwd=workdir, env=os.environ.copy(), check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (workdir / "run.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0 or "Driver completed." not in completed.stdout:
            raise RuntimeError(
                f"command failed ({completed.returncode}): {' '.join(command)}\n"
                f"{completed.stdout[-8000:]}"
            )
    outputs = sorted(workdir.glob("*sync_grmhd_puncture*.phdf"))
    if len(outputs) < 2:
        raise RuntimeError(f"need at least two PHDF files in {workdir}")
    snapshots = [analyze(path) for path in outputs]

    names, values = history(next(workdir.glob("*.hst")))
    columns = {name: values[:, index] for index, name in enumerate(names)}
    final_time = float(columns["time"][-1])
    final_cycle = int(columns["cycle"][-1])
    tracker_displacement = float(columns["nr_puncture_tracker_0"][-1] - columns["nr_puncture_tracker_0"][0])
    centroid_displacement = float(snapshots[-1]["fine_level_centroid"][0] - snapshots[0]["fine_level_centroid"][0])
    if final_time < 0.75 - 1.0e-14 or final_cycle < 20:
        raise RuntimeError(f"moving-puncture run ended early: t={final_time}, cycle={final_cycle}")
    if tracker_displacement < 0.25:
        raise RuntimeError(f"puncture tracker did not move far enough: {tracker_displacement}")
    if centroid_displacement < 1.5:
        raise RuntimeError(f"AMR hierarchy did not follow the tracker: {centroid_displacement}")

    report = {
        "schema": "pangu.sync3.grmhd-amr.v1",
        "pass": True,
        "final_time": final_time,
        "final_cycle": final_cycle,
        "tracker_x1_displacement": tracker_displacement,
        "fine_level_centroid_x1_displacement": centroid_displacement,
        "history_maximum_abs_divb": float(np.max(np.abs(columns["mhd_max_abs_divb"]))),
        "history_maximum_fofc_cells": int(np.max(columns["mhd_fofc_cells"])),
        "history_maximum_puncture_cells": int(np.max(columns["mhd_puncture_mask_cells"])),
        "cumulative_recorded_mass_injection": float(np.sum(columns["mhd_floor_mass_injection"])),
        "cumulative_recorded_energy_injection": float(np.sum(columns["mhd_floor_energy_injection"])),
        "snapshots": snapshots,
    }
    (workdir / "amr_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
