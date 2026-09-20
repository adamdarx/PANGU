#!/usr/bin/env python3
"""Compare six fixed-GR torus snapshots with AthenaK in 3-D and float64."""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re

import h5py
import numpy as np


FIELDS = ("density", "u1", "u2", "u3", "pressure")


def metrics(delta: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        field: {
            "l1": float(np.mean(values)),
            "l2": float(np.sqrt(np.mean(values * values))),
            "linf": float(np.max(values)),
        }
        for field, values in zip(FIELDS, delta)
    }


def pangu_snapshot(path: pathlib.Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["hydro.prim"])
        x_locations = np.asarray(handle["VolumeLocations/x"])
        y_locations = np.asarray(handle["VolumeLocations/y"])
        z_locations = np.asarray(handle["VolumeLocations/z"])
        payloads: list[np.ndarray] = []
        coordinates: list[np.ndarray] = []
        for block in range(values.shape[0]):
            z, y, x = np.meshgrid(z_locations[block], y_locations[block],
                                  x_locations[block], indexing="ij")
            payloads.append(values[block].reshape(5, -1))
            coordinates.append(np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1))
        payload = np.concatenate(payloads, axis=1)
        coordinate = np.concatenate(coordinates, axis=0)
        order = np.lexsort((coordinate[:, 0], coordinate[:, 1], coordinate[:, 2]))
        return {
            "cycle": int(handle["Info"].attrs["NCycle"]),
            "time": float(handle["Info"].attrs["Time"]),
            "values": payload[:, order],
            "coordinates": coordinate[order],
            "path": path.name,
        }


def pangu_snapshots(directory: pathlib.Path) -> list[dict[str, object]]:
    paths = [pathlib.Path(name) for name in sorted(glob.glob(str(directory / "*.phdf")))]
    if len(paths) < 5:
        raise RuntimeError(f"{directory}: expected at least five PHDF snapshots")
    return [pangu_snapshot(path) for path in paths]


def athenak_history_times(directory: pathlib.Path) -> np.ndarray:
    histories = sorted(directory.glob("*.hst"))
    if len(histories) != 1:
        raise RuntimeError(f"{directory}: expected exactly one AthenaK history file")
    return np.loadtxt(histories[0], comments="#", ndmin=2)[:, 0]


def athenak_vtk(path: pathlib.Path, gamma: float) -> dict[str, object]:
    with path.open("rb") as handle:
        header = bytearray()
        first_scalar = b""
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"{path}: missing VTK scalar payload")
            header.extend(line)
            if line.startswith(b"SCALARS "):
                first_scalar = line
                break
        metadata = re.search(rb"time=\s*([^ ]+).*cycle=(\d+)", bytes(header))
        cell_count_match = re.search(rb"CELL_DATA\s+(\d+)", bytes(header))
        if metadata is None or cell_count_match is None:
            raise RuntimeError(f"{path}: incomplete VTK metadata")
        cell_count = int(cell_count_match.group(1))
        labels: list[str] = []
        arrays: list[np.ndarray] = []
        scalar = first_scalar
        while scalar:
            labels.append(scalar.split()[1].decode("utf-8"))
            lookup = handle.readline()
            if not lookup.startswith(b"LOOKUP_TABLE"):
                raise RuntimeError(f"{path}: malformed VTK lookup table")
            arrays.append(np.frombuffer(handle.read(4 * cell_count), dtype=">f4").astype(float))
            handle.readline()
            scalar = handle.readline()
    if labels != ["dens", "velx", "vely", "velz", "eint"]:
        raise RuntimeError(f"{path}: unexpected fields {labels}")
    values = np.asarray(arrays)
    values[4] *= gamma - 1.0
    return {
        "cycle": int(metadata.group(2)),
        "time": float(metadata.group(1)),
        "values": values,
        "path": path.name,
    }


def athenak_vtk_snapshots(directory: pathlib.Path, gamma: float) -> list[dict[str, object]]:
    paths = sorted((directory / "vtk").glob("*.vtk"))
    if len(paths) < 5:
        raise RuntimeError(f"{directory}: expected at least five AthenaK VTK snapshots")
    return [athenak_vtk(path, gamma) for path in paths]


def athenak_tab_snapshots(directory: pathlib.Path, gamma: float) -> list[dict[str, object]]:
    paths = sorted((directory / "tab").glob("*.tab"))
    times = athenak_history_times(directory)
    if len(paths) < 5 or len(paths) != len(times):
        raise RuntimeError(f"{directory}: history/table count mismatch")
    snapshots: list[dict[str, object]] = []
    for index, path in enumerate(paths):
        lines = path.read_text(encoding="utf-8").splitlines()
        metadata = re.search(r"cycle=(\d+)", lines[0])
        if metadata is None:
            raise RuntimeError(f"{path}: missing cycle")
        rows = np.loadtxt(path, comments="#", ndmin=2)
        order = np.argsort(rows[:, 2])
        values = rows[order, 3:8].T
        values[4] *= gamma - 1.0
        snapshots.append({
            "cycle": int(metadata.group(1)),
            "time": float(times[index]),
            "x": rows[order, 2],
            "values": values,
            "path": path.name,
        })
    return snapshots


def pangu_line(snapshot: dict[str, object], y_slice: float,
                z_slice: float) -> tuple[np.ndarray, np.ndarray]:
    coordinates = snapshot["coordinates"]
    y_values = np.unique(coordinates[:, 1])
    z_values = np.unique(coordinates[:, 2])
    y = y_values[np.argmin(np.abs(y_values - y_slice))]
    z = z_values[np.argmin(np.abs(z_values - z_slice))]
    selected = np.isclose(coordinates[:, 1], y) & np.isclose(coordinates[:, 2], z)
    order = np.argsort(coordinates[selected, 0])
    return coordinates[selected, 0][order], snapshot["values"][:, selected][:, order]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=pathlib.Path)
    parser.add_argument("--athenak-vtk", required=True, type=pathlib.Path)
    parser.add_argument("--athenak-tab", required=True, type=pathlib.Path)
    parser.add_argument("--slice-y", type=float, default=0.75)
    parser.add_argument("--slice-z", type=float, default=0.75)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    gamma = 4.0 / 3.0
    pangu = pangu_snapshots(args.pangu)
    vtk = athenak_vtk_snapshots(args.athenak_vtk, gamma)
    tab = athenak_tab_snapshots(args.athenak_tab, gamma)
    if not (len(pangu) == len(vtk) == len(tab)):
        raise RuntimeError("PANGU/VTK/table snapshot counts differ")

    records: list[dict[str, object]] = []
    for index, (p_snapshot, v_snapshot, t_snapshot) in enumerate(zip(pangu, vtk, tab)):
        full_delta = np.abs(p_snapshot["values"] - v_snapshot["values"])
        line_x, line_values = pangu_line(p_snapshot, args.slice_y, args.slice_z)
        line_delta = np.abs(line_values - t_snapshot["values"])
        records.append({
            "snapshot": index,
            "pangu_file": p_snapshot["path"],
            "athenak_vtk_file": v_snapshot["path"],
            "athenak_tab_file": t_snapshot["path"],
            "cycle": p_snapshot["cycle"],
            "cycle_delta": max(abs(p_snapshot["cycle"] - v_snapshot["cycle"]),
                               abs(p_snapshot["cycle"] - t_snapshot["cycle"])),
            "time": p_snapshot["time"],
            "time_delta": max(abs(p_snapshot["time"] - v_snapshot["time"]),
                              abs(p_snapshot["time"] - t_snapshot["time"])),
            "line_x_linf": float(np.max(np.abs(line_x - t_snapshot["x"]))),
            "full_grid_maximum_linf": float(np.max(full_delta)),
            "high_precision_line_maximum_linf": float(np.max(line_delta)),
            "full_grid_fields": metrics(full_delta),
            "high_precision_line_fields": metrics(line_delta),
        })

    full_grid_tolerance = 4.0e-8
    high_precision_tolerance = 1.0e-11
    metadata_tolerance = 1.0e-12
    passed = all(
        record["full_grid_maximum_linf"] <= full_grid_tolerance
        and record["high_precision_line_maximum_linf"] <= high_precision_tolerance
        and record["time_delta"] <= metadata_tolerance
        and record["line_x_linf"] <= metadata_tolerance
        and record["cycle_delta"] == 0
        for record in records)
    report = {
        "passed": passed,
        "case": "fixed-GR Fishbone-Moncrief hydrodynamic torus",
        "snapshot_pairs": len(records),
        "full_grid_output_precision": "AthenaK VTK float32",
        "full_grid_tolerance": full_grid_tolerance,
        "high_precision_line_tolerance": high_precision_tolerance,
        "metadata_tolerance": metadata_tolerance,
        "full_grid_maximum_linf": max(
            record["full_grid_maximum_linf"] for record in records),
        "high_precision_line_maximum_linf": max(
            record["high_precision_line_maximum_linf"] for record in records),
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"GR torus: snapshots={len(records)} passed={passed} "
          f"full_grid_max_Linf={report['full_grid_maximum_linf']:.17e} "
          f"float64_line_max_Linf={report['high_precision_line_maximum_linf']:.17e}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
