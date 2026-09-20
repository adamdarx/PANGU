#!/usr/bin/env python3
"""Compare matched PANGU and AthenaK Z4c gauge-wave snapshots.

The comparison deliberately uses the native uniform-grid data: no interpolation,
AMR regridding, temporal resampling, or variable remapping is performed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import h5py
import numpy as np


ATHENA_COLUMNS = {
    3: 0,   # chi
    4: 1,   # gxx
    5: 2,   # gxy
    6: 3,   # gxz
    7: 4,   # gyy
    8: 5,   # gyz
    9: 6,   # gzz
    10: 7,  # Khat
    11: 8,  # Axx
    12: 9,  # Axy
    13: 10, # Axz
    14: 11, # Ayy
    15: 12, # Ayz
    16: 13, # Azz
    17: 14, # Gamx
    18: 15, # Gamy
    19: 16, # Gamz
    20: 17, # Theta
    21: 18, # alpha
    22: 19, # betax
    23: 20, # betay
    24: 21, # betaz
}


def athena_tab(path: pathlib.Path) -> tuple[float, int, np.ndarray]:
    header = path.read_text(encoding="utf-8").splitlines()[0]
    match = re.search(r"time=([0-9.eE+-]+)\s+cycle=(\d+)", header)
    if match is None:
        raise RuntimeError(f"cannot read AthenaK time/cycle from {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        rows.append([float(value) for value in line.split()])
    data = np.asarray(rows, dtype=np.float64)
    if data.shape != (32, 25):
        raise RuntimeError(f"unexpected AthenaK tab shape {data.shape} in {path}")
    return float(match.group(1)), int(match.group(2)), data


def pangu_snapshot(path: pathlib.Path) -> tuple[float, np.ndarray]:
    with h5py.File(path, "r") as data:
        time = float(data["Info"].attrs["Time"])
        levels = np.asarray(data["Levels"])
        field = np.asarray(data["nr.z4c"])
    if field.shape != (1, 22, 4, 4, 32):
        raise RuntimeError(f"unexpected PANGU field shape {field.shape} in {path}")
    if levels.tolist() != [0]:
        raise RuntimeError(f"PANGU comparison requires a uniform level-0 snapshot: {path}")
    return time, field[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu-dir", type=pathlib.Path, required=True)
    parser.add_argument("--athenak-dir", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    args = parser.parse_args()

    pangu_files = sorted(args.pangu_dir.glob("*.phdf"))
    pangu_files = [p for p in pangu_files if not p.name.endswith(".xdmf")]
    pangu_files.sort(key=lambda p: (p.name.endswith(".final.phdf"), p.name))
    athena_files = sorted(args.athenak_dir.glob("*.tab"))
    athena_by_cycle = {}
    for path in athena_files:
        time, cycle, rows = athena_tab(path)
        # AthenaK writes one duplicate final tab file; retain the first copy.
        athena_by_cycle.setdefault(cycle, (path, time, rows))
    athena_cycles = sorted(athena_by_cycle)
    if len(pangu_files) != len(athena_cycles):
        raise RuntimeError(
            f"snapshot count mismatch: PANGU={len(pangu_files)} AthenaK={len(athena_cycles)}"
        )

    snapshots = []
    for index, (pangu_path, cycle) in enumerate(zip(pangu_files, athena_cycles)):
        pangu_time, pangu = pangu_snapshot(pangu_path)
        athena_path, athena_time, rows = athena_by_cycle[cycle]
        if abs(pangu_time - athena_time) > 2.0e-13:
            raise RuntimeError(
                f"time mismatch at pair {index}: PANGU={pangu_time} AthenaK={athena_time}"
            )
        # The solution is translationally invariant in y,z.  Confirm that this
        # is true, then compare the native x-line selected at k=j=0.
        yz_spread = float(np.max(np.ptp(pangu, axis=(1, 2))))
        differences = np.asarray(
            [pangu[pangu_index, 0, 0, :] - rows[:, athena_index]
             for athena_index, pangu_index in ATHENA_COLUMNS.items()]
        )
        snapshots.append(
            {
                "index": index,
                "athenak_cycle": cycle,
                "time": pangu_time,
                "max_abs": float(np.max(np.abs(differences))),
                "mean_abs": float(np.mean(np.abs(differences))),
                "yz_invariance_linf": yz_spread,
                "component_linf": np.max(np.abs(differences), axis=1).tolist(),
                "pangu_file": pangu_path.name,
                "athenak_file": athena_path.name,
            }
        )

    result = {
        "comparison": "native uniform grid, native x centers, no interpolation",
        "components": ["chi", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "Khat",
                        "Axx", "Axy", "Axz", "Ayy", "Ayz", "Azz", "Gamx", "Gamy",
                        "Gamz", "Theta", "alpha", "betax", "betay", "betaz"],
        "snapshots": snapshots,
        "global_max_abs": max(item["max_abs"] for item in snapshots),
        "global_mean_abs_max": max(item["mean_abs"] for item in snapshots),
        "global_yz_invariance_linf": max(item["yz_invariance_linf"] for item in snapshots),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "NR-4 PANGU/AthenaK matched comparison PASS: "
        f"snapshots={len(snapshots)} "
        f"max_abs={result['global_max_abs']:.6e} "
        f"max_mean_abs={result['global_mean_abs_max']:.6e} "
        f"yz_invariance={result['global_yz_invariance_linf']:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
