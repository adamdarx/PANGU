#!/usr/bin/env python3
"""Compare the SYNC-3 magnetized TOV trajectory with AthenaK.

AthenaK's binary reference stores variables as float32.  The gate therefore
separates binary-output quantization, the physical stellar interior, the thin
discrete stellar surface, and the artificial atmosphere instead of hiding the
surface error in a whole-domain norm.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


KINDS = {
    "mhd_w_bcc": ("mhd.prim", "mhd.b_cell"),
    "mhd_u": ("mhd.cons",),
    "z4c": ("nr.z4c",),
    "adm": ("nr.adm",),
    "mhd_divb": ("mhd.divb",),
}


def reader(source: Path) -> Any:
    path = source / "vis/python/bin_convert.py"
    spec = importlib.util.spec_from_file_location("pangu_athenak_bin_convert", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load AthenaK binary reader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pangu_snapshots(directory: Path) -> dict[float, Path]:
    result: dict[float, Path] = {}
    for path in directory.glob("*.phdf"):
        with h5py.File(path, "r") as stream:
            result[float(stream["Info"].attrs["Time"])] = path
    if not result:
        raise RuntimeError(f"no PANGU PHDF files in {directory}")
    return result


def athena_snapshots(directory: Path, kind: str, convert: Any) -> list[dict[str, Any]]:
    paths = sorted((directory / "bin").glob(f"*.{kind}.*.bin"))
    if not paths:
        raise RuntimeError(f"no AthenaK {kind} files in {directory / 'bin'}")
    return [convert.read_binary(str(path)) for path in paths]


def reorder(snapshot: dict[str, Any], stream: h5py.File, count: int) -> np.ndarray:
    mapping = {
        (int(stream["Levels"][index]), *map(int, stream["LogicalLocations"][index])): index
        for index in range(len(stream["Levels"]))
    }
    keys = [
        (int(row[3]), int(row[0]), int(row[1]), int(row[2]))
        for row in snapshot["mb_logical"]
    ]
    if set(keys) != set(mapping):
        raise RuntimeError("PANGU and AthenaK logical MeshBlock sets differ")
    destination = np.asarray([mapping[key] for key in keys], dtype=np.int64)
    values = np.stack(
        [np.asarray(snapshot["mb_data"][name], dtype=np.float64)
         for name in snapshot["var_names"][:count]],
        axis=1,
    )
    result = np.empty_like(values)
    result[destination] = values
    return result


def pangu_field(stream: h5py.File, kind: str) -> np.ndarray:
    values = [np.asarray(stream[name], dtype=np.float64) for name in KINDS[kind]]
    if kind == "mhd_divb":
        return values[0][:, None]
    return np.concatenate(values, axis=1) if len(values) > 1 else values[0]


def masked_maximum(difference: np.ndarray, mask: np.ndarray) -> float:
    per_cell = np.max(difference, axis=1)
    return float(np.max(per_cell[mask])) if np.any(mask) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu-workdir", required=True, type=Path)
    parser.add_argument("--athenak-workdir", required=True, type=Path)
    parser.add_argument("--athenak-source", required=True, type=Path)
    args = parser.parse_args()

    convert = reader(args.athenak_source.resolve())
    pangu = pangu_snapshots(args.pangu_workdir.resolve())
    athena = {
        kind: athena_snapshots(args.athenak_workdir.resolve(), kind, convert)
        for kind in KINDS
    }
    sample_count = len(athena["mhd_w_bcc"])
    if any(len(snapshots) != sample_count for snapshots in athena.values()):
        raise RuntimeError("AthenaK output streams contain different sample counts")

    maxima = {
        "initial_matter": 0.0,
        "deep_interior_primitive": 0.0,
        "material_primitive": 0.0,
        "deep_interior_conserved": 0.0,
        "z4c_float32_envelope": 0.0,
        "adm_float32_envelope": 0.0,
        "central_density_trajectory": 0.0,
        "peak_magnetic_trajectory": 0.0,
        "maximum_abs_divb": 0.0,
    }
    samples: list[dict[str, Any]] = []
    for index in range(sample_count):
        time = float(athena["mhd_w_bcc"][index]["time"])
        matched_time = min(pangu, key=lambda value: abs(value - time))
        if abs(matched_time - time) > 2.0e-14:
            raise RuntimeError(f"no PANGU sample at AthenaK time {time}")
        record: dict[str, Any] = {"time": time, "fields": {}}
        with h5py.File(pangu[matched_time], "r") as stream:
            primitive = np.asarray(stream["mhd.prim"], dtype=np.float64)
            density = primitive[:, 0]
            deep = density > 2.0e-4
            material = density > 1.0e-6
            if not np.any(deep) or not np.any(material):
                raise RuntimeError(f"empty TOV material mask at t={time}")
            for kind in KINDS:
                candidate = pangu_field(stream, kind)
                reference = reorder(athena[kind][index], stream, candidate.shape[1])
                if not np.isfinite(candidate).all():
                    raise RuntimeError(f"nonfinite PANGU {kind} at t={time}")
                difference = np.abs(candidate - reference)
                field_record = {
                    "all_maximum": float(np.max(difference)),
                    "material_maximum": masked_maximum(difference, material),
                    "deep_interior_maximum": masked_maximum(difference, deep),
                }
                record["fields"][kind] = field_record
                if index == 0 and kind in ("mhd_w_bcc", "mhd_u"):
                    maxima["initial_matter"] = max(
                        maxima["initial_matter"], field_record["deep_interior_maximum"]
                    )
                if kind == "mhd_w_bcc":
                    maxima["deep_interior_primitive"] = max(
                        maxima["deep_interior_primitive"], field_record["deep_interior_maximum"]
                    )
                    maxima["material_primitive"] = max(
                        maxima["material_primitive"], field_record["material_maximum"]
                    )
                    maxima["central_density_trajectory"] = max(
                        maxima["central_density_trajectory"],
                        abs(float(np.max(candidate[:, 0])) - float(np.max(reference[:, 0]))),
                    )
                    maxima["peak_magnetic_trajectory"] = max(
                        maxima["peak_magnetic_trajectory"],
                        abs(float(np.max(np.linalg.norm(candidate[:, 5:8], axis=1))) -
                            float(np.max(np.linalg.norm(reference[:, 5:8], axis=1)))),
                    )
                elif kind == "mhd_u":
                    maxima["deep_interior_conserved"] = max(
                        maxima["deep_interior_conserved"], field_record["deep_interior_maximum"]
                    )
                elif kind == "z4c":
                    maxima["z4c_float32_envelope"] = max(
                        maxima["z4c_float32_envelope"], field_record["deep_interior_maximum"]
                    )
                elif kind == "adm":
                    maxima["adm_float32_envelope"] = max(
                        maxima["adm_float32_envelope"], field_record["deep_interior_maximum"]
                    )
                elif kind == "mhd_divb":
                    maxima["maximum_abs_divb"] = max(
                        maxima["maximum_abs_divb"], float(np.max(np.abs(candidate))),
                        float(np.max(np.abs(reference))),
                    )
        samples.append(record)

    limits = {
        "initial_matter": 2.0e-10,
        "deep_interior_primitive": 1.0e-8,
        "material_primitive": 2.0e-6,
        "deep_interior_conserved": 2.0e-9,
        "z4c_float32_envelope": 2.0e-7,
        "adm_float32_envelope": 3.0e-7,
        "central_density_trajectory": 1.0e-9,
        "peak_magnetic_trajectory": 1.0e-8,
        "maximum_abs_divb": 1.0e-12,
    }
    failures = {name: value for name, value in maxima.items() if value > limits[name]}
    report = {
        "schema": "pangu.sync3.magnetized-tov-athenak.v1",
        "pass": not failures,
        "comparison": "same logical cells without interpolation",
        "athenak_variable_storage_bytes": 4,
        "regions": {
            "deep_interior": "rho > 2e-4",
            "material": "rho > 1e-6",
            "artificial_atmosphere": "reported but excluded from physical gates",
        },
        "maxima": maxima,
        "limits": limits,
        "samples": samples,
    }
    output = args.pangu_workdir.resolve() / "athenak_comparison.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"magnetized TOV AthenaK gate failed: {failures}")
    print(json.dumps({"pass": True, "maxima": maxima}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
