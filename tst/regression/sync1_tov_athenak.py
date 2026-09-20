#!/usr/bin/env python3
"""Compare synchronized TOV evolution with AthenaK at matched RK2 times."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ATHENA_FIELDS = {
    "mhd_w": ("dens", "velx", "vely", "velz", "press"),
    "mhd_u": ("dens", "mom1", "mom2", "mom3", "ener"),
    "z4c": (
        "z4c_chi", "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz",
        "z4c_gzz", "z4c_Khat", "z4c_Axx", "z4c_Axy", "z4c_Axz", "z4c_Ayy",
        "z4c_Ayz", "z4c_Azz", "z4c_Gamx", "z4c_Gamy", "z4c_Gamz", "z4c_Theta",
        "z4c_alpha", "z4c_betax", "z4c_betay", "z4c_betaz",
    ),
    "adm": (
        "adm_gxx", "adm_gxy", "adm_gxz", "adm_gyy", "adm_gyz", "adm_gzz",
        "adm_Kxx", "adm_Kxy", "adm_Kxz", "adm_Kyy", "adm_Kyz", "adm_Kzz",
        "adm_psi4",
    ),
}
PANGU_FIELDS = {
    "mhd_w": "hydro.prim",
    "mhd_u": "hydro.cons",
    "z4c": "nr.z4c",
    "adm": "nr.adm",
}


def load_binary_reader(source: Path) -> Any:
    path = source / "vis/python/bin_convert.py"
    spec = importlib.util.spec_from_file_location("pangu_athenak_bin_convert", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load AthenaK binary reader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(command: list[str], directory: Path, log_name: str, completion: str) -> None:
    completed = subprocess.run(
        command,
        cwd=directory,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (directory / log_name).write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or completion not in completed.stdout:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-8000:]}"
        )


def logical_map(stream: h5py.File) -> dict[tuple[int, int, int, int], int]:
    levels = np.asarray(stream["Levels"], dtype=np.int64)
    locations = np.asarray(stream["LogicalLocations"], dtype=np.int64)
    return {
        (int(levels[index]), *map(int, locations[index])): index
        for index in range(levels.size)
    }


def reorder(snapshot: dict[str, Any], stream: h5py.File, names: tuple[str, ...]) -> np.ndarray:
    mapping = logical_map(stream)
    keys = [
        (int(row[3]), int(row[0]), int(row[1]), int(row[2]))
        for row in np.asarray(snapshot["mb_logical"])
    ]
    if set(keys) != set(mapping):
        raise RuntimeError("PANGU and AthenaK logical MeshBlock sets differ")
    destination = np.asarray([mapping[key] for key in keys], dtype=np.int64)
    source = np.stack(
        [np.asarray(snapshot["mb_data"][name], dtype=np.float64) for name in names], axis=1
    )
    result = np.empty_like(source)
    result[destination] = source
    return result


def norm(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    left = reference[mask]
    right = candidate[mask]
    difference = right - left
    absolute = np.abs(difference)
    return {
        "linf": float(np.max(absolute)),
        "relative_l1": float(np.sum(absolute) / max(np.sum(np.abs(left)), 1.0e-300)),
        "rms": float(np.sqrt(np.mean(difference * difference))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=Path)
    parser.add_argument("--athenak", required=True, type=Path)
    parser.add_argument("--athenak-source", required=True, type=Path)
    parser.add_argument("--pangu-input", required=True, type=Path)
    parser.add_argument("--athenak-input", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--v-pert", type=float, default=0.0)
    parser.add_argument("--tlim", type=float, default=0.5)
    parser.add_argument("--output-dt", type=float, default=0.1)
    parser.add_argument("--expected-samples", type=int, default=4)
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    pangu_dir = workdir / "pangu"
    athenak_dir = workdir / "athenak"
    pangu_dir.mkdir(parents=True)
    athenak_dir.mkdir(parents=True)
    pangu_command = [
        str(args.pangu.resolve()), "-i", str(args.pangu_input.resolve()), "-d", ".",
        f"parthenon/time/tlim={args.tlim:.17g}", "parthenon/time/nlim=100000",
        f"parthenon/output1/dt={args.output_dt:.17g}",
        f"problem/v_pert={args.v_pert:.17g}",
    ]
    run(
        pangu_command,
        pangu_dir,
        "run.log",
        "Driver completed.",
    )
    athenak_command = [
        str(args.athenak.resolve()), "-i", str(args.athenak_input.resolve()), "-d", ".",
        f"time/tlim={args.tlim:.17g}", "time/nlim=100000",
        f"problem/v_pert={args.v_pert:.17g}",
        *[f"output{output}/dt={args.output_dt:.17g}" for output in range(1, 6)],
    ]
    run(
        athenak_command,
        athenak_dir,
        "run.log",
        "Terminating on time limit",
    )

    bin_convert = load_binary_reader(args.athenak_source.resolve())
    pangu_outputs = sorted(pangu_dir.glob("*.phdf"))
    athena_u = sorted((athenak_dir / "bin").glob("*.mhd_u.*.bin"))
    if (len(pangu_outputs) != args.expected_samples or
            len(athena_u) != args.expected_samples):
        raise RuntimeError(
            f"expected {args.expected_samples} matched samples, got PANGU={len(pangu_outputs)}, "
            f"AthenaK={len(athena_u)}"
        )

    samples: list[dict[str, Any]] = []
    maxima = {
        "conserved_relative_l1": 0.0,
        "core_primitive_linf": 0.0,
        "z4c_linf": 0.0,
        "adm_linf": 0.0,
        "density_trajectory_linf": 0.0,
        "surface_velocity_linf": 0.0,
        "deep_core_velocity_linf": 0.0,
    }
    perturbed = abs(args.v_pert) > 0.0
    core_density_threshold = 5.0e-4 if perturbed else 1.0e-4
    for pangu_path, athena_u_path in zip(pangu_outputs, athena_u, strict=True):
        snapshots = {
            kind: bin_convert.read_binary(
                str(athena_u_path).replace("mhd_u", kind)
            )
            for kind in ATHENA_FIELDS
        }
        with h5py.File(pangu_path, "r") as stream:
            pangu_time = float(stream["Info"].attrs["Time"])
            athena_time = float(snapshots["mhd_u"]["time"])
            if not math.isclose(pangu_time, athena_time, abs_tol=2.0e-15):
                raise RuntimeError(f"sample time mismatch {pangu_time} != {athena_time}")
            primitive = np.asarray(stream["hydro.prim"], dtype=np.float64)
            reference_primitive = reorder(snapshots["mhd_w"], stream, ATHENA_FIELDS["mhd_w"])
            core = primitive[:, 0] > core_density_threshold
            if not np.any(core):
                raise RuntimeError("TOV comparison core mask is empty")
            maxima["density_trajectory_linf"] = max(
                maxima["density_trajectory_linf"],
                abs(float(np.max(reference_primitive[:, 0])) - float(np.max(primitive[:, 0]))),
            )
            surface = ((primitive[:, 0] > 1.0e-4) &
                       (primitive[:, 0] <= 5.0e-4))
            velocity = np.moveaxis(primitive[:, 1:4], 1, -1)
            reference_velocity = np.moveaxis(reference_primitive[:, 1:4], 1, -1)
            if np.any(surface):
                maxima["surface_velocity_linf"] = max(
                    maxima["surface_velocity_linf"],
                    float(np.max(np.abs(velocity[surface] - reference_velocity[surface]))),
                )
            maxima["deep_core_velocity_linf"] = max(
                maxima["deep_core_velocity_linf"],
                float(np.max(np.abs(velocity[core] - reference_velocity[core]))),
            )
            record: dict[str, Any] = {"time": pangu_time, "fields": {}}
            for kind, names in ATHENA_FIELDS.items():
                reference = reorder(snapshots[kind], stream, names)
                candidate = np.asarray(stream[PANGU_FIELDS[kind]], dtype=np.float64)
                field_records = []
                for component, name in enumerate(names):
                    all_mask = np.ones(reference[:, component].shape, dtype=bool)
                    selected = core if kind == "mhd_w" else all_mask
                    value = norm(reference[:, component], candidate[:, component], selected)
                    field_records.append({"name": name, **value})
                    if kind == "mhd_u":
                        maxima["conserved_relative_l1"] = max(
                            maxima["conserved_relative_l1"], value["relative_l1"]
                        )
                    elif kind == "mhd_w":
                        maxima["core_primitive_linf"] = max(
                            maxima["core_primitive_linf"], value["linf"]
                        )
                    elif kind == "z4c":
                        maxima["z4c_linf"] = max(maxima["z4c_linf"], value["linf"])
                    elif kind == "adm":
                        maxima["adm_linf"] = max(maxima["adm_linf"], value["linf"])
                record["fields"][kind] = field_records
            samples.append(record)

    if perturbed:
        limits = {
            # A radial perturbation deliberately advects the discrete stellar
            # surface.  The dense core and spacetime remain close to AthenaK's
            # float32 reference output; the surface velocity is reported and
            # bounded separately rather than hidden in a global norm.
            "conserved_relative_l1": 2.5e-4,
            "core_primitive_linf": 1.8e-6,
            "z4c_linf": 5.0e-7,
            "adm_linf": 1.2e-6,
            "density_trajectory_linf": 2.0e-9,
            "surface_velocity_linf": 1.0e-5,
            "deep_core_velocity_linf": 1.8e-6,
        }
    else:
        limits = {
            # AthenaK binary fields are stored as float32.  These bounds cover that
            # output quantization plus the thin stellar-surface recovery layer.
            "conserved_relative_l1": 5.0e-6,
            "core_primitive_linf": 2.0e-8,
            "z4c_linf": 1.5e-7,
            "adm_linf": 2.0e-7,
            "density_trajectory_linf": 2.0e-10,
            "surface_velocity_linf": 2.0e-8,
            "deep_core_velocity_linf": 2.0e-8,
        }
    failures = {key: maxima[key] for key in limits if maxima[key] > limits[key]}
    report = {
        "schema": "pangu.sync1.tov-athenak-multitime.v2",
        "pass": not failures,
        "comparison": "same logical cells, no interpolation",
        "athenak_storage_precision": "float32",
        "radial_velocity_perturbation": args.v_pert,
        "core_density_threshold": core_density_threshold,
        "maxima": maxima,
        "limits": limits,
        "samples": samples,
    }
    (workdir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(
            f"AthenaK TOV multitime gate failed: {failures}; "
            f"all maxima={maxima}"
        )
    for pattern in ("*.phdf", "*.xdmf", "*.bin"):
        for path in workdir.rglob(pattern):
            path.unlink()
    print(json.dumps({key: value for key, value in report.items() if key != "samples"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
