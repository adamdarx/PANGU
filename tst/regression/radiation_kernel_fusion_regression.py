#!/usr/bin/env python3
"""Prove that the fused radiation update is byte-equivalent to its reference."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile

import h5py
import numpy as np


PHDF_DATASETS = (
    "mhd.b_cell",
    "mhd.cons",
    "mhd.divb",
    "mhd.prim",
    "radiation.cooling_fraction",
    "radiation.cooling_mask",
    "radiation.cooling_rate",
    "radiation.cooling_time",
    "radiation.cumulative_boundary_energy",
    "radiation.cumulative_removed_energy",
    "radiation.target_internal_energy",
)

RHDF_DATASETS = (
    "mhd.cons",
    "mhd.b_face",
    "radiation.cumulative_boundary_energy",
    "radiation.cumulative_removed_energy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-executable", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-executable", required=True, type=pathlib.Path)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--nlim", type=int, default=4)
    return parser.parse_args()


def run(executable: pathlib.Path, input_path: pathlib.Path, directory: pathlib.Path, nlim: int) -> None:
    variables = ",".join(PHDF_DATASETS)
    command = [
        str(executable.resolve()),
        "-i",
        str(input_path.resolve()),
        f"parthenon/time/nlim={nlim}",
        "problem/pert_amp=0.0",
        "radiation/start_time=0.0",
        "radiation/ramp_time=0.0",
        "parthenon/output1/dt=1.0e9",
        "parthenon/output2/dt=1.0e9",
        f"parthenon/output2/variables={variables}",
        "parthenon/output2/single_precision_output=false",
        "parthenon/output2/hdf5_compression_level=0",
        "parthenon/output3/dt=1.0e9",
        "parthenon/output3/hdf5_compression_level=0",
    ]
    completed = subprocess.run(
        command,
        cwd=directory,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (directory / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed with {completed.returncode}\n{completed.stdout[-8000:]}"
        )


def unique(directory: pathlib.Path, pattern: str) -> pathlib.Path:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {paths}")
    return paths[0]


def compare(
    reference: pathlib.Path, candidate: pathlib.Path, datasets: tuple[str, ...]
) -> dict[str, float]:
    differences: dict[str, float] = {}
    with h5py.File(reference, "r") as left, h5py.File(candidate, "r") as right:
        if float(left["Info"].attrs["Time"]) != float(right["Info"].attrs["Time"]):
            raise RuntimeError(f"time mismatch between {reference.name} and {candidate.name}")
        if int(left["Info"].attrs["NCycle"]) != int(right["Info"].attrs["NCycle"]):
            raise RuntimeError(f"cycle mismatch between {reference.name} and {candidate.name}")
        for name in datasets:
            if name not in left or name not in right:
                raise RuntimeError(f"dataset {name} is missing from the equivalence comparison")
            lhs = np.asarray(left[name])
            rhs = np.asarray(right[name])
            if lhs.shape != rhs.shape:
                raise RuntimeError(f"shape mismatch for {name}: {lhs.shape} != {rhs.shape}")
            difference = float(np.max(np.abs(lhs - rhs), initial=0.0))
            differences[name] = difference
            if not np.array_equal(lhs, rhs):
                raise RuntimeError(
                    f"fused update is not byte-equivalent for {name}: max={difference:.17e}"
                )
    return differences


def main() -> int:
    args = parse_args()
    if args.nlim < 2:
        raise ValueError("--nlim must exercise at least two complete RK steps")
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    report_path = (args.output or workdir / "radiation_kernel_fusion_regression.json").resolve()
    with tempfile.TemporaryDirectory(prefix="rc6-fusion-", dir=workdir) as temporary:
        root = pathlib.Path(temporary)
        reference_dir = root / "reference"
        candidate_dir = root / "candidate"
        reference_dir.mkdir()
        candidate_dir.mkdir()
        run(args.reference_executable, args.input, reference_dir, args.nlim)
        run(args.candidate_executable, args.input, candidate_dir, args.nlim)
        phdf = compare(
            unique(reference_dir, "*.prim.final.phdf"),
            unique(candidate_dir, "*.prim.final.phdf"),
            PHDF_DATASETS,
        )
        rhdf = compare(
            unique(reference_dir, "*.restart.final.rhdf"),
            unique(candidate_dir, "*.restart.final.rhdf"),
            RHDF_DATASETS,
        )
        logs = {
            "reference": (reference_dir / "run.log").read_text(encoding="utf-8"),
            "candidate": (candidate_dir / "run.log").read_text(encoding="utf-8"),
        }

    report = {
        "schema_version": 1,
        "status": "pass",
        "nlim": args.nlim,
        "reference_executable": str(args.reference_executable.resolve()),
        "candidate_executable": str(args.candidate_executable.resolve()),
        "phdf_max_abs": phdf,
        "rhdf_max_abs": rhdf,
        "global_max_abs": max((*phdf.values(), *rhdf.values()), default=0.0),
        "logs": logs,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "PANGU radiation RC-6 fusion PASS: "
        f"global_max_abs={report['global_max_abs']:.17e} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
