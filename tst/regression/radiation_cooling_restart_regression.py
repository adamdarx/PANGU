#!/usr/bin/env python3
"""Validate decomposition and restart invariance of target-thickness cooling."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import tempfile

import h5py
import numpy as np


DATASETS = (
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


def run(command: list[str], directory: pathlib.Path, log_name: str) -> str:
    completed = subprocess.run(
        command,
        cwd=directory,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (directory / log_name).write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-8000:]}"
        )
    return completed.stdout


def final_phdf(directory: pathlib.Path) -> pathlib.Path:
    matches = sorted(directory.glob("*.prim.final.phdf"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one final PHDF in {directory}, found {matches}")
    return matches[0]


def restart_final(directory: pathlib.Path) -> pathlib.Path:
    matches = sorted(directory.glob("*.restart.final.rhdf"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one final RHDF in {directory}, found {matches}")
    return matches[0]


def stitched(path: pathlib.Path, name: str) -> np.ndarray:
    with h5py.File(path, "r") as stream:
        values = np.asarray(stream[name])
        logical = np.asarray(stream["LogicalLocations"])
    if values.shape[0] == 1:
        return values[0]
    block_nz, block_ny, block_nx = values.shape[-3:]
    global_shape = values.shape[1:-3] + (
        (int(logical[:, 2].max()) + 1) * block_nz,
        (int(logical[:, 1].max()) + 1) * block_ny,
        (int(logical[:, 0].max()) + 1) * block_nx,
    )
    result = np.empty(global_shape, dtype=values.dtype)
    for block, (lx1, lx2, lx3) in enumerate(logical):
        result[
            ...,
            lx3 * block_nz : (lx3 + 1) * block_nz,
            lx2 * block_ny : (lx2 + 1) * block_ny,
            lx1 * block_nx : (lx1 + 1) * block_nx,
        ] = values[block]
    return result


def compare_decompositions(single: pathlib.Path, multi: pathlib.Path) -> float:
    maximum_scaled = 0.0
    for name in DATASETS:
        reference = stitched(single, name)
        candidate = stitched(multi, name)
        if reference.shape != candidate.shape:
            raise RuntimeError(f"shape mismatch for {name}: {reference.shape} != {candidate.shape}")
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise RuntimeError(f"non-finite data in {name}")
        scale = np.maximum(1.0, np.abs(reference))
        field_scaled = float(np.max(np.abs(candidate - reference) / scale, initial=0.0))
        maximum_scaled = max(maximum_scaled, field_scaled)
        # Block-interface reductions perturb the conserved state only at
        # roundoff (observed below 3e-18).  Primitive recovery can amplify that
        # perturbation in the low-density atmosphere, so gate the derived
        # primitive independently while retaining the strict gate elsewhere.
        tolerance = 2.0e-11 if name == "mhd.prim" else 2.0e-12
        if field_scaled > tolerance:
            raise RuntimeError(
                f"single-/multi-block scaled difference in {name} "
                f"{field_scaled:.17e} exceeds {tolerance:.1e}"
            )
    with h5py.File(multi, "r") as stream:
        cooled = np.asarray(stream["radiation.cooling_mask"]) > 0.5
        removed = np.asarray(stream["radiation.cumulative_removed_energy"])
    if not cooled.any() or float(np.max(removed, initial=0.0)) <= 0.0:
        raise RuntimeError("decomposition test never activated cooling")
    return maximum_scaled


def compare_restart(continuous: pathlib.Path, restarted: pathlib.Path) -> tuple[float, float]:
    maximum_absolute = 0.0
    active_primitive_maximum = 0.0
    with h5py.File(continuous, "r") as baseline, h5py.File(restarted, "r") as candidate:
        if baseline["Info"].attrs["Time"] != candidate["Info"].attrs["Time"]:
            raise RuntimeError("continuous and restarted final times differ")
        if baseline["Info"].attrs["NCycle"] != candidate["Info"].attrs["NCycle"]:
            raise RuntimeError("continuous and restarted cycle counts differ")
        for name in DATASETS:
            reference = np.asarray(baseline[name])
            resumed = np.asarray(candidate[name])
            if not np.isfinite(reference).all() or not np.isfinite(resumed).all():
                raise RuntimeError(f"non-finite restart data in {name}")
            absolute = np.abs(reference - resumed)
            difference = float(np.max(absolute, initial=0.0))
            maximum_absolute = max(maximum_absolute, difference)
            if name == "mhd.prim":
                # A roundoff perturbation below 3e-17 in the conserved state is
                # amplified by C2P only in the density-floor atmosphere.  Keep
                # a separate strict gate on cells eligible for cooling.
                active = reference[:, 0, ...] > 1.0e-6
                for component in range(reference.shape[1]):
                    if active.any():
                        active_primitive_maximum = max(
                            active_primitive_maximum,
                            float(np.max(absolute[:, component, ...][active], initial=0.0)),
                        )
                if difference > 2.0e-10 or active_primitive_maximum > 2.0e-12:
                    raise RuntimeError(
                        "restart primitive mismatch exceeds the atmosphere/active-fluid gates: "
                        f"all={difference:.17e} active={active_primitive_maximum:.17e}"
                    )
            elif difference > 2.0e-14:
                raise RuntimeError(
                    f"restart mismatch in {name}: max={difference:.17e} exceeds 2e-14"
                )
    return maximum_absolute, active_primitive_maximum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    parser.add_argument("--nx3", type=int, default=1)
    args = parser.parse_args()
    if args.nx3 < 1 or (args.nx3 > 1 and args.nx3 % 2 != 0):
        raise ValueError("--nx3 must be one or a positive even integer")
    multi_nx3 = 1 if args.nx3 == 1 else args.nx3 // 2

    args.workdir.mkdir(parents=True, exist_ok=True)
    case = pathlib.Path(tempfile.mkdtemp(prefix="radiation-rc3-", dir=args.workdir))
    executable = str(args.executable.resolve())
    input_path = str(args.input.resolve())
    base = [
        executable,
        "-i",
        input_path,
        "parthenon/mesh/nx1=64",
        "parthenon/mesh/nx2=32",
        f"parthenon/mesh/nx3={args.nx3}",
        "parthenon/mesh/refinement=none",
        "parthenon/mesh/nghost=2",
        "parthenon/time/nlim=100",
        "parthenon/time/dt_force=0.01",
        "mhd/reconstruct=plm",
        "problem/pert_amp=0.0",
        "radiation/start_time=0.0",
        "radiation/ramp_time=0.0",
        "radiation/h_target=0.1",
        "radiation/beta_cool=0.6283185307179586",
        "parthenon/output1/dt=1.0",
        "parthenon/output2/dt=1.0",
        "parthenon/output2/variables=mhd.prim,mhd.cons,mhd.b_cell,mhd.divb,"
        "radiation.cooling_rate,radiation.target_internal_energy,radiation.cooling_mask,"
        "radiation.cooling_fraction,radiation.cooling_time,"
        "radiation.cumulative_removed_energy,radiation.cumulative_boundary_energy",
        "parthenon/output2/single_precision_output=false",
        "parthenon/output2/hdf5_compression_level=0",
        "parthenon/output3/dt=1.0",
        "parthenon/output3/hdf5_compression_level=0",
    ]

    single = case / "single"
    multi = case / "multi"
    continuous = case / "continuous"
    split = case / "split"
    for directory in (single, multi, continuous, split):
        directory.mkdir()

    run(
        [
            *base,
            "parthenon/meshblock/nx1=64",
            "parthenon/meshblock/nx2=32",
            f"parthenon/meshblock/nx3={args.nx3}",
            "parthenon/time/tlim=0.02",
        ],
        single,
        "run.log",
    )
    run(
        [
            *base,
            "parthenon/meshblock/nx1=16",
            "parthenon/meshblock/nx2=16",
            f"parthenon/meshblock/nx3={multi_nx3}",
            "parthenon/time/tlim=0.02",
        ],
        multi,
        "run.log",
    )
    scaled = compare_decompositions(final_phdf(single), final_phdf(multi))

    multiblock = [
        *base,
        "parthenon/meshblock/nx1=16",
        "parthenon/meshblock/nx2=16",
        f"parthenon/meshblock/nx3={multi_nx3}",
    ]
    run([*multiblock, "parthenon/time/tlim=0.04"], continuous, "run.log")
    run([*multiblock, "parthenon/time/tlim=0.02"], split, "initial.log")
    checkpoint = split / "checkpoint_t002.rhdf"
    shutil.copy2(restart_final(split), checkpoint)
    with h5py.File(checkpoint, "r") as stream:
        if stream["Info"].attrs["Time"] != 0.02 or stream["Info"].attrs["NCycle"] != 2:
            raise RuntimeError("the saved midpoint checkpoint has unexpected metadata")
        if "radiation.cumulative_removed_energy" not in stream:
            raise RuntimeError("checkpoint omitted cumulative removed radiation energy")
        if "radiation.cumulative_boundary_energy" not in stream:
            raise RuntimeError("checkpoint omitted cumulative boundary energy")
    restart_log = run(
        [executable, "-r", str(checkpoint.resolve()), "parthenon/time/tlim=0.04"],
        split,
        "restart.log",
    )
    if "Var: radiation.cumulative_removed_energy:1" not in restart_log:
        raise RuntimeError("restart did not restore cumulative removed radiation energy")
    if "Var: radiation.cumulative_boundary_energy:1" not in restart_log:
        raise RuntimeError("restart did not restore cumulative boundary energy")
    restart_maximum, active_primitive_maximum = compare_restart(
        final_phdf(continuous), final_phdf(split)
    )

    print(
        "PANGU radiation RC-3 PASS: "
        f"decomposition_scaled_max={scaled:.17e} "
        f"restart_max_abs={restart_maximum:.17e} "
        f"restart_active_primitive_max_abs={active_primitive_maximum:.17e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
