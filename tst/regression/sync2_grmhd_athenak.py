#!/usr/bin/env python3
"""Run the SYNC-2 eigenmode in PANGU and AthenaK and compare every dump.

AthenaK's linear-wave problem rewrites ``time/tlim`` through ParameterInput.
That path stores the computed period with six decimal places.  The exact time
actually used by AthenaK is therefore read from its full-precision binary
header and passed to PANGU before the pointwise comparison.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess

import h5py
import numpy as np


FIELDS = (
    "rho",
    "u1",
    "u2",
    "u3",
    "pressure",
    "B1",
    "B2",
    "B3",
    "D",
    "S1",
    "S2",
    "S3",
    "tau",
)


def run(command: list[str], directory: pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    (directory / "run.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}{completed.stderr}"
        )


def binary_time(path: pathlib.Path) -> tuple[float, int]:
    with path.open("rb") as stream:
        version = stream.readline().decode().strip()
        if version != "Athena binary output version=1.1":
            raise RuntimeError(f"unexpected AthenaK binary header in {path}: {version}")
        stream.readline()
        time_line = stream.readline().decode()
        cycle_line = stream.readline().decode()
    return float(time_line.split("=", 1)[1]), int(cycle_line.split("=", 1)[1])


def tab_time(path: pathlib.Path) -> tuple[float, int]:
    header = path.read_text().splitlines()[0]
    match = re.search(r"time=([+\-0-9.eE]+)\s+cycle=(\d+)", header)
    if match is None:
        raise RuntimeError(f"cannot parse AthenaK tab header in {path}: {header}")
    return float(match.group(1)), int(match.group(2))


def pangu_line(path: pathlib.Path, gamma: float) -> tuple[float, int, np.ndarray]:
    with h5py.File(path, "r") as data:
        time = float(data["Info"].attrs["Time"])
        cycle = int(data["Info"].attrs["NCycle"])
        order = np.argsort(np.asarray(data["VolumeLocations/x"]).reshape(-1))

        def line(field: str) -> np.ndarray:
            values = np.asarray(data[field], dtype=np.float64)
            # The eigenmode is constant in x2 and x3.  Taking one transverse
            # cell avoids counting the deliberately thin dimensions fourfold.
            return values[:, :, 0, 0, :].transpose(0, 2, 1).reshape(-1, values.shape[1])[order]

        primitive = line("mhd.prim")
        magnetic = line("mhd.b_cell")
        conserved = line("mhd.cons")
        state = np.column_stack(
            (
                primitive[:, :4],
                (gamma - 1.0) * primitive[:, 4],
                magnetic,
                conserved,
            )
        )
        divergence = float(np.max(np.abs(np.asarray(data["mhd.divb"]))))
        z4c = np.asarray(data["nr.z4c"], dtype=np.float64)
    return time, cycle, state, divergence, z4c


def athena_line(primitive: pathlib.Path, conserved: pathlib.Path) -> tuple[float, int, np.ndarray]:
    time, cycle = tab_time(primitive)
    conserved_time, conserved_cycle = tab_time(conserved)
    if cycle != conserved_cycle or abs(time - conserved_time) > 5.0e-7:
        raise RuntimeError("AthenaK primitive and conserved outputs are not synchronized")
    primitive_values = np.loadtxt(primitive, dtype=np.float64)[:, 3:11]
    conserved_values = np.loadtxt(conserved, dtype=np.float64)[:, 3:8]
    return time, cycle, np.column_stack((primitive_values, conserved_values))


def ordered_phdf(directory: pathlib.Path) -> list[pathlib.Path]:
    paths = list(directory.glob("*.phdf"))
    return sorted(paths, key=lambda path: (path.name.endswith(".final.phdf"), path.name))


def rms_l1(initial: pathlib.Path, final: pathlib.Path) -> float:
    component_errors: list[float] = []
    with h5py.File(initial, "r") as before, h5py.File(final, "r") as after:
        for field in ("mhd.cons", "mhd.b_cell"):
            left = np.asarray(before[field], dtype=np.float64)
            right = np.asarray(after[field], dtype=np.float64)
            component_errors.extend(
                float(np.mean(np.abs(right[:, component] - left[:, component])))
                for component in range(left.shape[1])
            )
    return float(np.sqrt(np.sum(np.square(component_errors))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", type=pathlib.Path, required=True)
    parser.add_argument("--athenak", type=pathlib.Path, required=True)
    parser.add_argument("--pangu-input", type=pathlib.Path, required=True)
    parser.add_argument("--athenak-input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    parser.add_argument("--gamma", type=float, default=1.333333)
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    athena_dir = workdir / "athenak"
    pangu_dir = workdir / "pangu"

    run(
        [str(args.athenak.resolve()), "-i", str(args.athenak_input.resolve())],
        athena_dir,
    )
    binary_outputs = sorted((athena_dir / "bin").glob("*.bin"))
    if len(binary_outputs) != 2:
        raise RuntimeError(f"expected AthenaK initial/final binary outputs, found {len(binary_outputs)}")
    terminal_time, terminal_cycle = binary_time(binary_outputs[-1])

    run(
        [
            str(args.pangu.resolve()),
            "-i",
            str(args.pangu_input.resolve()),
            f"parthenon/time/tlim={terminal_time:.17g}",
        ],
        pangu_dir,
    )

    pangu_outputs = ordered_phdf(pangu_dir)
    athena_primitive = sorted((athena_dir / "tab").glob("*.mhd_w_bcc.*.tab"))
    athena_conserved = sorted((athena_dir / "tab").glob("*.mhd_u_bcc.*.tab"))
    counts = (len(pangu_outputs), len(athena_primitive), len(athena_conserved))
    if len(set(counts)) != 1 or counts[0] != 11:
        raise RuntimeError(f"expected eleven synchronized outputs from each code, found {counts}")

    maximum_by_field = {field: 0.0 for field in FIELDS}
    snapshot_errors: list[dict[str, float | int]] = []
    maximum_divergence = 0.0
    initial_z4c: np.ndarray | None = None
    maximum_spacetime_change = 0.0
    scale = 1.0
    for pangu_path, primitive_path, conserved_path in zip(
        pangu_outputs, athena_primitive, athena_conserved
    ):
        pangu_time, pangu_cycle, pangu_state, divergence, z4c = pangu_line(
            pangu_path, args.gamma
        )
        athena_time, athena_cycle, athena_state = athena_line(primitive_path, conserved_path)
        if pangu_cycle != athena_cycle:
            raise RuntimeError(
                f"cycle mismatch: PANGU {pangu_cycle}, AthenaK {athena_cycle}"
            )
        # Tab headers print only six decimal places.  At the final dump use the
        # full-precision binary time already applied to PANGU.
        if abs(pangu_time - athena_time) > 5.1e-7:
            raise RuntimeError(
                f"time mismatch at cycle {pangu_cycle}: {pangu_time} vs {athena_time}"
            )
        difference = np.abs(pangu_state - athena_state)
        per_field = np.max(difference, axis=0)
        scale = max(scale, float(np.max(np.abs(athena_state))))
        for field, value in zip(FIELDS, per_field):
            maximum_by_field[field] = max(maximum_by_field[field], float(value))
        maximum_divergence = max(maximum_divergence, divergence)
        if initial_z4c is None:
            initial_z4c = z4c
        maximum_spacetime_change = max(
            maximum_spacetime_change, float(np.max(np.abs(z4c - initial_z4c)))
        )
        snapshot_errors.append(
            {
                "cycle": pangu_cycle,
                "time": pangu_time,
                "maximum_absolute_difference": float(np.max(difference)),
            }
        )

    tolerance = 256.0 * np.finfo(np.float64).eps * scale
    global_maximum = max(maximum_by_field.values())
    if global_maximum > tolerance:
        raise RuntimeError(
            f"PANGU/AthenaK pointwise error {global_maximum:.17e} exceeds "
            f"roundoff gate {tolerance:.17e}"
        )
    if maximum_divergence != 0.0:
        raise RuntimeError(f"CT divergence is nonzero: {maximum_divergence:.17e}")
    if maximum_spacetime_change != 0.0:
        raise RuntimeError(
            f"zero-source Minkowski spacetime changed: {maximum_spacetime_change:.17e}"
        )

    pangu_error = rms_l1(pangu_outputs[0], pangu_outputs[-1])
    error_file = next(athena_dir.glob("*-errs.dat"))
    athena_error = float(np.loadtxt(error_file, comments="#")[4])
    report = {
        "schema": "pangu.sync2.grmhd-athenak.v1",
        "status": "pass",
        "terminal_time": terminal_time,
        "terminal_cycle": terminal_cycle,
        "roundoff_tolerance": tolerance,
        "global_maximum_absolute_difference": global_maximum,
        "maximum_absolute_difference_by_field": maximum_by_field,
        "snapshots": snapshot_errors,
        "maximum_divergence": maximum_divergence,
        "maximum_zero_source_spacetime_change": maximum_spacetime_change,
        "pangu_rms_l1": pangu_error,
        "athenak_rms_l1_printed": athena_error,
    }
    (workdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "SYNC-2 PANGU/AthenaK PASS: "
        f"snapshots={len(snapshot_errors)} max_abs={global_maximum:.3e} "
        f"gate={tolerance:.3e} divB={maximum_divergence:.1e} "
        f"L1={pangu_error:.9e}/{athena_error:.9e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
