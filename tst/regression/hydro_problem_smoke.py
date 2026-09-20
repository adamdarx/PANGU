#!/usr/bin/env python3
"""Run every stage-2 Newtonian Hydro problem generator end to end."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import subprocess
import sys
import tempfile


PROBLEMS = (
    "sod",
    "hydro_advection",
    "linear_wave",
    "blast",
    "kh",
    "rt",
)


def run(command: list[str], cwd: pathlib.Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def check_profile(path: pathlib.Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing final profile: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty final profile: {path}")
    for row in rows:
        values = tuple(float(row[key]) for key in
                       ("density", "velocity1", "velocity2", "velocity3", "pressure"))
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite state in {path}")
        if values[0] <= 0.0 or values[4] <= 0.0:
            raise RuntimeError(f"non-positive thermodynamic state in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--input-dir", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    parser.add_argument("--hdf5-enabled", action="store_true")
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    executable = args.executable.resolve()
    input_dir = args.input_dir.resolve()
    for problem in PROBLEMS:
        case = pathlib.Path(tempfile.mkdtemp(prefix=f"{problem}-", dir=args.workdir))
        command = [str(executable), "-i", str(input_dir / f"{problem}.in")]
        if problem == "blast" and not args.hdf5_enabled:
            command.extend(("parthenon/output1/dt=-1", "parthenon/output2/dt=-1"))
        log = run(command, case)
        (case / "run.log").write_text(log, encoding="utf-8")
        if "Driver completed." not in log:
            raise RuntimeError(f"{problem} did not report driver completion")
        check_profile(case / "pangu-hydro-final.csv")
        if not list(case.glob("*.hst")):
            raise RuntimeError(f"{problem} did not write history output")
        if problem in ("hydro_advection", "linear_wave"):
            errors = case / "pangu-hydro-errors.dat"
            if not errors.is_file() or len(errors.read_text(encoding="utf-8").splitlines()) < 2:
                raise RuntimeError(f"{problem} did not write analytic error norms")
        if problem == "blast" and "Total number of MeshBlocks = 16" not in log:
            raise RuntimeError("blast did not exercise adaptive refinement")
        if problem == "blast" and args.hdf5_enabled:
            if not list(case.glob("*.phdf")) or not list(case.glob("*.rhdf")):
                raise RuntimeError("blast did not write HDF5 field and restart outputs")

    print(f"Hydro problem smoke PASS: {len(PROBLEMS)} problem generators")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - regression harness reports context
        print(f"Hydro problem smoke FAIL: {error}", file=sys.stderr)
        raise
