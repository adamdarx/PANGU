#!/usr/bin/env python3
"""Compact stage-3 MHD problem and algorithm matrix."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import subprocess


def run(command: list[str], directory: pathlib.Path) -> list[dict[str, str]]:
    directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=directory, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (directory / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n"
                           f"{completed.stdout[-6000:]}")
    profile = directory / "pangu-mhd-final.csv"
    if not profile.exists():
        raise RuntimeError(f"missing final profile in {directory}")
    with profile.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty final profile in {directory}")
    for row in rows:
        values = [float(value) for value in row.values()]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite MHD state in {directory}")
        if float(row["density"]) <= 0.0 or float(row["pressure"]) <= 0.0:
            raise RuntimeError(f"non-positive primitive state in {directory}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=pathlib.Path, required=True)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    executable = str(args.executable.resolve())
    inputs = args.input_dir.resolve()

    for solver in ("llf", "hlle", "hlld"):
        rows = run([executable, "-i", str(inputs / "brio_wu.in"),
                    "parthenon/mesh/nx1=64", "parthenon/meshblock/nx1=64",
                    "parthenon/time/tlim=0.01", "parthenon/time/nlim=100",
                    "parthenon/output1/dt=1.0", f"mhd/rsolver={solver}"],
                   args.workdir / f"brio-{solver}")
        if max(abs(float(row["divB"])) for row in rows) > 1.0e-13:
            raise RuntimeError(f"Brio-Wu divB gate failed for {solver}")

    for reconstruction in ("dc", "plm", "ppm", "ppmc"):
        rows = run([executable, "-i", str(inputs / "brio_wu.in"),
                    "parthenon/mesh/nx1=64", "parthenon/meshblock/nx1=64",
                    "parthenon/mesh/nghost=4", "parthenon/time/tlim=0.005",
                    "parthenon/time/nlim=100", "parthenon/output1/dt=1.0",
                    "mhd/rsolver=hlld", f"mhd/reconstruct={reconstruction}"],
                   args.workdir / f"brio-hlld-{reconstruction}")
        if max(abs(float(row["divB"])) for row in rows) > 1.0e-13:
            raise RuntimeError(f"Brio-Wu divB gate failed for {reconstruction}")

    problem_overrides = {
        "cpaw": ["parthenon/mesh/nx1=32", "parthenon/meshblock/nx1=32",
                 "parthenon/time/tlim=0.01"],
        "field_loop": ["parthenon/mesh/nx1=32", "parthenon/mesh/nx2=16",
                       "parthenon/meshblock/nx1=32", "parthenon/meshblock/nx2=16",
                       "parthenon/time/tlim=0.01"],
        "orszag_tang": ["parthenon/mesh/nx1=32", "parthenon/mesh/nx2=32",
                        "parthenon/meshblock/nx1=32", "parthenon/meshblock/nx2=32",
                        "parthenon/time/tlim=0.005"],
    }
    for problem, overrides in problem_overrides.items():
        rows = run([executable, "-i", str(inputs / f"{problem}.in"),
                    "parthenon/time/nlim=100", "parthenon/output1/dt=1.0",
                    *overrides], args.workdir / problem)
        max_divb = max(abs(float(row["divB"])) for row in rows)
        if max_divb > 2.0e-12:
            raise RuntimeError(f"{problem} divB={max_divb:.17e} exceeds gate")

    print("PANGU stage-3 MHD problem matrix PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
