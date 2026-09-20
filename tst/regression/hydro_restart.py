#!/usr/bin/env python3
"""Prove that stage-2 Hydro state, derived fields, AMR, and HDF5 restart compose."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import subprocess
import sys
import tempfile


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    case = pathlib.Path(tempfile.mkdtemp(prefix="hydro-restart-", dir=args.workdir))
    executable = str(args.executable.resolve())
    initial = run([executable, "-i", str(args.input.resolve())], case)
    (case / "initial.log").write_text(initial, encoding="utf-8")
    if "Total number of MeshBlocks = 16" not in initial:
        raise RuntimeError("initial blast run did not refine to 16 MeshBlocks")
    if not list(case.glob("*.phdf")):
        raise RuntimeError("initial blast run did not produce PHDF output")
    restarts = sorted(case.glob("*.rhdf"))
    if not restarts:
        raise RuntimeError("initial blast run did not produce restart output")

    restarted = run(
        [executable, "-r", str(restarts[-1].resolve()), "parthenon/time/tlim=0.025"],
        case,
    )
    (case / "restart.log").write_text(restarted, encoding="utf-8")
    if "restart" not in restarted.lower() or "time=2.50e-02" not in restarted:
        raise RuntimeError("restart run did not restore state and advance to t=0.025")

    profile = case / "pangu-hydro-final.csv"
    with profile.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("restart run produced no final profile rows")
    for row in rows:
        density = float(row["density"])
        pressure = float(row["pressure"])
        if not math.isfinite(density) or not math.isfinite(pressure):
            raise RuntimeError("restart profile contains non-finite values")
        if density <= 0.0 or pressure <= 0.0:
            raise RuntimeError("restart profile contains a non-positive state")

    print(f"Hydro HDF5/AMR restart PASS: {case}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - regression harness reports context
        print(f"Hydro restart FAIL: {error}", file=sys.stderr)
        raise
