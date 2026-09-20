#!/usr/bin/env python3
"""Exercise every supported stage-2 Newtonian Hydro algorithm combination."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import subprocess
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    combinations = [
        (eos, reconstruction, solver)
        for eos, solvers in (("ideal", ("llf", "hlle", "hllc", "roe")),
                             ("isothermal", ("llf", "hlle", "roe")))
        for reconstruction in ("dc", "plm", "ppm")
        for solver in solvers
    ]
    for eos, reconstruction, solver in combinations:
        case = pathlib.Path(tempfile.mkdtemp(
            prefix=f"{eos}-{reconstruction}-{solver}-", dir=args.workdir))
        command = [
            str(args.executable.resolve()), "-i", str(args.input.resolve()),
            f"hydro/eos={eos}", f"hydro/reconstruct={reconstruction}",
            f"hydro/rsolver={solver}", "parthenon/mesh/nghost=4",
            "parthenon/mesh/nx1=32", "parthenon/meshblock/nx1=32",
            "parthenon/time/tlim=0.005", "parthenon/time/nlim=20",
            "parthenon/output1/dt=-1",
        ]
        completed = subprocess.run(command, cwd=case, check=False, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (case / "run.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"failed {eos}/{reconstruction}/{solver}:\n{completed.stdout}")
        profile = case / "pangu-hydro-final.csv"
        if not profile.is_file():
            raise RuntimeError(f"missing final profile for {eos}/{reconstruction}/{solver}")
        with profile.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise RuntimeError(f"empty profile for {eos}/{reconstruction}/{solver}")
        for row in rows:
            values = [float(row[name]) for name in
                      ("density", "velocity1", "velocity2", "velocity3", "pressure")]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"non-finite state for {eos}/{reconstruction}/{solver}")
            if float(row["density"]) <= 0.0 or float(row["pressure"]) <= 0.0:
                raise RuntimeError(f"non-positive state for {eos}/{reconstruction}/{solver}")

    print(f"Hydro algorithm matrix PASS: {len(combinations)} combinations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
