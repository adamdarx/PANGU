#!/usr/bin/env python3
"""Measure second-order convergence of the stage-3 circular Alfvén wave."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import subprocess


RESOLUTIONS = (32, 64, 128)


def run(executable: pathlib.Path, input_path: pathlib.Path, directory: pathlib.Path,
        resolution: int) -> tuple[float, float]:
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable.resolve()), "-i", str(input_path.resolve()),
        f"parthenon/mesh/nx1={resolution}",
        f"parthenon/meshblock/nx1={resolution}",
        "parthenon/time/tlim=1.0", "parthenon/time/nlim=10000",
        "parthenon/output1/dt=2.0",
    ]
    completed = subprocess.run(command, cwd=directory, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (directory / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n"
                           f"{completed.stdout[-8000:]}")

    with (directory / "pangu-mhd-final.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != resolution:
        raise RuntimeError(f"resolution {resolution} wrote {len(rows)} rows")
    face_curl_amplitude = math.sin(math.pi / resolution) / (math.pi / resolution)
    errors: list[float] = []
    for row in rows:
        x = float(row["x"])
        phase = 2.0 * math.pi * x
        exact = (1.0, 0.0, -0.1 * math.sin(phase), -0.1 * math.cos(phase),
                 0.1, 1.0, 0.1 * face_curl_amplitude * math.sin(phase),
                 0.1 * face_curl_amplitude * math.cos(phase))
        numerical = tuple(float(row[name]) for name in
                          ("density", "velocity1", "velocity2", "velocity3",
                           "pressure", "B1", "B2", "B3"))
        errors.extend(abs(value - reference)
                      for value, reference in zip(numerical, exact))
        if abs(float(row["divB"])) > 1.0e-13:
            raise RuntimeError(f"resolution {resolution} violates divB gate")
    return sum(errors) / len(errors), max(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    parser.add_argument("--resolutions", nargs="+", type=int,
                        default=list(RESOLUTIONS))
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    resolutions = tuple(args.resolutions)
    errors = [run(args.executable, args.input, args.workdir / f"n{resolution}", resolution)
              for resolution in resolutions]
    orders = [math.log(errors[index][0] / errors[index + 1][0], 2.0)
              for index in range(len(errors) - 1)]
    passed = all(errors[index + 1][0] < errors[index][0]
                 for index in range(len(errors) - 1)) and min(orders) > 1.75
    report = {
        "passed": passed,
        "resolutions": list(resolutions),
        "l1": [value[0] for value in errors],
        "linf": [value[1] for value in errors],
        "orders": orders,
    }
    (args.workdir / "cpaw-convergence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PANGU CPAW convergence: " + ", ".join(
        f"N={resolution} L1={error[0]:.17e} Linf={error[1]:.17e}"
        for resolution, error in zip(resolutions, errors)))
    print("observed orders: " + ", ".join(f"{order:.8f}" for order in orders))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
