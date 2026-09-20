#!/usr/bin/env python3
"""Compare every stage-2 PANGU Hydro algorithm combination with AthenaK."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import subprocess
import sys
import tempfile


RECONSTRUCTIONS = ("dc", "plm", "ppm")
SOLVERS = {
    "ideal": ("llf", "hlle", "hllc", "roe"),
    "isothermal": ("llf", "hlle", "roe"),
}


def run(command: list[str], cwd: pathlib.Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (cwd / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def pangu_profile(path: pathlib.Path) -> list[tuple[float, float, float, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            (float(row["x"]), float(row["density"]), float(row["velocity1"]),
             float(row["pressure"]))
            for row in csv.DictReader(stream)
        ]


def athenak_profile(path: pathlib.Path, eos: str) -> list[tuple[float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            columns = line.split()
            density = float(columns[3])
            pressure = 0.4 * float(columns[7]) if eos == "ideal" else density
            rows.append((float(columns[2]), density, float(columns[4]), pressure))
    return rows


def compare(left: list[tuple[float, float, float, float]],
            right: list[tuple[float, float, float, float]]) -> dict[str, float]:
    if len(left) != 64 or len(right) != 64:
        raise RuntimeError(f"expected 64 profile rows, got {len(left)} and {len(right)}")
    for pangu_row, athenak_row in zip(left, right):
        if abs(pangu_row[0] - athenak_row[0]) > 1.0e-6:
            raise RuntimeError("PANGU and AthenaK cell centers differ")
    result: dict[str, float] = {}
    for field, index in (("density", 1), ("velocity", 2), ("pressure", 3)):
        differences = [abs(p[index] - a[index]) for p, a in zip(left, right)]
        reference = [abs(a[index]) for a in right]
        absolute_l1 = sum(differences) / len(differences)
        absolute_l2 = math.sqrt(sum(value * value for value in differences) / len(differences))
        absolute_linf = max(differences)
        reference_l1 = sum(reference) / len(reference)
        reference_l2 = math.sqrt(sum(value * value for value in reference) / len(reference))
        reference_linf = max(reference)
        result[f"{field}_l1"] = absolute_l1
        result[f"{field}_l2"] = absolute_l2
        result[f"{field}_linf"] = absolute_linf
        result[f"{field}_relative_l1"] = absolute_l1 / max(reference_l1, 1.0e-300)
        result[f"{field}_relative_l2"] = absolute_l2 / max(reference_l2, 1.0e-300)
        result[f"{field}_relative_linf"] = absolute_linf / max(reference_linf, 1.0e-300)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=pathlib.Path)
    parser.add_argument("--pangu-sod", required=True, type=pathlib.Path)
    parser.add_argument("--pangu-advection", required=True, type=pathlib.Path)
    parser.add_argument("--athenak", required=True, type=pathlib.Path)
    parser.add_argument("--athenak-sod", required=True, type=pathlib.Path)
    parser.add_argument("--athenak-advection", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(tempfile.mkdtemp(prefix="athenak-hydro-matrix-", dir=args.workdir))
    results: dict[str, dict[str, float]] = {}
    for eos, solvers in SOLVERS.items():
        for reconstruction in RECONSTRUCTIONS:
            for solver in solvers:
                name = f"{eos}-{reconstruction}-{solver}"
                pangu_dir = root / name / "pangu"
                athenak_dir = root / name / "athenak"
                pangu_dir.mkdir(parents=True)
                athenak_dir.mkdir(parents=True)
                common_pangu = [
                    "parthenon/mesh/nghost=4", "parthenon/mesh/nx1=64",
                    "parthenon/meshblock/nx1=64", "parthenon/time/tlim=0.02",
                    "parthenon/time/nlim=1000", "hydro/cfl=0.2",
                    f"hydro/reconstruct={reconstruction}", f"hydro/rsolver={solver}",
                    "parthenon/output1/dt=-1",
                ]
                if eos == "ideal":
                    pangu_input = args.pangu_sod
                    athenak_input = args.athenak_sod
                    pangu_extra = []
                    athenak_extra = []
                else:
                    pangu_input = args.pangu_advection
                    athenak_input = args.athenak_advection
                    pangu_extra = ["problem/x0=-0.5"]
                    athenak_extra = ["time/evolution=dynamic"]
                run([
                    str(args.pangu.resolve()), "-i", str(pangu_input.resolve()),
                    *common_pangu, *pangu_extra,
                ], pangu_dir)
                athenak_reconstruction = "ppm4" if reconstruction == "ppm" else reconstruction
                run([
                    str(args.athenak.resolve()), "-i", str(athenak_input.resolve()),
                    "mesh/nghost=4", "mesh/nx1=64", "meshblock/nx1=64",
                    "time/tlim=0.02", "time/nlim=1000", "time/cfl_number=0.2",
                    f"hydro/reconstruct={athenak_reconstruction}",
                    f"hydro/rsolver={solver}", "output1/dt=0.02",
                    "output1/data_format=%24.17e", "output2/dt=0.02",
                    *athenak_extra,
                ], athenak_dir)
                tabs = sorted((athenak_dir / "tab").glob("*hydro_w*.tab"))
                if not tabs:
                    raise RuntimeError(f"AthenaK profile missing for {name}")
                results[name] = compare(
                    pangu_profile(pangu_dir / "pangu-hydro-final.csv"),
                    athenak_profile(tabs[-1], eos),
                )

    maximum_l1 = max(
        result[f"{field}_l1"]
        for result in results.values()
        for field in ("density", "velocity", "pressure")
    )
    maximum_l2 = max(
        result[f"{field}_l2"]
        for result in results.values()
        for field in ("density", "velocity", "pressure")
    )
    maximum_linf = max(
        result[f"{field}_linf"]
        for result in results.values()
        for field in ("density", "velocity", "pressure")
    )
    maximum_relative_linf = max(
        result[f"{field}_relative_linf"]
        for result in results.values()
        for field in ("density", "velocity", "pressure")
    )
    summary = {
        "absolute_tolerance_linf": 1.0e-12,
        "combinations": len(results),
        "maximum_l1": maximum_l1,
        "maximum_l2": maximum_l2,
        "maximum_linf": maximum_linf,
        "maximum_relative_linf": maximum_relative_linf,
        "relative_tolerance_linf": 1.0e-11,
        "results": results,
    }
    (root / "athenak-hydro-matrix.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if len(results) != 21:
        raise RuntimeError(f"expected 21 legal algorithm combinations, got {len(results)}")
    if maximum_linf >= summary["absolute_tolerance_linf"]:
        raise RuntimeError(
            f"cross-framework matrix maximum absolute Linf {maximum_linf} exceeds "
            f"{summary['absolute_tolerance_linf']}"
        )
    if maximum_relative_linf >= summary["relative_tolerance_linf"]:
        raise RuntimeError(
            f"cross-framework matrix maximum relative Linf {maximum_relative_linf} exceeds "
            f"{summary['relative_tolerance_linf']}"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"AthenaK Hydro algorithm alignment PASS: {len(results)} combinations")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - regression harness reports context
        print(f"AthenaK Hydro algorithm alignment FAIL: {error}", file=sys.stderr)
        raise
