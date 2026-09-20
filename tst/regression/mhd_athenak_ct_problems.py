#!/usr/bin/env python3
"""Strict final-slice checks for two multidimensional AthenaK CT problems."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import subprocess


FIELDS = ("x", "density", "velocity1", "velocity2", "velocity3",
          "pressure", "B1", "B2", "B3")
ABS_TOLERANCE = 1.0e-12


def run(command: list[str], directory: pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=directory, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (directory / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n"
                           f"{completed.stdout[-8000:]}")


def athenak_deck(problem: str) -> str:
    if problem == "orszag_tang":
        x1min, x1max, x2min, x2max = 0.0, 1.0, 0.0, 1.0
        nx1 = nx2 = 32
        solver, tlim, slice_x2 = "llf", 0.005, 0.015625
        pgen = "pgen_name = orszag_tang"
        parameters = ""
        basename = "OrszagTang"
    else:
        x1min, x1max, x2min, x2max = -1.0, 1.0, -0.5, 0.5
        nx1, nx2 = 32, 16
        solver, tlim, slice_x2 = "hlld", 0.01, 0.03125
        pgen = ""
        parameters = "rad = 0.3\namp = 1.0e-3\niprob = 1"
        basename = "Loop"
    return f"""<job>
basename = {basename}
<mesh>
nghost = 3
nx1 = {nx1}
x1min = {x1min}
x1max = {x1max}
ix1_bc = periodic
ox1_bc = periodic
nx2 = {nx2}
x2min = {x2min}
x2max = {x2max}
ix2_bc = periodic
ox2_bc = periodic
nx3 = 1
x3min = -0.5
x3max = 0.5
ix3_bc = periodic
ox3_bc = periodic
<meshblock>
nx1 = {nx1}
nx2 = {nx2}
nx3 = 1
<time>
evolution = dynamic
integrator = rk2
cfl_number = 0.3
nlim = 100
tlim = {tlim}
ndiag = 1
<mhd>
eos = ideal
reconstruct = plm
rsolver = {solver}
gamma = 1.6666666666666667
<problem>
{pgen}
{parameters}
<output1>
file_type = tab
variable = mhd_w_bcc
data_format = %24.17e
dt = {tlim}
slice_x2 = {slice_x2}
slice_x3 = 0.0
"""


def pangu_rows(path: pathlib.Path, slice_x2: float) -> list[tuple[float, ...]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = [tuple(float(row[field]) for field in FIELDS) for row in rows
                if abs(float(row["y"]) - slice_x2) < 1.0e-14]
    return sorted(selected)


def athenak_rows(directory: pathlib.Path) -> list[tuple[float, ...]]:
    paths = sorted((directory / "tab").glob("*.tab"))
    if len(paths) != 2:
        raise RuntimeError(f"expected initial/final AthenaK tables in {directory}")
    rows = []
    for line in paths[-1].read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            values = list(map(float, line.split()[2:]))
            rows.append((values[0], values[1], values[2], values[3], values[4],
                         (2.0 / 3.0) * values[5], values[6], values[7], values[8]))
    return sorted(rows)


def compare(name: str, pangu: list[tuple[float, ...]],
            athenak: list[tuple[float, ...]]) -> dict[str, float | int | str]:
    if len(pangu) != len(athenak) or not pangu:
        raise RuntimeError(f"{name} slice row mismatch: {len(pangu)} vs {len(athenak)}")
    differences = [abs(lhs - rhs) for left, right in zip(pangu, athenak)
                   for lhs, rhs in zip(left, right)]
    metric = {
        "problem": name,
        "rows": len(pangu),
        "l1": sum(differences) / len(differences),
        "l2": math.sqrt(sum(value * value for value in differences) / len(differences)),
        "linf": max(differences),
    }
    if metric["linf"] > ABS_TOLERANCE:
        raise RuntimeError(f"{name} L-infinity={metric['linf']:.17e} exceeds gate")
    return metric


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=pathlib.Path)
    parser.add_argument("--pangu-input-dir", required=True, type=pathlib.Path)
    parser.add_argument("--athenak-built-in", required=True, type=pathlib.Path)
    parser.add_argument("--athenak-field-loop", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    configurations = {
        "orszag_tang": {
            "pangu_input": "orszag_tang.in", "nx1": 32, "nx2": 32,
            "tlim": 0.005, "solver": "llf", "slice": 0.015625,
            "athenak": args.athenak_built_in,
            "extra": [],
        },
        "field_loop": {
            "pangu_input": "field_loop.in", "nx1": 32, "nx2": 16,
            "tlim": 0.01, "solver": "hlld", "slice": 0.03125,
            "athenak": args.athenak_field_loop,
            "extra": [f"problem/vflow={math.sqrt(5.0):.17g}"],
        },
    }
    metrics = []
    for name, config in configurations.items():
        pangu_dir = args.workdir / name / "pangu"
        athenak_dir = args.workdir / name / "athenak"
        run([
            str(args.pangu.resolve()), "-i",
            str((args.pangu_input_dir / str(config["pangu_input"])).resolve()),
            f"parthenon/mesh/nx1={config['nx1']}",
            f"parthenon/mesh/nx2={config['nx2']}",
            f"parthenon/meshblock/nx1={config['nx1']}",
            f"parthenon/meshblock/nx2={config['nx2']}",
            f"parthenon/time/tlim={config['tlim']}", "parthenon/time/nlim=100",
            "parthenon/output1/dt=1.0", "mhd/reconstruct=plm",
            f"mhd/rsolver={config['solver']}", "mhd/cfl=0.3",
            "mhd/gamma=1.6666666666666667", *config["extra"],
        ], pangu_dir)
        athenak_dir.mkdir(parents=True, exist_ok=True)
        deck = athenak_dir / f"{name}.athinput"
        deck.write_text(athenak_deck(name), encoding="utf-8")
        run([str(pathlib.Path(config["athenak"]).resolve()), "-i", str(deck.resolve())],
            athenak_dir)
        metrics.append(compare(
            name, pangu_rows(pangu_dir / "pangu-mhd-final.csv", float(config["slice"])),
            athenak_rows(athenak_dir)))

    report = {"absolute_tolerance": ABS_TOLERANCE, "passed": True, "metrics": metrics}
    (args.workdir / "mhd-ct-problem-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for metric in metrics:
        print(f"PANGU-AthenaK {metric['problem']}: rows={metric['rows']} "
              f"L1={metric['l1']:.17e} L2={metric['l2']:.17e} "
              f"Linf={metric['linf']:.17e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
