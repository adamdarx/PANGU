#!/usr/bin/env python3
"""Strict multi-snapshot Hydro comparison against AthenaK and AthenaPK."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import subprocess
import sys
import tempfile

import h5py
import numpy as np


FIELDS = ("density", "velocity1", "velocity2", "velocity3", "pressure")
RECONSTRUCTIONS = ("dc", "plm", "ppm")
ATHENAK_SOLVERS = {
    "ideal": ("llf", "hlle", "hllc", "roe"),
    "isothermal": ("llf", "hlle", "roe"),
}
ATHENAPK_CASES = (
    ("dc", "hlle"),
    ("plm", "hlle"),
    ("ppm", "hlle"),
    ("dc", "hllc"),
    ("plm", "hllc"),
    ("ppm", "hllc"),
)
SNAPSHOT_COUNT = 5
ABSOLUTE_TOLERANCE = 1.0e-12
RELATIVE_TOLERANCE = 1.0e-11
METADATA_TOLERANCE = 1.0e-14


def run(command: list[str], directory: pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
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
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-8000:]}"
        )


def hdf_snapshots(directory: pathlib.Path, dataset: str,
                  isothermal: bool = False) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.phdf")):
        with h5py.File(path, "r") as handle:
            info = handle["Info"].attrs
            values = np.asarray(handle[dataset][0, :, 0, 0, :], dtype=np.float64)
            if isothermal:
                values = np.vstack((values[:4], values[0]))
            snapshots.append({
                "cycle": int(info["NCycle"]),
                "dt": float(info["dt"]),
                "path": path.name,
                "time": float(info["Time"]),
                "values": values,
                "x": np.asarray(handle["VolumeLocations/x"][0], dtype=np.float64),
            })
    if len(snapshots) != SNAPSHOT_COUNT:
        raise RuntimeError(
            f"expected {SNAPSHOT_COUNT} PHDF snapshots in {directory}, got {len(snapshots)}"
        )
    return snapshots


def tab_snapshots(directory: pathlib.Path, isothermal: bool) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    for path in sorted((directory / "tab").glob("*hydro_w*.tab")):
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise RuntimeError(f"empty AthenaK tab file: {path}")
        cycle_match = re.search(r"cycle=(\d+)", lines[0])
        if cycle_match is None:
            raise RuntimeError(f"cycle metadata missing from {path}")
        x_values: list[float] = []
        rows: list[list[float]] = []
        for line in lines:
            if not line or line.startswith("#"):
                continue
            columns = line.split()
            density = float(columns[3])
            pressure = density if isothermal else 0.4 * float(columns[7])
            x_values.append(float(columns[2]))
            rows.append([
                density,
                float(columns[4]),
                float(columns[5]),
                float(columns[6]),
                pressure,
            ])
        snapshots.append({
            "cycle": int(cycle_match.group(1)),
            "path": path.name,
            "values": np.asarray(rows, dtype=np.float64).T,
            "x": np.asarray(x_values, dtype=np.float64),
        })
    if len(snapshots) != SNAPSHOT_COUNT:
        raise RuntimeError(
            f"expected {SNAPSHOT_COUNT} AthenaK tab snapshots in {directory}, "
            f"got {len(snapshots)}"
        )
    return snapshots


def history_snapshots(directory: pathlib.Path) -> list[dict[str, float]]:
    histories = sorted(directory.glob("*.hst"))
    if len(histories) != 1:
        raise RuntimeError(f"expected one history file in {directory}, got {histories}")
    rows: list[dict[str, float]] = []
    for line in histories[0].read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            values = [float(value) for value in line.split()]
            rows.append({"time": values[0], "dt": values[1]})
    if len(rows) != SNAPSHOT_COUNT:
        raise RuntimeError(
            f"expected {SNAPSHOT_COUNT} history snapshots in {histories[0]}, got {len(rows)}"
        )
    return rows


def previous_step_history(directory: pathlib.Path, snapshots: list[dict[str, object]],
                          history: list[dict[str, float]]) -> list[dict[str, float]]:
    """Convert Parthenon next-step dt metadata to AthenaK HST's completed-step meaning."""
    trajectory: dict[int, float] = {}
    pattern = re.compile(r"cycle=(\d+)\s+time=\S+\s+dt=(\S+)")
    for line in (directory / "run.log").read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match is not None:
            trajectory[int(match.group(1))] = float(match.group(2))
    result: list[dict[str, float]] = []
    for snapshot, row in zip(snapshots, history):
        cycle = int(snapshot["cycle"])
        is_terminal = abs(row["time"] - 0.02) < METADATA_TOLERANCE
        source_cycle = cycle if cycle == 0 or is_terminal else cycle - 1
        if source_cycle not in trajectory:
            raise RuntimeError(
                f"cycle {source_cycle} dt missing from trajectory log in {directory}"
            )
        result.append({"time": row["time"], "dt": trajectory[source_cycle]})
    return result


def field_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, dict[str, float]]:
    if left.shape != right.shape:
        raise RuntimeError(f"field shapes differ: {left.shape} and {right.shape}")
    result: dict[str, dict[str, float]] = {}
    for index, field in enumerate(FIELDS):
        difference = np.abs(left[index] - right[index])
        reference = np.abs(right[index])
        absolute_l1 = float(np.mean(difference))
        absolute_l2 = float(math.sqrt(float(np.mean(difference * difference))))
        absolute_linf = float(np.max(difference))
        reference_l1 = float(np.mean(reference))
        reference_l2 = float(math.sqrt(float(np.mean(reference * reference))))
        reference_linf = float(np.max(reference))
        result[field] = {
            "l1": absolute_l1,
            "l2": absolute_l2,
            "linf": absolute_linf,
            "relative_l1": absolute_l1 / max(reference_l1, 1.0e-300),
            "relative_l2": absolute_l2 / max(reference_l2, 1.0e-300),
            "relative_linf": absolute_linf / max(reference_linf, 1.0e-300),
        }
    return result


def compare_series(case: str, pair: str, left: list[dict[str, object]],
                   right: list[dict[str, object]], left_history: list[dict[str, float]],
                   right_history: list[dict[str, float]], gate: bool = True) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, (left_snapshot, right_snapshot) in enumerate(zip(left, right)):
        coordinate_linf = float(np.max(np.abs(left_snapshot["x"] - right_snapshot["x"])))
        cycle_delta = abs(int(left_snapshot["cycle"]) - int(right_snapshot["cycle"]))
        time_delta = abs(left_history[index]["time"] - right_history[index]["time"])
        dt_delta = abs(left_history[index]["dt"] - right_history[index]["dt"])
        metrics = field_metrics(left_snapshot["values"], right_snapshot["values"])
        maximum_linf = max(field["linf"] for field in metrics.values())
        maximum_relative_linf = max(field["relative_linf"] for field in metrics.values())
        passed = (
            coordinate_linf < METADATA_TOLERANCE
            and cycle_delta == 0
            and time_delta < METADATA_TOLERANCE
            and dt_delta < METADATA_TOLERANCE
            and maximum_linf < ABSOLUTE_TOLERANCE
            and maximum_relative_linf < RELATIVE_TOLERANCE
        )
        result.append({
            "case": case,
            "cycle": int(left_snapshot["cycle"]),
            "cycle_delta": cycle_delta,
            "dt": left_history[index]["dt"],
            "dt_delta": dt_delta,
            "fields": metrics,
            "gate": gate,
            "maximum_linf": maximum_linf,
            "maximum_relative_linf": maximum_relative_linf,
            "pair": pair,
            "passed": passed if gate else True,
            "reference_divergence": not passed if not gate else False,
            "snapshot": index,
            "time": left_history[index]["time"],
            "time_delta": time_delta,
            "x_linf": coordinate_linf,
        })
    return result


def pangu_command(args: argparse.Namespace, input_path: pathlib.Path, eos: str,
                  reconstruction: str, solver: str) -> list[str]:
    return [
        str(args.pangu.resolve()), "-i", str(input_path.resolve()),
        "parthenon/mesh/nghost=4", "parthenon/mesh/nx1=64",
        "parthenon/meshblock/nx1=64", "parthenon/time/tlim=0.02",
        "parthenon/time/integrator=rk2", "parthenon/time/nlim=1000",
        "hydro/cfl=0.2", f"hydro/reconstruct={reconstruction}",
        f"hydro/rsolver={solver}", "hydro/fofc=false",
        "parthenon/output1/dt=0.005", "parthenon/output1/data_format=%24.17e",
        "parthenon/output2/file_type=hdf5", "parthenon/output2/id=prim",
        "parthenon/output2/dt=0.005", "parthenon/output2/variables=hydro.prim",
        *(["problem/x0=-0.5"] if eos == "isothermal" else []),
    ]


def athenak_command(args: argparse.Namespace, input_path: pathlib.Path, eos: str,
                    reconstruction: str, solver: str) -> list[str]:
    reference_reconstruction = "ppm4" if reconstruction == "ppm" else reconstruction
    history_output = "output2" if eos == "ideal" else "output3"
    return [
        str(args.athenak.resolve()), "-i", str(input_path.resolve()),
        "mesh/nghost=4", "mesh/nx1=64", "meshblock/nx1=64", "time/tlim=0.02",
        "time/integrator=rk2", "time/nlim=1000", "time/cfl_number=0.2",
        f"hydro/reconstruct={reference_reconstruction}", f"hydro/rsolver={solver}",
        "output1/dt=0.005", "output1/data_format=%24.17e",
        f"{history_output}/dt=0.005", f"{history_output}/data_format=%24.17e",
        *(["time/evolution=dynamic"] if eos == "isothermal" else []),
    ]


def athenapk_command(args: argparse.Namespace, reconstruction: str,
                     solver: str) -> list[str]:
    return [
        str(args.athenapk.resolve()), "-i", str(args.athenapk_sod.resolve()),
        "parthenon/mesh/nghost=4", "parthenon/mesh/nx1=64",
        "parthenon/mesh/x1min=-0.5", "parthenon/mesh/x1max=0.5",
        "parthenon/meshblock/nx1=64", "problem/sod/x_discont=0.0",
        "parthenon/time/tlim=0.02", "parthenon/time/integrator=rk2",
        "parthenon/time/nlim=1000", "parthenon/time/cfl=0.2",
        f"hydro/reconstruction={reconstruction}", f"hydro/riemann={solver}",
        "hydro/first_order_flux_correct=false", "parthenon/output0/dt=0.005",
        "parthenon/output1/file_type=hst", "parthenon/output1/dt=0.005",
        "parthenon/output1/data_format=%24.17e",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=pathlib.Path)
    parser.add_argument("--pangu-sod", required=True, type=pathlib.Path)
    parser.add_argument("--pangu-advection", required=True, type=pathlib.Path)
    parser.add_argument("--athenak", required=True, type=pathlib.Path)
    parser.add_argument("--athenak-sod", required=True, type=pathlib.Path)
    parser.add_argument("--athenak-advection", required=True, type=pathlib.Path)
    parser.add_argument("--athenapk", required=True, type=pathlib.Path)
    parser.add_argument("--athenapk-sod", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(tempfile.mkdtemp(prefix="hydro-multitime-", dir=args.workdir))
    comparisons: list[dict[str, object]] = []
    cached: dict[str, tuple[list[dict[str, object]], list[dict[str, float]],
                            list[dict[str, object]], list[dict[str, float]]]] = {}

    for eos, solvers in ATHENAK_SOLVERS.items():
        for reconstruction in RECONSTRUCTIONS:
            for solver in solvers:
                case = f"{eos}-{reconstruction}-{solver}"
                case_root = root / case
                pangu_dir = case_root / "pangu"
                athenak_dir = case_root / "athenak"
                pangu_input = args.pangu_sod if eos == "ideal" else args.pangu_advection
                athenak_input = args.athenak_sod if eos == "ideal" else args.athenak_advection
                run(pangu_command(args, pangu_input, eos, reconstruction, solver), pangu_dir)
                run(athenak_command(args, athenak_input, eos, reconstruction, solver), athenak_dir)
                pangu = hdf_snapshots(pangu_dir, "hydro.prim", eos == "isothermal")
                athenak = tab_snapshots(athenak_dir, eos == "isothermal")
                pangu_history = history_snapshots(pangu_dir)
                athenak_history = history_snapshots(athenak_dir)
                pangu_completed_step_history = previous_step_history(
                    pangu_dir, pangu, pangu_history)
                comparisons.extend(compare_series(
                    case, "PANGU-AthenaK", pangu, athenak,
                    pangu_completed_step_history, athenak_history))
                cached[case] = (pangu, pangu_history, athenak, athenak_history)

    for reconstruction, solver in ATHENAPK_CASES:
        case = f"ideal-{reconstruction}-{solver}"
        pangu, pangu_history, athenak, athenak_history = cached[case]
        case_root = root / case
        athenapk_dir = case_root / "athenapk"
        run(athenapk_command(args, reconstruction, solver), athenapk_dir)
        athenapk = hdf_snapshots(athenapk_dir, "prim")
        athenapk_history = history_snapshots(athenapk_dir)
        athenapk_completed_step_history = previous_step_history(
            athenapk_dir, athenapk, athenapk_history)
        if reconstruction == "ppm":
            pangu_pk_dir = case_root / "pangu-ppmc"
            run(pangu_command(args, args.pangu_sod, "ideal", "ppmc", solver), pangu_pk_dir)
            pangu_pk = hdf_snapshots(pangu_pk_dir, "hydro.prim")
            pangu_pk_history = history_snapshots(pangu_pk_dir)
            comparisons.extend(compare_series(
                case + "-ppmc", "PANGU-AthenaPK", pangu_pk, athenapk,
                pangu_pk_history, athenapk_history))
            comparisons.extend(compare_series(
                case + "-reference-algorithm-difference", "AthenaK-AthenaPK",
                athenak, athenapk, athenak_history,
                athenapk_completed_step_history, gate=False))
        else:
            comparisons.extend(compare_series(
                case, "PANGU-AthenaPK", pangu, athenapk,
                pangu_history, athenapk_history))
            comparisons.extend(compare_series(
                case, "AthenaK-AthenaPK", athenak, athenapk,
                athenak_history, athenapk_completed_step_history))

    gated = [record for record in comparisons if record["gate"]]
    failures = [record for record in gated if not record["passed"]]
    maximum_l1 = max(
        field["l1"] for record in gated for field in record["fields"].values())
    maximum_l2 = max(
        field["l2"] for record in gated for field in record["fields"].values())
    maximum_linf = max(record["maximum_linf"] for record in gated)
    maximum_relative_linf = max(record["maximum_relative_linf"] for record in gated)
    maximum_time_delta = max(record["time_delta"] for record in gated)
    maximum_dt_delta = max(record["dt_delta"] for record in gated)
    maximum_x_delta = max(record["x_linf"] for record in gated)
    summary = {
        "absolute_tolerance_linf": ABSOLUTE_TOLERANCE,
        "athenak_algorithm_cases": 21,
        "athenapk_executable_algorithm_cases": len(ATHENAPK_CASES),
        "athenapk_llf_exclusion": (
            "AthenaPK's declared DC+LLF tight-loop path aborts on the Sod discontinuity "
            "with negative pressure before snapshot 1, including a 0.1% weak jump."
        ),
        "comparisons": comparisons,
        "failed_gated_snapshots": len(failures),
        "gated_snapshot_comparisons": len(gated),
        "maximum_dt_delta": maximum_dt_delta,
        "maximum_l1": maximum_l1,
        "maximum_l2": maximum_l2,
        "maximum_linf": maximum_linf,
        "maximum_relative_linf": maximum_relative_linf,
        "maximum_time_delta": maximum_time_delta,
        "maximum_x_delta": maximum_x_delta,
        "metadata_tolerance": METADATA_TOLERANCE,
        "relative_tolerance_linf": RELATIVE_TOLERANCE,
        "snapshots_per_case": SNAPSHOT_COUNT,
    }
    (root / "hydro-multitime-references.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {key: value for key, value in summary.items() if key != "comparisons"}
    compact["result_directory"] = str(root)
    print(json.dumps(compact, indent=2, sort_keys=True))
    if failures:
        first = failures[0]
        raise RuntimeError(
            "strict multi-time gate failed first at "
            f"{first['case']} snapshot {first['snapshot']} ({first['pair']})"
        )
    print("Hydro multi-time AthenaK/AthenaPK alignment PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - report the exact regression context
        print(f"Hydro multi-time AthenaK/AthenaPK alignment FAIL: {error}", file=sys.stderr)
        raise
