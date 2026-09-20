#!/usr/bin/env python3
"""Five-snapshot CPAW comparison against AthenaK CT and AthenaPK GLM-MHD."""

from __future__ import annotations

import argparse
import glob
import h5py
import json
import math
import pathlib
import re
import subprocess

import numpy as np


NX1 = 64
NX2 = 16
NX3 = 16
GAMMA = 5.0 / 3.0
TLIM = 0.014
SNAPSHOTS = 5
ABS_TOLERANCE = 1.0e-12
REL_TOLERANCE = 5.0e-11
METADATA_TOLERANCE = 2.0e-14
FIELDS = ("density", "velocity1", "velocity2", "velocity3", "pressure",
          "B1", "B2", "B3")


def run(command: list[str], directory: pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=directory, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (directory / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n"
                           f"{completed.stdout[-8000:]}")


def pangu_command(executable: pathlib.Path, input_path: pathlib.Path,
                  magnetic_amplitude: float, velocity_amplitude: float) -> list[str]:
    return [
        str(executable.resolve()), "-i", str(input_path.resolve()),
        f"parthenon/mesh/nx1={NX1}", f"parthenon/mesh/nx2={NX2}",
        f"parthenon/mesh/nx3={NX3}", f"parthenon/meshblock/nx1={NX1}",
        f"parthenon/meshblock/nx2={NX2}", f"parthenon/meshblock/nx3={NX3}",
        "parthenon/mesh/x2min=0.0", "parthenon/mesh/x2max=1.0",
        "parthenon/mesh/x3min=0.0", "parthenon/mesh/x3max=1.0",
        f"parthenon/time/tlim={TLIM:.17g}", "parthenon/time/nlim=100",
        "parthenon/time/integrator=rk2", "mhd/reconstruct=plm", "mhd/rsolver=hlle",
        "mhd/gamma=1.6666666666666667", "mhd/cfl=0.3", "mhd/fofc=false",
        f"problem/b_perp={magnetic_amplitude:.17g}",
        f"problem/v_perp={velocity_amplitude:.17g}", "parthenon/output1/dt=0.001",
        "parthenon/output2/file_type=hdf5", "parthenon/output2/id=prim",
        "parthenon/output2/dt=0.001",
        "parthenon/output2/variables=mhd.prim,mhd.b_cell,mhd.divb",
    ]


def athenak_command(executable: pathlib.Path, input_path: pathlib.Path) -> list[str]:
    return [
        str(executable.resolve()), "-i", str(input_path.resolve()),
        f"mesh/nx1={NX1}", f"mesh/nx2={NX2}", f"mesh/nx3={NX3}",
        f"meshblock/nx1={NX1}", f"meshblock/nx2={NX2}", f"meshblock/nx3={NX3}",
        "mesh/x1max=1.0", "mesh/x2max=1.0", "mesh/x3max=1.0",
        f"time/tlim={TLIM:.17g}", "time/nlim=100", "time/integrator=rk2",
        "time/cfl_number=0.3", "problem/along_x1=true", "mhd/reconstruct=plm",
        "mhd/rsolver=hlle", "mhd/gamma=1.6666666666666667",
        "output1/dt=0.001", "output1/variable=mhd_w_bcc",
        "output1/data_format=%24.17e", "output2/dt=1.0", "output3/dt=0.001",
        "output3/data_format=%24.17e",
    ]


def athenapk_command(executable: pathlib.Path, input_path: pathlib.Path) -> list[str]:
    return [
        str(executable.resolve()), "-i", str(input_path.resolve()),
        f"parthenon/mesh/nx1={NX1}", f"parthenon/mesh/nx2={NX2}",
        f"parthenon/mesh/nx3={NX3}", f"parthenon/meshblock/nx1={NX1}",
        f"parthenon/meshblock/nx2={NX2}", f"parthenon/meshblock/nx3={NX3}",
        "parthenon/mesh/x1min=0.0", "parthenon/mesh/x1max=1.0",
        "parthenon/mesh/x2min=0.0", "parthenon/mesh/x2max=1.0",
        "parthenon/mesh/x3min=0.0", "parthenon/mesh/x3max=1.0",
        "parthenon/time/integrator=rk2", "parthenon/time/cfl=0.3",
        f"parthenon/time/tlim={TLIM:.17g}", "parthenon/time/nlim=100",
        "hydro/reconstruction=plm", "hydro/riemann=hlle",
        "hydro/gamma=1.6666666666666667", "problem/cpaw/ang_2=0.0",
        "problem/cpaw/ang_3=0.0", "parthenon/output0/dt=0.001",
    ]


def pangu_snapshots(directory: pathlib.Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path_string in sorted(glob.glob(str(directory / "*.phdf"))):
        path = pathlib.Path(path_string)
        with h5py.File(path, "r") as handle:
            primitive = np.asarray(handle["mhd.prim"][0, :, 0, 0, :])
            magnetic = np.asarray(handle["mhd.b_cell"][0, :, 0, 0, :])
            result.append({
                "cycle": int(handle["Info"].attrs["NCycle"]),
                "time": float(handle["Info"].attrs["Time"]),
                "values": np.vstack((primitive, magnetic)),
                "x": np.asarray(handle["VolumeLocations/x"][0]),
                "max_divb": float(np.max(np.abs(handle["mhd.divb"][...]))),
                "path": path.name,
            })
    if len(result) != SNAPSHOTS:
        raise RuntimeError(f"expected {SNAPSHOTS} PANGU snapshots, got {len(result)}")
    return result


def athenak_snapshots(directory: pathlib.Path) -> list[dict[str, object]]:
    history = []
    history_path = directory / "CPAW.mhd.hst"
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            history.append(float(line.split()[0]))
    paths = sorted((directory / "tab").glob("*mhd_w_bcc*.tab"))
    if len(paths) != SNAPSHOTS or len(history) != SNAPSHOTS:
        raise RuntimeError("AthenaK did not produce five tab/history snapshots")
    result: list[dict[str, object]] = []
    for path, time in zip(paths, history):
        lines = path.read_text(encoding="utf-8").splitlines()
        cycle_match = re.search(r"cycle=(\d+)", lines[0])
        if cycle_match is None:
            raise RuntimeError(f"cycle metadata missing from {path}")
        rows = [list(map(float, line.split()[2:])) for line in lines
                if line.strip() and not line.startswith("#")]
        data = np.asarray(rows).T
        values = np.vstack((data[1:5], (GAMMA - 1.0) * data[5], data[6:9]))
        result.append({"cycle": int(cycle_match.group(1)), "time": time,
                       "values": values, "x": data[0], "path": path.name})
    return result


def athenapk_snapshots(directory: pathlib.Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path_string in sorted(glob.glob(str(directory / "*.phdf"))):
        path = pathlib.Path(path_string)
        with h5py.File(path, "r") as handle:
            conserved = np.asarray(handle["cons"][0, :, 0, 0, :])
            density = conserved[0]
            velocity = conserved[1:4] / density
            magnetic = conserved[5:8]
            pressure = (GAMMA - 1.0) * (
                conserved[4] - 0.5 * np.sum(conserved[1:4] ** 2, axis=0) / density
                - 0.5 * np.sum(magnetic ** 2, axis=0))
            result.append({
                "cycle": int(handle["Info"].attrs["NCycle"]),
                "time": float(handle["Info"].attrs["Time"]),
                "values": np.vstack((density, velocity, pressure, magnetic)),
                "x": np.asarray(handle["VolumeLocations/x"][0]), "path": path.name,
            })
    if len(result) != SNAPSHOTS:
        raise RuntimeError(f"expected {SNAPSHOTS} AthenaPK snapshots, got {len(result)}")
    return result


def compare(pair: str, left: list[dict[str, object]],
            right: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        difference = np.abs(lhs["values"] - rhs["values"])
        field_metrics = {}
        for field_index, field in enumerate(FIELDS):
            delta = difference[field_index]
            scale = np.max(np.abs(rhs["values"][field_index]))
            field_metrics[field] = {
                "l1": float(np.mean(delta)),
                "l2": float(np.sqrt(np.mean(delta * delta))),
                "linf": float(np.max(delta)),
                "relative_linf": float(np.max(delta) / max(float(scale), 1.0e-300)),
            }
        maximum = float(np.max(difference))
        relative = max(metric["relative_linf"] for metric in field_metrics.values())
        x_linf = float(np.max(np.abs(lhs["x"] - rhs["x"])))
        time_delta = abs(float(lhs["time"]) - float(rhs["time"]))
        cycle_delta = abs(int(lhs["cycle"]) - int(rhs["cycle"]))
        passed = (maximum < ABS_TOLERANCE and relative < REL_TOLERANCE
                  and x_linf < METADATA_TOLERANCE and time_delta < METADATA_TOLERANCE
                  and cycle_delta == 0)
        records.append({"pair": pair, "snapshot": index, "cycle": int(lhs["cycle"]),
                        "time": float(lhs["time"]), "cycle_delta": cycle_delta,
                        "time_delta": time_delta, "x_linf": x_linf,
                        "maximum_linf": maximum, "maximum_relative_linf": relative,
                        "fields": field_metrics, "passed": passed})
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", type=pathlib.Path, required=True)
    parser.add_argument("--pangu-input", type=pathlib.Path, required=True)
    parser.add_argument("--athenak", type=pathlib.Path, required=True)
    parser.add_argument("--athenak-input", type=pathlib.Path, required=True)
    parser.add_argument("--athenapk", type=pathlib.Path, required=True)
    parser.add_argument("--athenapk-input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    pangu_ak = args.workdir / "pangu-athenak"
    athenak = args.workdir / "athenak"
    pangu_apk = args.workdir / "pangu-athenapk"
    athenapk = args.workdir / "athenapk"
    run(pangu_command(args.pangu, args.pangu_input, 0.1, 0.1), pangu_ak)
    run(athenak_command(args.athenak, args.athenak_input), athenak)
    # AthenaPK stores the vector-potential curl at cell centers.  The cosine factor
    # converts PANGU's face curl to that identical discrete cell-centered amplitude;
    # v_perp keeps the physical Alfvén velocity unchanged.
    compatible_bperp = 0.1 * math.cos(math.pi / NX1)
    run(pangu_command(args.pangu, args.pangu_input, compatible_bperp, 0.1), pangu_apk)
    run(athenapk_command(args.athenapk, args.athenapk_input), athenapk)

    pangu_ak_data = pangu_snapshots(pangu_ak)
    pangu_apk_data = pangu_snapshots(pangu_apk)
    records = compare("PANGU-AthenaK", pangu_ak_data, athenak_snapshots(athenak))
    records += compare("PANGU-AthenaPK", pangu_apk_data,
                       athenapk_snapshots(athenapk))
    max_divb = max(max(item["max_divb"] for item in pangu_ak_data),
                   max(item["max_divb"] for item in pangu_apk_data))
    passed = all(record["passed"] for record in records) and max_divb < 1.0e-13
    report = {"absolute_tolerance": ABS_TOLERANCE, "passed": passed,
              "maximum_abs_divb": max_divb, "records": records,
              "relative_tolerance": REL_TOLERANCE, "snapshot_count": SNAPSHOTS}
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "mhd-multitime-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for pair in ("PANGU-AthenaK", "PANGU-AthenaPK"):
        selected = [record for record in records if record["pair"] == pair]
        print(f"{pair}: snapshots={len(selected)} "
              f"max_abs={max(record['maximum_linf'] for record in selected):.17e} "
              f"max_rel={max(record['maximum_relative_linf'] for record in selected):.17e}")
    print(f"PANGU max |divB|={max_divb:.17e}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
