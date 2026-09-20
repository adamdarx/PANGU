#!/usr/bin/env python3
"""Run and quantify a matched Sod problem in PANGU, AthenaK, and AthenaPK."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Callable


GAMMA = 1.4
FINAL_TIME = 0.2
FIELDS = ("density", "velocity", "pressure")


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


def pressure_function(pressure: float, density: float, initial_pressure: float,
                      sound_speed: float) -> tuple[float, float]:
    if pressure > initial_pressure:
        a_coefficient = 2.0 / ((GAMMA + 1.0) * density)
        b_coefficient = (GAMMA - 1.0) / (GAMMA + 1.0) * initial_pressure
        root = math.sqrt(a_coefficient / (pressure + b_coefficient))
        value = (pressure - initial_pressure) * root
        derivative = root * (1.0 - 0.5 * (pressure - initial_pressure) /
                             (pressure + b_coefficient))
        return value, derivative
    ratio = pressure / initial_pressure
    exponent = (GAMMA - 1.0) / (2.0 * GAMMA)
    value = 2.0 * sound_speed / (GAMMA - 1.0) * (ratio**exponent - 1.0)
    derivative = (
        ratio ** (-(GAMMA + 1.0) / (2.0 * GAMMA)) / (density * sound_speed)
    )
    return value, derivative


def star_state() -> tuple[float, float]:
    density_left, pressure_left, velocity_left = 1.0, 1.0, 0.0
    density_right, pressure_right, velocity_right = 0.125, 0.1, 0.0
    sound_left = math.sqrt(GAMMA * pressure_left / density_left)
    sound_right = math.sqrt(GAMMA * pressure_right / density_right)
    guess = max(
        1.0e-12,
        0.5 * (pressure_left + pressure_right) -
        0.125 * (velocity_right - velocity_left) *
        (density_left + density_right) * (sound_left + sound_right),
    )
    pressure = guess
    for _ in range(50):
        f_left, df_left = pressure_function(
            pressure, density_left, pressure_left, sound_left)
        f_right, df_right = pressure_function(
            pressure, density_right, pressure_right, sound_right)
        updated = pressure - (
            f_left + f_right + velocity_right - velocity_left
        ) / (df_left + df_right)
        updated = max(updated, 1.0e-12)
        if abs(updated - pressure) <= 1.0e-13 * (updated + pressure):
            pressure = updated
            break
        pressure = updated
    f_left, _ = pressure_function(pressure, density_left, pressure_left, sound_left)
    f_right, _ = pressure_function(pressure, density_right, pressure_right, sound_right)
    velocity = 0.5 * (velocity_left + velocity_right + f_right - f_left)
    return pressure, velocity


def exact_sod(x: float) -> dict[str, float]:
    left = (1.0, 0.0, 1.0)
    right = (0.125, 0.0, 0.1)
    pressure_star, velocity_star = star_state()
    similarity = x / FINAL_TIME
    density_left, velocity_left, pressure_left = left
    density_right, velocity_right, pressure_right = right
    sound_left = math.sqrt(GAMMA * pressure_left / density_left)
    sound_right = math.sqrt(GAMMA * pressure_right / density_right)
    ratio_constant = (GAMMA - 1.0) / (GAMMA + 1.0)

    if similarity <= velocity_star:
        if pressure_star > pressure_left:
            shock = velocity_left - sound_left * math.sqrt(
                (GAMMA + 1.0) / (2.0 * GAMMA) * pressure_star / pressure_left +
                (GAMMA - 1.0) / (2.0 * GAMMA))
            if similarity <= shock:
                density, velocity, pressure = left
            else:
                ratio = pressure_star / pressure_left
                density = (
                    density_left * (ratio + ratio_constant) /
                    (ratio_constant * ratio + 1.0)
                )
                velocity, pressure = velocity_star, pressure_star
        else:
            sound_star = sound_left * (pressure_star / pressure_left) ** (
                (GAMMA - 1.0) / (2.0 * GAMMA))
            head = velocity_left - sound_left
            tail = velocity_star - sound_star
            if similarity <= head:
                density, velocity, pressure = left
            elif similarity >= tail:
                density = density_left * (pressure_star / pressure_left) ** (1.0 / GAMMA)
                velocity, pressure = velocity_star, pressure_star
            else:
                velocity = 2.0 / (GAMMA + 1.0) * (
                    sound_left + 0.5 * (GAMMA - 1.0) * velocity_left + similarity)
                sound = 2.0 / (GAMMA + 1.0) * (
                    sound_left + 0.5 * (GAMMA - 1.0) * (velocity_left - similarity))
                density = density_left * (sound / sound_left) ** (2.0 / (GAMMA - 1.0))
                pressure = pressure_left * (sound / sound_left) ** (
                    2.0 * GAMMA / (GAMMA - 1.0))
    else:
        if pressure_star > pressure_right:
            shock = velocity_right + sound_right * math.sqrt(
                (GAMMA + 1.0) / (2.0 * GAMMA) * pressure_star / pressure_right +
                (GAMMA - 1.0) / (2.0 * GAMMA))
            if similarity >= shock:
                density, velocity, pressure = right
            else:
                ratio = pressure_star / pressure_right
                density = (
                    density_right * (ratio + ratio_constant) /
                    (ratio_constant * ratio + 1.0)
                )
                velocity, pressure = velocity_star, pressure_star
        else:
            sound_star = sound_right * (pressure_star / pressure_right) ** (
                (GAMMA - 1.0) / (2.0 * GAMMA))
            head = velocity_right + sound_right
            tail = velocity_star + sound_star
            if similarity >= head:
                density, velocity, pressure = right
            elif similarity <= tail:
                density = density_right * (pressure_star / pressure_right) ** (1.0 / GAMMA)
                velocity, pressure = velocity_star, pressure_star
            else:
                velocity = 2.0 / (GAMMA + 1.0) * (
                    -sound_right + 0.5 * (GAMMA - 1.0) * velocity_right + similarity)
                sound = 2.0 / (GAMMA + 1.0) * (
                    sound_right - 0.5 * (GAMMA - 1.0) * (velocity_right - similarity))
                density = density_right * (sound / sound_right) ** (2.0 / (GAMMA - 1.0))
                pressure = pressure_right * (sound / sound_right) ** (
                    2.0 * GAMMA / (GAMMA - 1.0))
    return {"density": density, "velocity": velocity, "pressure": pressure}


def read_pangu_profile(path: pathlib.Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {
                "x": float(row["x"]),
                "density": float(row["density"]),
                "velocity": float(row["velocity1"]),
                "pressure": float(row["pressure"]),
            }
            for row in csv.DictReader(stream)
        ]


def read_athenak_profile(path: pathlib.Path) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        rows.append({
            "x": float(columns[2]),
            "density": float(columns[3]),
            "velocity": float(columns[4]),
            "pressure": (GAMMA - 1.0) * float(columns[7]),
        })
    return rows


def l1_errors(rows: list[dict[str, float]],
              reference: Callable[[float], dict[str, float]]) -> dict[str, float]:
    return {
        field: sum(abs(row[field] - reference(row["x"])[field]) for row in rows) / len(rows)
        for field in FIELDS
    }


def pairwise_l1(left: list[dict[str, float]],
                right: list[dict[str, float]]) -> dict[str, float]:
    if len(left) != len(right):
        raise RuntimeError("profile lengths do not match")
    for left_row, right_row in zip(left, right):
        if abs(left_row["x"] - right_row["x"]) > 1.0e-6:
            raise RuntimeError("profile coordinates do not match")
    return {
        field: sum(abs(left_row[field] - right_row[field])
                   for left_row, right_row in zip(left, right)) / len(left)
        for field in FIELDS
    }


def read_last_history(path: pathlib.Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8").splitlines()
    labels: dict[int, str] = {}
    data: list[float] | None = None
    for line in text:
        if line.startswith("#"):
            for index, label in re.findall(r"\[(\d+)\]=([^\s]+)", line):
                labels[int(index) - 1] = label
        elif line.strip():
            data = [float(value) for value in line.split()]
    if data is None:
        raise RuntimeError(f"history contains no data: {path}")
    return {label: data[index] for index, label in labels.items()}


def pick(history: dict[str, float], *names: str) -> float:
    for name in names:
        if name in history:
            return history[name]
    raise RuntimeError(f"none of {names} found in history columns {tuple(history)}")


def global_state(history: dict[str, float]) -> dict[str, float]:
    return {
        "mass": pick(history, "hydro_mass", "mass"),
        "momentum": pick(history, "hydro_momentum_1", "1-mom"),
        "energy": pick(history, "hydro_energy", "tot-E"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=pathlib.Path)
    parser.add_argument("--pangu-input", required=True, type=pathlib.Path)
    parser.add_argument("--athenak", required=True, type=pathlib.Path)
    parser.add_argument("--athenak-input", required=True, type=pathlib.Path)
    parser.add_argument("--athenapk", required=True, type=pathlib.Path)
    parser.add_argument("--athenapk-input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(tempfile.mkdtemp(prefix="sod-three-way-", dir=args.workdir))
    directories = {name: root / name for name in ("pangu", "athenak", "athenapk")}
    for directory in directories.values():
        directory.mkdir()

    pangu_log = run([
        str(args.pangu.resolve()), "-i", str(args.pangu_input.resolve()),
        "parthenon/meshblock/nx1=128",
    ], directories["pangu"])
    athenak_log = run([
        str(args.athenak.resolve()), "-i", str(args.athenak_input.resolve()),
        "mesh/nx1=128", "meshblock/nx1=128", "time/tlim=0.2",
        "time/cfl_number=0.4", "time/nlim=10000", "hydro/reconstruct=dc",
        "hydro/rsolver=hlle", "output1/dt=0.2", "output1/data_format=%24.17e",
        "output2/dt=0.2",
    ], directories["athenak"])
    athenapk_log = run([
        str(args.athenapk.resolve()), "-i", str(args.athenapk_input.resolve()),
        "parthenon/mesh/nx1=128", "parthenon/mesh/x1min=-0.5",
        "parthenon/mesh/x1max=0.5", "parthenon/meshblock/nx1=128",
        "problem/sod/x_discont=0.0", "parthenon/time/tlim=0.2",
        "parthenon/time/integrator=rk2", "parthenon/time/cfl=0.4",
        "hydro/reconstruction=dc", "hydro/riemann=hlle",
        "parthenon/output0/file_type=hst", "parthenon/output0/dt=0.2",
    ], directories["athenapk"])
    for name, log in (("PANGU", pangu_log), ("AthenaK", athenak_log),
                      ("AthenaPK", athenapk_log)):
        if "135" not in log or "2.000000" not in log:
            raise RuntimeError(f"{name} did not complete the matched 135-cycle run")

    pangu_profile = read_pangu_profile(directories["pangu"] / "pangu-hydro-final.csv")
    athenak_tabs = sorted((directories["athenak"] / "tab").glob("*.tab"))
    if not athenak_tabs:
        raise RuntimeError("AthenaK did not write its final tabular profile")
    athenak_profile = read_athenak_profile(athenak_tabs[-1])
    exact = {
        "pressure_star": star_state()[0],
        "velocity_star": star_state()[1],
    }
    profile_errors = {
        "PANGU": l1_errors(pangu_profile, exact_sod),
        "AthenaK": l1_errors(athenak_profile, exact_sod),
        "PANGU_minus_AthenaK": pairwise_l1(pangu_profile, athenak_profile),
    }

    histories = {
        "PANGU": global_state(read_last_history(next(directories["pangu"].glob("*.hst")))),
        "AthenaK": global_state(read_last_history(next(directories["athenak"].glob("*.hst")))),
        "AthenaPK": global_state(read_last_history(next(directories["athenapk"].glob("*.hst")))),
    }
    expected = {"mass": 0.5625, "momentum": 0.18, "energy": 1.375}
    conservation_errors = {
        code: {quantity: abs(values[quantity] - expected[quantity])
               for quantity in expected}
        for code, values in histories.items()
    }
    result = {
        "configuration": {
            "cells": 128,
            "blocks": 1,
            "domain": [-0.5, 0.5],
            "final_time": FINAL_TIME,
            "gamma": GAMMA,
            "integrator": "rk2",
            "cfl": 0.4,
            "reconstruction": "donor-cell",
            "riemann_solver": "HLLE",
            "boundary": "outflow",
        },
        "exact_star_state": exact,
        "profile_l1": profile_errors,
        "global_integrals": histories,
        "global_absolute_errors": conservation_errors,
    }
    (root / "sod-three-way.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# Matched Sod comparison",
        "",
        "All three executables used 128 cells, one block, CFL 0.4, donor-cell ",
        "reconstruction, HLLE, gamma 1.4, outflow boundaries, and t=0.2.",
        "",
        "| Framework | density L1 vs exact | velocity L1 vs exact | pressure L1 vs exact |",
        "|---|---:|---:|---:|",
        f"| PANGU | {profile_errors['PANGU']['density']:.9e} | "
        f"{profile_errors['PANGU']['velocity']:.9e} | "
        f"{profile_errors['PANGU']['pressure']:.9e} |",
        f"| AthenaK | {profile_errors['AthenaK']['density']:.9e} | "
        f"{profile_errors['AthenaK']['velocity']:.9e} | "
        f"{profile_errors['AthenaK']['pressure']:.9e} |",
        "",
        "AthenaPK's no-HDF5 reference build contributes the independently reduced global ",
        "integrals; PANGU and AthenaK additionally contribute cell profiles.",
        "",
        "| Framework | mass | x-momentum | total energy |",
        "|---|---:|---:|---:|",
    ]
    for code, values in histories.items():
        markdown.append(
            f"| {code} | {values['mass']:.9e} | {values['momentum']:.9e} | "
            f"{values['energy']:.9e} |")
    markdown.extend((
        "",
        "PANGU−AthenaK profile L1: " + ", ".join(
            f"{field}={profile_errors['PANGU_minus_AthenaK'][field]:.9e}"
            for field in FIELDS) + ".",
        "",
    ))
    (root / "sod-three-way.md").write_text("\n".join(markdown), encoding="utf-8")

    if max(profile_errors["PANGU"].values()) >= 5.0e-2:
        raise RuntimeError("PANGU profile exceeds the stage-2 exact-solution tolerance")
    if max(profile_errors["PANGU_minus_AthenaK"].values()) >= 1.0e-12:
        raise RuntimeError("PANGU and AthenaK profiles exceed the strict stage-2 tolerance")
    if any(error >= 1.0e-10 for code in conservation_errors.values() for error in code.values()):
        raise RuntimeError("a framework failed the matched global-integral tolerance")

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Three-way Sod comparison PASS: {root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - regression harness reports context
        print(f"Three-way Sod comparison FAIL: {error}", file=sys.stderr)
        raise
