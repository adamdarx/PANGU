#!/usr/bin/env python3
"""Run the EH-1 CKS/Dynamic GPU transport, restart, PPM4, and SMR gates."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile

import h5py
import numpy as np


def run(command: list[str], directory: pathlib.Path, log_name: str) -> str:
    result = subprocess.run(command, cwd=directory, text=True, capture_output=True)
    text = result.stdout + result.stderr
    (directory / log_name).write_text(text, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed; see {directory / log_name}")
    return text


def electron_consistency(path: pathlib.Path) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as data:
        electron_cons = data["electrons.cons"][...]
        electron_prim = data["electrons.prim"][...]
        fluid_density = data["mhd.cons"][:, 0:1, ...]
    if not np.isfinite(electron_cons).all() or not np.isfinite(electron_prim).all():
        raise RuntimeError(f"{path} contains a non-finite electron state")
    scale = max(float(np.max(np.abs(electron_cons))), np.finfo(np.float64).tiny)
    relative = float(np.max(np.abs(electron_cons - fluid_density * electron_prim)) / scale)
    if relative > 5.0e-15:
        raise RuntimeError(f"{path} violates U_K=D*K: relative residual={relative:.17e}")
    return electron_cons.sum(axis=(0, 2, 3, 4)), relative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--periodic-input", required=True, type=pathlib.Path)
    parser.add_argument("--disabled-input", required=True, type=pathlib.Path)
    parser.add_argument("--smr-input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    executable = str(args.executable.resolve())
    args.workdir.mkdir(parents=True, exist_ok=True)
    case = pathlib.Path(tempfile.mkdtemp(prefix="eh1-electrons-", dir=args.workdir))
    periodic = case / "periodic"
    restarted = case / "restarted"
    disabled = case / "disabled"
    ppm = case / "ppm"
    smr = case / "smr"
    for directory in (periodic, restarted, disabled, ppm, smr):
        directory.mkdir()

    run([executable, "-i", str(args.periodic_input.resolve())], periodic, "run.log")
    initial = periodic / "sr_mhd_linear_wave.out2.00000.phdf"
    final = periodic / "sr_mhd_linear_wave.out2.final.phdf"
    initial_sum, initial_residual = electron_consistency(initial)
    final_sum, final_residual = electron_consistency(final)
    conservation = np.abs(final_sum - initial_sum) / np.maximum(np.abs(initial_sum), 1.0e-300)
    if float(np.max(conservation)) > 5.0e-15:
        raise RuntimeError(f"PLM electron conservation residual={np.max(conservation):.17e}")

    restart_files = sorted(periodic.glob("*.rhdf"))
    restart_file = next((path for path in restart_files if ".00001." in path.name), None)
    if restart_file is None:
        raise RuntimeError("periodic run did not produce an intermediate restart")
    restart_log = run(
        [executable, "-r", str(restart_file.resolve()), "parthenon/time/tlim=0.1"],
        restarted,
        "run.log",
    )
    if "Var: electrons.cons:2" not in restart_log:
        raise RuntimeError("restart did not restore electrons.cons")
    restart_final = restarted / "sr_mhd_linear_wave.out2.final.phdf"
    restart_linf: dict[str, float] = {}
    with h5py.File(final, "r") as continuous_data, h5py.File(restart_final, "r") as restart_data:
        for name in ("electrons.cons", "electrons.prim", "mhd.cons", "mhd.prim"):
            left = continuous_data[name][...]
            right = restart_data[name][...]
            restart_linf[name] = float(np.max(np.abs(left - right)))
            if not np.array_equal(left, right):
                raise RuntimeError(f"restart is not byte-exact for {name}")

    run(
        [
            executable,
            "-i",
            str(args.disabled_input.resolve()),
            "parthenon/mesh/nx1=64",
            "parthenon/meshblock/nx1=32",
            "parthenon/time/tlim=0.1",
            "mhd/fofc=true",
            "parthenon/output1/dt=0.05",
            "parthenon/output2/dt=0.05",
        ],
        disabled,
        "run.log",
    )
    disabled_final = disabled / "sr_mhd_linear_wave.out2.final.phdf"
    with h5py.File(final, "r") as enabled_data, h5py.File(disabled_final, "r") as disabled_data:
        if any(name.startswith("electrons") for name in disabled_data.keys()):
            raise RuntimeError("disabled run registered electron output fields")
        passive_mhd_linf = float(
            np.max(np.abs(enabled_data["mhd.prim"][...] - disabled_data["mhd.prim"][...]))
        )
        if passive_mhd_linf > 1.0e-12:
            raise RuntimeError(f"passive electron module changed MHD by {passive_mhd_linf:.17e}")

    run(
        [
            executable,
            "-i",
            str(args.periodic_input.resolve()),
            "parthenon/mesh/nghost=4",
            "mhd/reconstruct=ppm",
            "parthenon/time/tlim=0.02",
            "parthenon/output1/dt=0.01",
            "parthenon/output2/dt=0.01",
            "parthenon/output3/dt=0.01",
        ],
        ppm,
        "run.log",
    )
    ppm_initial_sum, ppm_initial_residual = electron_consistency(
        ppm / "sr_mhd_linear_wave.out2.00000.phdf"
    )
    ppm_final_sum, ppm_final_residual = electron_consistency(
        ppm / "sr_mhd_linear_wave.out2.final.phdf"
    )
    ppm_conservation = np.abs(ppm_final_sum - ppm_initial_sum) / np.maximum(
        np.abs(ppm_initial_sum), 1.0e-300
    )
    if float(np.max(ppm_conservation)) > 5.0e-15:
        raise RuntimeError(f"PPM electron conservation residual={np.max(ppm_conservation):.17e}")

    smr_log = run([executable, "-i", str(args.smr_input.resolve())], smr, "run.log")
    if "Physical level = 0" not in smr_log or "Physical level = 1" not in smr_log:
        raise RuntimeError("SMR run did not retain both physical refinement levels")
    _, smr_residual = electron_consistency(smr / "gr_monopole.out2.final.phdf")
    with h5py.File(smr / "gr_monopole.out2.final.phdf", "r") as data:
        levels = sorted(set(int(value) for value in data["Levels"][...]))
    if levels != [0, 1]:
        raise RuntimeError(f"unexpected SMR output levels: {levels}")

    report = {
        "assembly": {"metric": "cks", "mode": "dynamic", "backend": "cuda"},
        "plm_conservation_relative": conservation.tolist(),
        "plm_u_equals_dk_relative": max(initial_residual, final_residual),
        "restart_linf": restart_linf,
        "disabled_has_electron_fields": False,
        "passive_mhd_primitive_linf": passive_mhd_linf,
        "ppm_conservation_relative": ppm_conservation.tolist(),
        "ppm_u_equals_dk_relative": max(ppm_initial_residual, ppm_final_residual),
        "smr_levels": levels,
        "smr_u_equals_dk_relative": smr_residual,
    }
    (case / "eh1-electron-transport-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "EH-1 CKS/Dynamic electron transport PASS: PLM/PPM conservative, "
        "restart byte-exact, disabled path field-free, SMR levels 0/1"
    )
    print(case)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"EH-1 electron transport FAIL: {error}")
        raise SystemExit(1)
