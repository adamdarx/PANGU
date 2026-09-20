#!/usr/bin/env python3
"""Validate independent electron heating on the CKS/Dynamic SMR SANE path."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile

import h5py
import numpy as np


def run(command: list[str], directory: pathlib.Path) -> str:
    result = subprocess.run(command, cwd=directory, text=True, capture_output=True)
    output = result.stdout + result.stderr
    (directory / "run.log").write_text(output, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed; see {directory / 'run.log'}")
    return output


def final_dump(directory: pathlib.Path) -> pathlib.Path:
    paths = sorted(directory.glob("gr_torus_sane.prim.final.phdf"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one final SANE dump in {directory}, found {len(paths)}")
    return paths[0]


def radius(stream: h5py.File, shape: tuple[int, ...]) -> np.ndarray:
    x = np.broadcast_to(
        np.asarray(stream["VolumeLocations/x"], dtype=np.float64)[:, None, None, :], shape
    )
    y = np.broadcast_to(
        np.asarray(stream["VolumeLocations/y"], dtype=np.float64)[:, None, :, None], shape
    )
    z = np.broadcast_to(
        np.asarray(stream["VolumeLocations/z"], dtype=np.float64)[:, :, None, None], shape
    )
    return np.sqrt(x * x + y * y + z * z)


def check_active(path: pathlib.Path) -> dict[str, object]:
    with h5py.File(path, "r") as stream:
        primitive = np.asarray(stream["mhd.prim"], dtype=np.float64)
        conserved = np.asarray(stream["mhd.cons"], dtype=np.float64)
        electron = np.asarray(stream["electrons.prim"], dtype=np.float64)
        electron_cons = np.asarray(stream["electrons.cons"], dtype=np.float64)
        diagnostics = np.asarray(stream["electrons.diagnostics"], dtype=np.float64)
        divergence = np.asarray(stream["mhd.divb"], dtype=np.float64)
        levels = sorted(set(int(value) for value in stream["Levels"][...]))
        cell_radius = radius(stream, primitive[:, 0].shape)
    if electron.shape[1] != 6:
        raise RuntimeError(f"expected Ktot plus five kinetic models, found {electron.shape[1]}")
    arrays = (primitive, conserved, electron, electron_cons, diagnostics, divergence)
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("CKS SANE electron output contains NaN or Inf")
    if levels != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"CKS SANE did not retain four nested SMR levels: {levels}")
    scale = max(float(np.max(np.abs(electron_cons))), np.finfo(np.float64).tiny)
    u_equals_dk = float(
        np.max(np.abs(electron_cons - conserved[:, 0:1] * electron)) / scale
    )
    outside = cell_radius >= 1.0
    gamma = 4.0 / 3.0
    energy_entropy = (gamma - 1.0) * primitive[:, 4] * primitive[:, 0] ** (-gamma)
    entropy_scale = np.maximum(np.abs(energy_entropy[outside]), np.finfo(np.float64).tiny)
    ktot_outside = float(
        np.max(np.abs(electron[:, 0][outside] - energy_entropy[outside]) / entropy_scale)
    )
    divb_outside = float(np.max(np.abs(divergence[outside])))
    if u_equals_dk > 5.0e-15:
        raise RuntimeError(f"CKS SANE violates U_K=D*K: {u_equals_dk:.17e}")
    if ktot_outside > 5.0e-15:
        raise RuntimeError(f"CKS SANE Ktot is not synchronized outside excision: {ktot_outside:.17e}")
    if divb_outside > 1.0e-12:
        raise RuntimeError(f"CKS SANE divB outside excision is {divb_outside:.17e}")
    return {
        "levels": levels,
        "outside_excision_cells": int(np.count_nonzero(outside)),
        "inside_excision_cells": int(np.count_nonzero(~outside)),
        "u_equals_dk_relative": u_equals_dk,
        "ktot_energy_relative_outside_excision": ktot_outside,
        "max_abs_divb_outside_excision": divb_outside,
        "diagnostic_fraction_range": [
            float(np.min(diagnostics[:, 2])),
            float(np.max(diagnostics[:, 2])),
        ],
        "diagnostic_flag_values": sorted(
            set(int(value) for value in diagnostics[:, 4].reshape(-1))
        ),
    }


def field_comparison(
    left: pathlib.Path, right: pathlib.Path, names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with h5py.File(left, "r") as left_stream, h5py.File(right, "r") as right_stream:
        density = np.asarray(right_stream["mhd.prim"])[:, 0]
        outside = radius(right_stream, density.shape) >= 1.0
        for name in names:
            left_field = np.asarray(left_stream[name])
            right_field = np.asarray(right_stream[name])
            difference = np.abs(left_field - right_field)
            mask = outside[:, None] if difference.ndim == 5 else outside
            mask = np.broadcast_to(mask, difference.shape)
            denominator = max(
                float(np.sum(np.abs(right_field[mask]))), np.finfo(np.float64).tiny
            )
            result[name] = {
                "linf_all": float(np.max(difference)),
                "linf_outside_excision": float(np.max(difference[mask])),
                "relative_l1_outside_excision": float(np.sum(difference[mask])) / denominator,
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    executable = str(args.executable.resolve())
    input_path = str(args.input.resolve())
    args.workdir.mkdir(parents=True, exist_ok=True)
    case = pathlib.Path(tempfile.mkdtemp(prefix="eh4-cks-sane-", dir=args.workdir))
    active = case / "active"
    passive = case / "passive"
    disabled = case / "disabled"
    restarted = case / "restarted"
    for directory in (active, passive, disabled, restarted):
        directory.mkdir()

    run([executable, "-i", input_path], active)
    run([executable, "-i", input_path, "electrons/heating=false"], passive)
    run(
        [
            executable,
            "-i",
            input_path,
            "electrons/enabled=false",
            "parthenon/output2/variables=mhd.prim,mhd.cons,mhd.b_cell,mhd.divb,mhd.fofc",
        ],
        disabled,
    )

    active_final = final_dump(active)
    passive_final = final_dump(passive)
    disabled_final = final_dump(disabled)
    active_checks = check_active(active_final)
    mhd_names = ("mhd.prim", "mhd.cons", "mhd.b_cell", "mhd.divb", "mhd.fofc")
    heating_mhd = field_comparison(active_final, passive_final, mhd_names)
    disabled_mhd = field_comparison(active_final, disabled_final, mhd_names)
    if any(value["linf_all"] != 0.0 for value in heating_mhd.values()):
        raise RuntimeError(f"electron heating changed CKS SANE MHD: {heating_mhd}")
    disabled_limits = {
        "mhd.prim": (1.0e-11, None),
        "mhd.cons": (1.0e-11, None),
        "mhd.b_cell": (1.0e-7, None),
        "mhd.divb": (None, 1.0e-15),
        "mhd.fofc": (None, 0.0),
    }
    disabled_violations: dict[str, object] = {}
    for name, (relative_limit, absolute_limit) in disabled_limits.items():
        measured = disabled_mhd[name]
        if relative_limit is not None and measured["relative_l1_outside_excision"] > relative_limit:
            disabled_violations[name] = {
                "measured": measured,
                "relative_l1_limit": relative_limit,
            }
        if absolute_limit is not None and measured["linf_outside_excision"] > absolute_limit:
            disabled_violations[name] = {
                "measured": measured,
                "linf_limit": absolute_limit,
            }
    if disabled_violations:
        raise RuntimeError(
            f"electron package changed CKS SANE MHD beyond roundoff gates: {disabled_violations}"
        )
    with h5py.File(disabled_final, "r") as stream:
        disabled_fields = [name for name in stream.keys() if name.startswith("electrons")]
    if disabled_fields:
        raise RuntimeError(f"disabled electron package registered fields: {disabled_fields}")

    restart_files = sorted(active.glob("gr_torus_sane.restart.*.rhdf"))
    restart = next((path for path in restart_files if ".00001." in path.name), None)
    if restart is None:
        raise RuntimeError("CKS SANE run did not write the cycle-one restart")
    restart_log = run(
        [executable, "-r", str(restart.resolve()), "parthenon/time/nlim=2"], restarted
    )
    if "Var: electrons.cons:6" not in restart_log:
        raise RuntimeError("CKS SANE restart did not restore all electron conserved components")
    restart_final = final_dump(restarted)
    restart_names = mhd_names + ("electrons.prim", "electrons.cons", "electrons.diagnostics")
    restart_difference = field_comparison(active_final, restart_final, restart_names)
    mhd_restart_failure = {
        name: restart_difference[name]
        for name in mhd_names
        if restart_difference[name]["linf_all"] != 0.0
    }
    electron_restart_failure = {
        name: restart_difference[name]
        for name in ("electrons.prim", "electrons.cons", "electrons.diagnostics")
        if restart_difference[name]["relative_l1_outside_excision"] > 5.0e-15
    }
    if mhd_restart_failure or electron_restart_failure:
        raise RuntimeError(
            "CKS SANE restart exceeds its identity/roundoff gates: "
            f"MHD={mhd_restart_failure}, electrons={electron_restart_failure}"
        )

    report = {
        "assembly": {"metric": "cks", "mode": "dynamic", "backend": "cuda"},
        "active": active_checks,
        "mhd_heating_vs_passive": heating_mhd,
        "mhd_electrons_vs_disabled": disabled_mhd,
        "mhd_electrons_vs_disabled_limits": {
            name: {
                "relative_l1_outside_excision": relative_limit,
                "linf_outside_excision": absolute_limit,
            }
            for name, (relative_limit, absolute_limit) in disabled_limits.items()
        },
        "electron_enabled_kernel_note": (
            "WritePassiveTransport adds mass-flux coefficient outputs to the GRMHD kernel; "
            "the changed CUDA register schedule perturbs MHD roundoff but heating is non-backreacting"
        ),
        "disabled_has_electron_fields": False,
        "restart_difference": restart_difference,
        "restart_gate": {
            "mhd_linf": 0.0,
            "electron_relative_l1_outside_excision": 5.0e-15,
        },
        "passed": True,
    }
    (case / "eh4-cks-sane-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "EH-4 CKS/Dynamic SANE PASS: four-level SMR, finite inside/outside excision, "
        "heating has zero MHD backreaction, disabled-path differences below roundoff gates, "
        "MHD restart byte-exact and electron restart within roundoff"
    )
    print(case)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"EH-4 CKS/Dynamic SANE FAIL: {error}")
        raise SystemExit(1)
