#!/usr/bin/env python3
"""Run the EH-2 CKS/Dynamic GPU Constant-heating acceptance gates."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile

import h5py
import numpy as np


GAMMA_HUBBLE = 4.0 / 3.0
GAMMA_NOH = 5.0 / 3.0
GAMMA_E = 4.0 / 3.0
NOH_FRACTION = 0.5


def run(command: list[str], directory: pathlib.Path, log_name: str = "run.log") -> str:
    result = subprocess.run(command, cwd=directory, text=True, capture_output=True)
    output = result.stdout + result.stderr
    (directory / log_name).write_text(output, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed; see {directory / log_name}")
    return output


def run_expected_failure(
    command: list[str], directory: pathlib.Path, expected: str, log_name: str = "run.log"
) -> None:
    result = subprocess.run(command, cwd=directory, text=True, capture_output=True)
    output = result.stdout + result.stderr
    (directory / log_name).write_text(output, encoding="utf-8")
    if result.returncode == 0 or expected not in output:
        raise RuntimeError(
            f"{' '.join(command)} did not fail with the required diagnostic {expected!r}"
        )


def cell_dataset(data: h5py.File, name: str) -> np.ndarray:
    array = np.asarray(data[name])
    return array[:, :, 0, 0, :].transpose(0, 2, 1).reshape(-1, array.shape[1])


def load_state(path: pathlib.Path) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as data:
        x = np.asarray(data["VolumeLocations/x"]).reshape(-1)
        state: dict[str, np.ndarray | float] = {
            "time": float(data["Info"].attrs["Time"]),
            "x": x,
            "mhd_prim": cell_dataset(data, "mhd.prim"),
            "mhd_cons": cell_dataset(data, "mhd.cons"),
            "electron_prim": cell_dataset(data, "electrons.prim"),
            "electron_cons": cell_dataset(data, "electrons.cons"),
            "diagnostics": cell_dataset(data, "electrons.diagnostics"),
        }
    order = np.argsort(state["x"])
    for name in ("x", "mhd_prim", "mhd_cons", "electron_prim", "electron_cons", "diagnostics"):
        state[name] = state[name][order]
    return state


def check_electron_state(state: dict[str, np.ndarray | float], gamma: float) -> dict[str, float]:
    arrays = [
        state["mhd_prim"],
        state["mhd_cons"],
        state["electron_prim"],
        state["electron_cons"],
        state["diagnostics"],
    ]
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("electron-heating output contains NaN or Inf")
    primitive = state["mhd_prim"]
    conserved = state["mhd_cons"]
    electron = state["electron_prim"]
    electron_cons = state["electron_cons"]
    expected_cons = conserved[:, 0:1] * electron
    conserved_scale = max(float(np.max(np.abs(electron_cons))), np.finfo(np.float64).tiny)
    u_equals_dk = float(np.max(np.abs(electron_cons - expected_cons)) / conserved_scale)
    if u_equals_dk > 5.0e-15:
        raise RuntimeError(f"U_K=D*K residual is {u_equals_dk:.17e}")
    energy_entropy = (gamma - 1.0) * primitive[:, 4] * primitive[:, 0] ** (-gamma)
    entropy_sync = float(np.max(np.abs(electron[:, 0] - energy_entropy)))
    if entropy_sync > 5.0e-14:
        raise RuntimeError(f"Ktot is not synchronized to the energy entropy: {entropy_sync:.17e}")
    return {"u_equals_dk_relative": u_equals_dk, "ktot_energy_linf": entropy_sync}


def relative_l1(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - expected)) / max(np.mean(np.abs(expected)), 1.0e-300))


def check_hubble(directory: pathlib.Path) -> dict[str, object]:
    files = sorted(directory.glob("electron_hubble.out2*.phdf"))
    if len(files) != 5:
        raise RuntimeError(f"Hubble run produced {len(files)} states instead of five")
    rho0 = 0.1 / 0.1 * np.sqrt(GAMMA_HUBBLE * (GAMMA_HUBBLE - 1.0))
    internal0 = 0.1 / 0.1 / np.sqrt(GAMMA_HUBBLE * (GAMMA_HUBBLE - 1.0))
    records: list[dict[str, float]] = []
    for path in files:
        state = load_state(path)
        consistency = check_electron_state(state, GAMMA_HUBBLE)
        time = float(state["time"])
        x = state["x"]
        primitive = state["mhd_prim"]
        diagnostics = state["diagnostics"]
        mask = np.abs(x) < 0.5
        factor = 1.0 + 0.1 * time
        density_reference = np.full(mask.sum(), rho0 / factor)
        internal_reference = np.full(mask.sum(), internal0 / factor**GAMMA_HUBBLE)
        velocity = 0.1 * x / factor
        four_velocity_reference = velocity / np.sqrt(1.0 - velocity * velocity)
        record = {
            "time": time,
            "density_l1": relative_l1(primitive[mask, 0], density_reference),
            "internal_energy_l1": relative_l1(primitive[mask, 4], internal_reference),
            "four_velocity_l1": relative_l1(primitive[mask, 1], four_velocity_reference[mask]),
            "transverse_velocity_linf": float(np.max(np.abs(primitive[:, 2:4]))),
            "raw_dissipation_linf": float(np.max(np.abs(diagnostics[mask, 0]))),
            **consistency,
        }
        records.append(record)
    final = records[-1]
    if final["density_l1"] > 3.0e-3 or final["internal_energy_l1"] > 4.0e-3:
        raise RuntimeError(f"Hubble analytic fluid error is too large: {final}")
    if final["four_velocity_l1"] > 5.0e-2:
        raise RuntimeError(f"Hubble velocity error is too large: {final['four_velocity_l1']:.17e}")
    if max(record["transverse_velocity_linf"] for record in records) > 1.0e-14:
        raise RuntimeError("Hubble expansion generated transverse velocity")
    if max(record["raw_dissipation_linf"] for record in records) > 1.0e-6:
        raise RuntimeError("Hubble smooth flow generated excessive numerical dissipation")
    return {"states": records}


def noh_analytic_ratio() -> float:
    gamma = GAMMA_NOH
    gamma_e = GAMMA_E
    return (
        NOH_FRACTION
        / 2.0
        * (
            ((gamma + 1.0) / (gamma - 1.0)) ** gamma_e * (1.0 - gamma / gamma_e)
            + 1.0
            + gamma / gamma_e
        )
        * ((gamma * gamma - 1.0) / (gamma_e * gamma_e - 1.0))
    )


def check_noh(states: dict[int, dict[str, np.ndarray | float]]) -> dict[str, object]:
    analytic_postshock = noh_analytic_ratio()
    errors: list[float] = []
    records: dict[str, dict[str, float]] = {}
    for resolution, state in states.items():
        consistency = check_electron_state(state, GAMMA_NOH)
        x = state["x"]
        primitive = state["mhd_prim"]
        electron = state["electron_prim"]
        diagnostics = state["diagnostics"]
        electron_internal = electron[:, 1] * primitive[:, 0] ** GAMMA_E / (GAMMA_E - 1.0)
        ratio = electron_internal / np.maximum(primitive[:, 4], np.finfo(np.float64).tiny)
        reference = np.where(primitive[:, 0] > 1.5, analytic_postshock, 0.0)
        error = float(np.mean(np.abs(ratio - reference)))
        errors.append(error)
        shocked = primitive[:, 0] > 1.5
        shock_position = float(np.max(np.abs(x[shocked])))
        spacing = float(x[1] - x[0])
        outside = np.abs(x) > shock_position + 4.0 * spacing
        record = {
            "l1_ue_over_ug": error,
            "shock_position": shock_position,
            "maximum_density": float(np.max(primitive[:, 0])),
            "raw_dissipation_min": float(np.min(diagnostics[:, 0])),
            "raw_dissipation_max": float(np.max(diagnostics[:, 0])),
            "outside_shock_dissipation_linf": float(np.max(np.abs(diagnostics[outside, 0]))),
            "applied_minus_raw_linf": float(np.max(np.abs(diagnostics[:, 1] - diagnostics[:, 0]))),
            "flags_linf": float(np.max(np.abs(diagnostics[:, 4]))),
            **consistency,
        }
        records[str(resolution)] = record
        if not 0.17 < shock_position < 0.23 or not 3.5 < record["maximum_density"] < 4.5:
            raise RuntimeError(f"Noh shock structure failed at N={resolution}: {record}")
        if record["raw_dissipation_max"] <= 0.0:
            raise RuntimeError(f"Noh shock generated no positive dissipation at N={resolution}")
        if record["outside_shock_dissipation_linf"] > 1.0e-16:
            raise RuntimeError(f"Noh dissipation leaked outside the shock at N={resolution}")
        if record["applied_minus_raw_linf"] > 1.0e-18 or record["flags_linf"] != 0.0:
            raise RuntimeError(f"unlimited Noh heating changed raw dissipation at N={resolution}")
    order = float(-np.polyfit(np.log(np.asarray(list(states))), np.log(errors), 1)[0])
    if not 0.85 < order < 1.15:
        raise RuntimeError(f"Noh electron-energy convergence order is {order:.17e}")
    return {"analytic_postshock_ue_over_ug": analytic_postshock, "order": order, "runs": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--hubble-input", required=True, type=pathlib.Path)
    parser.add_argument("--noh-input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    executable = str(args.executable.resolve())
    args.workdir.mkdir(parents=True, exist_ok=True)
    case = pathlib.Path(tempfile.mkdtemp(prefix="eh2-electrons-", dir=args.workdir))
    hubble = case / "hubble"
    hubble_disabled = case / "hubble-disabled"
    unsupported_source = case / "unsupported-source"
    hubble.mkdir()
    hubble_disabled.mkdir()
    unsupported_source.mkdir()
    run([executable, "-i", str(args.hubble_input.resolve())], hubble)
    hubble_report = check_hubble(hubble)
    run(
        [
            executable,
            "-i",
            str(args.hubble_input.resolve()),
            "electrons/heating=false",
            "electrons/write_diagnostics=false",
            "parthenon/output1/dt=1.0",
            "parthenon/output2/dt=1.0",
        ],
        hubble_disabled,
    )
    with h5py.File(hubble / "electron_hubble.out2.final.phdf", "r") as enabled, h5py.File(
        hubble_disabled / "electron_hubble.out2.final.phdf", "r"
    ) as disabled:
        passive_mhd_linf = float(np.max(np.abs(enabled["mhd.prim"][...] - disabled["mhd.prim"][...])))
    if passive_mhd_linf != 0.0:
        raise RuntimeError(f"Constant electron heating changed the MHD state by {passive_mhd_linf:.17e}")
    run_expected_failure(
        [
            executable,
            "-i",
            str(args.hubble_input.resolve()),
            "source_terms/heating_rate=0.1",
        ],
        unsupported_source,
        "electron heating does not yet support external fluid-energy sources",
    )

    noh_states: dict[int, dict[str, np.ndarray | float]] = {}
    for resolution in (128, 256, 512):
        directory = case / f"noh-{resolution}"
        directory.mkdir()
        run(
            [
                executable,
                "-i",
                str(args.noh_input.resolve()),
                f"parthenon/mesh/nx1={resolution}",
                "parthenon/output1/dt=600.0",
                "parthenon/output2/dt=600.0",
            ],
            directory,
        )
        noh_states[resolution] = load_state(directory / "electron_noh.out2.final.phdf")
    noh_report = check_noh(noh_states)

    report = {
        "assembly": {"metric": "cks", "mode": "dynamic", "backend": "cuda"},
        "constant_heating_mhd_linf": passive_mhd_linf,
        "unsupported_external_energy_source_rejected": True,
        "hubble": hubble_report,
        "noh": noh_report,
    }
    (case / "eh2-electron-heating-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "EH-2 CKS/Dynamic Constant heating PASS: Hubble analytic evolution, passive MHD, "
        f"Noh electron-energy convergence order={noh_report['order']:.6f}"
    )
    print(case)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"EH-2 electron heating FAIL: {error}")
        raise SystemExit(1)
