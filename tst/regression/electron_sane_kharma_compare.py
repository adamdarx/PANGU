#!/usr/bin/env python3
"""Compare PANGU and KHARMA electron thermodynamics in an MKS SANE torus.

The two codes use the same native mesh and primitive-variable convention.  At
initialization and in a deterministic one-cycle gate, fields can therefore be
compared directly in the disk.  Once the pressure perturbation is enabled,
the comparison switches to volume integrals and mass-weighted distributions,
because independent random-number streams and limiter decisions decorrelate
pointwise MRI trajectories.

The Rowan prescription is reported both on its published fit domain and over
the full disk.  KHARMA evaluates ``(1 - beta/beta_max)**3.3`` directly; outside
that domain the result is NaN and KHARMA's clip helper returns the lower
electron-temperature bound.  PANGU deliberately clamps the fit argument to
zero so that the heating fraction remains finite.  Treating the invalid cells
separately preserves a strict comparison without calling that robustness
difference a physical disagreement.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


GAMMA_GAS = 4.0 / 3.0
GAMMA_ELECTRON = 4.0 / 3.0
GAMMA_PROTON = 5.0 / 3.0
PROTON_TO_ELECTRON_MASS = 1836.15267343
SMALL = 1.0e-30
MODEL_NAMES = ("Kawazura", "Werner", "Rowan", "Sharma")
QUANTILES = np.asarray((0.05, 0.50, 0.95), dtype=np.float64)


@dataclass
class State:
    time: float
    cycle: int
    density: np.ndarray
    internal: np.ndarray
    velocity: np.ndarray
    magnetic: np.ndarray
    ktot: np.ndarray
    kel: dict[str, np.ndarray]
    electron_cons: dict[str, np.ndarray]
    proper_volume: np.ndarray
    native_cell_volume: float
    magnetic_squared: np.ndarray
    raw_dissipation: np.ndarray | None = None
    applied_dissipation: np.ndarray | None = None
    diagnostic_fraction: np.ndarray | None = None
    diagnostic_temperature_ratio: np.ndarray | None = None
    diagnostic_flags: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=Path)
    parser.add_argument("--kharma", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spin", type=float, default=0.9375)
    parser.add_argument("--hslope", type=float, default=0.3)
    parser.add_argument("--disk-density", type=float, default=1.0e-3)
    parser.add_argument("--time-tolerance", type=float, default=5.0e-2)
    return parser.parse_args()


def snapshots(directory: Path, pattern: str) -> list[tuple[float, Path]]:
    found: list[tuple[float, Path]] = []
    for path in directory.glob(pattern):
        with h5py.File(path, "r") as stream:
            found.append((float(stream["Info"].attrs["Time"]), path))
    ordered = sorted(found, key=lambda item: (item[0], ".final." in item[1].name, item[1].name))
    unique: list[tuple[float, Path]] = []
    for item in ordered:
        if unique and math.isclose(item[0], unique[-1][0], rel_tol=0.0, abs_tol=1.0e-12):
            continue
        unique.append(item)
    return unique


def cell_width(locations: np.ndarray) -> float:
    differences = np.diff(np.asarray(locations, dtype=np.float64), axis=1)
    nonzero = np.abs(differences) > 0.0
    if not np.any(nonzero):
        return 1.0
    return float(np.median(np.abs(differences[nonzero])))


def mks_metric(
    stream: h5py.File, spin: float, hslope: float, cell_shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x1_locations = np.asarray(stream["VolumeLocations/x"], dtype=np.float64)
    x2_locations = np.asarray(stream["VolumeLocations/y"], dtype=np.float64)
    x3_locations = np.asarray(stream["VolumeLocations/z"], dtype=np.float64)
    x1 = np.broadcast_to(x1_locations[:, None, None, :], cell_shape)
    x2 = np.broadcast_to(x2_locations[:, None, :, None], cell_shape)
    radius = np.exp(x1)
    theta = np.pi * x2 + 0.5 * (1.0 - hslope) * np.sin(2.0 * np.pi * x2)
    theta_jacobian = np.pi * (1.0 + (1.0 - hslope) * np.cos(2.0 * np.pi * x2))
    sine = np.sin(theta)
    cosine = np.cos(theta)
    sine2 = sine * sine
    radius2 = radius * radius
    spin2 = spin * spin
    rho2 = radius2 + spin2 * cosine * cosine
    factor = 2.0 * radius / rho2
    one_plus_factor = 1.0 + factor

    metric = np.zeros(cell_shape + (4, 4), dtype=np.float64)
    metric[..., 0, 0] = -1.0 + factor
    metric[..., 0, 1] = factor * radius
    metric[..., 1, 0] = metric[..., 0, 1]
    metric[..., 0, 3] = -spin * sine2 * factor
    metric[..., 3, 0] = metric[..., 0, 3]
    metric[..., 1, 1] = one_plus_factor * radius2
    metric[..., 1, 3] = -spin * sine2 * one_plus_factor * radius
    metric[..., 3, 1] = metric[..., 1, 3]
    metric[..., 2, 2] = rho2 * theta_jacobian * theta_jacobian
    metric[..., 3, 3] = sine2 * (rho2 + spin2 * sine2 * one_plus_factor)

    lapse = np.sqrt(1.0 / one_plus_factor)
    sqrt_minus_g = np.abs(rho2 * sine * radius * theta_jacobian)
    proper_volume = sqrt_minus_g / lapse
    native_cell_volume = (
        cell_width(x1_locations) * cell_width(x2_locations) * cell_width(x3_locations)
    )
    return metric, lapse, proper_volume, native_cell_volume


def comoving_magnetic_squared(
    velocity: np.ndarray, magnetic: np.ndarray, metric: np.ndarray, lapse: np.ndarray
) -> np.ndarray:
    velocity_last = np.moveaxis(velocity, 1, -1)
    magnetic_last = np.moveaxis(magnetic, 1, -1)
    inverse = np.linalg.inv(metric)
    spatial_metric = metric[..., 1:, 1:]
    spatial_u2 = np.einsum(
        "...i,...ij,...j->...", velocity_last, spatial_metric, velocity_last, optimize=True
    )
    lorentz = np.sqrt(np.maximum(1.0 + spatial_u2, 1.0))
    four_velocity = np.empty(velocity_last.shape[:-1] + (4,), dtype=np.float64)
    four_velocity[..., 0] = lorentz / lapse
    four_velocity[..., 1:] = velocity_last - (
        lapse * lorentz
    )[..., None] * inverse[..., 0, 1:]
    lower_velocity = np.einsum("...ij,...j->...i", metric, four_velocity, optimize=True)
    b0 = np.sum(lower_velocity[..., 1:] * magnetic_last, axis=-1)
    four_magnetic = np.empty_like(four_velocity)
    four_magnetic[..., 0] = b0
    four_magnetic[..., 1:] = (
        magnetic_last + b0[..., None] * four_velocity[..., 1:]
    ) / four_velocity[..., 0, None]
    magnetic_squared = np.einsum(
        "...i,...ij,...j->...", four_magnetic, metric, four_magnetic, optimize=True
    )
    return np.maximum(magnetic_squared, 0.0)


def pangu_state(path: Path, spin: float, hslope: float) -> State:
    with h5py.File(path, "r") as stream:
        primitive = np.asarray(stream["mhd.prim"], dtype=np.float64)
        magnetic = np.asarray(stream["mhd.b_cell"], dtype=np.float64)
        electron = np.asarray(stream["electrons.prim"], dtype=np.float64)
        electron_cons = np.asarray(stream["electrons.cons"], dtype=np.float64)
        diagnostics = np.asarray(stream["electrons.diagnostics"], dtype=np.float64)
        density = primitive[:, 0]
        metric, lapse, proper_volume, native_cell_volume = mks_metric(
            stream, spin, hslope, density.shape
        )
        magnetic_squared = comoving_magnetic_squared(
            primitive[:, 1:4], magnetic, metric, lapse
        )
        info = stream["Info"].attrs
        return State(
            time=float(info["Time"]),
            cycle=int(info["NCycle"]),
            density=density,
            internal=primitive[:, 4],
            velocity=primitive[:, 1:4],
            magnetic=magnetic,
            ktot=electron[:, 0],
            kel={name: electron[:, index + 1] for index, name in enumerate(MODEL_NAMES)},
            electron_cons={
                "Ktot": electron_cons[:, 0],
                **{
                    name: electron_cons[:, index + 1]
                    for index, name in enumerate(MODEL_NAMES)
                },
            },
            proper_volume=proper_volume,
            native_cell_volume=native_cell_volume,
            magnetic_squared=magnetic_squared,
            raw_dissipation=diagnostics[:, 0],
            applied_dissipation=diagnostics[:, 1],
            diagnostic_fraction=diagnostics[:, 2],
            diagnostic_temperature_ratio=diagnostics[:, 3],
            diagnostic_flags=diagnostics[:, 4],
        )


def kharma_state(path: Path, spin: float, hslope: float) -> State:
    with h5py.File(path, "r") as stream:
        density = np.asarray(stream["prims.rho"], dtype=np.float64)
        velocity = np.asarray(stream["prims.uvec"], dtype=np.float64)
        magnetic = np.asarray(stream["prims.B"], dtype=np.float64)
        metric, lapse, proper_volume, native_cell_volume = mks_metric(
            stream, spin, hslope, density.shape
        )
        magnetic_squared = comoving_magnetic_squared(velocity, magnetic, metric, lapse)
        info = stream["Info"].attrs
        return State(
            time=float(info["Time"]),
            cycle=int(info["NCycle"]),
            density=density,
            internal=np.asarray(stream["prims.u"], dtype=np.float64),
            velocity=velocity,
            magnetic=magnetic,
            ktot=np.asarray(stream["prims.Ktot"], dtype=np.float64),
            kel={
                name: np.asarray(stream[f"prims.Kel_{name}"], dtype=np.float64)
                for name in MODEL_NAMES
            },
            electron_cons={
                "Ktot": np.asarray(stream["cons.Ktot"], dtype=np.float64),
                **{
                    name: np.asarray(stream[f"cons.Kel_{name}"], dtype=np.float64)
                    for name in MODEL_NAMES
                },
            },
            proper_volume=proper_volume,
            native_cell_volume=native_cell_volume,
            magnetic_squared=magnetic_squared,
        )


def weighted_quantile(values: np.ndarray, weights: np.ndarray) -> list[float]:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[finite].reshape(-1)
    weights = weights[finite].reshape(-1)
    if values.size == 0:
        return [math.nan] * len(QUANTILES)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return [float(value) for value in np.interp(QUANTILES, cumulative, values)]


def relative_l1(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    numerator = float(np.sum(np.abs(left[mask] - right[mask])))
    denominator = max(float(np.sum(np.abs(right[mask]))), np.finfo(np.float64).tiny)
    return numerator / denominator


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), np.finfo(np.float64).tiny)


def temperature_fields(state: State, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    electron_temperature = state.kel[model] * state.density ** (GAMMA_ELECTRON - 1.0)
    proton_temperature = (
        (GAMMA_PROTON - 1.0) * state.internal / np.maximum(state.density, SMALL)
    )
    tp_over_te = proton_temperature / np.maximum(electron_temperature, SMALL)
    theta_e = PROTON_TO_ELECTRON_MASS * electron_temperature
    electron_internal = (
        state.kel[model] * state.density**GAMMA_ELECTRON / (GAMMA_ELECTRON - 1.0)
    )
    return tp_over_te, theta_e, electron_internal


def heating_fraction(
    state: State, model: str, robust_rowan: bool
) -> tuple[np.ndarray, np.ndarray]:
    electron_temperature = np.maximum(
        state.kel[model] * state.density ** (GAMMA_ELECTRON - 1.0), SMALL
    )
    proton_temperature = np.maximum(
        (GAMMA_PROTON - 1.0) * state.internal / np.maximum(state.density, SMALL), SMALL
    )
    temperature_ratio = proton_temperature / electron_temperature
    bsq = state.magnetic_squared
    if model == "Kawazura":
        proton_pressure = state.density * proton_temperature
        beta = np.minimum(2.0 * proton_pressure / np.maximum(bsq, SMALL), 1.0e20)
        ion_to_electron = 35.0 / (
            1.0 + (np.maximum(beta, SMALL) / 15.0) ** -1.4 * np.exp(-0.1 / temperature_ratio)
        )
        fraction = 1.0 / (1.0 + ion_to_electron)
        valid = np.ones_like(fraction, dtype=bool)
    elif model == "Werner":
        sigma = bsq / np.maximum(state.density, SMALL)
        scaled_sigma = sigma / 5.0
        fraction = 0.25 * (1.0 + np.sqrt(scaled_sigma / (2.0 + scaled_sigma)))
        valid = np.ones_like(fraction, dtype=bool)
    elif model == "Rowan":
        proton_pressure = (GAMMA_PROTON - 1.0) * state.internal
        gas_pressure = (GAMMA_GAS - 1.0) * state.internal
        enthalpy_density = np.maximum(state.density + state.internal + gas_pressure, SMALL)
        sigma = bsq / enthalpy_density
        fit_base = 1.0 - 8.0 * proton_pressure / enthalpy_density
        # KHARMA forms beta and beta_max separately.  At exactly zero field
        # their ratio is inf/inf even though the algebraic limit below exists.
        valid = (fit_base >= 0.0) & np.isfinite(bsq) & (bsq > 0.0)
        if robust_rowan:
            fit_power = np.maximum(fit_base, 0.0) ** 3.3
        else:
            fit_power = np.full_like(fit_base, np.nan)
            fit_power[valid] = fit_base[valid] ** 3.3
        fraction = 0.5 * np.exp(-fit_power / (1.0 + 1.2 * sigma**0.7))
    else:
        electron_to_ion = 0.33 * np.sqrt(1.0 / temperature_ratio)
        fraction = electron_to_ion / (1.0 + electron_to_ion)
        valid = np.ones_like(fraction, dtype=bool)
    return np.clip(fraction, 0.0, 1.0), valid


def state_diagnostics(state: State, disk_density: float, code: str) -> dict[str, object]:
    disk = state.density > disk_density
    volume_weight = state.proper_volume
    mass_weight = state.density * volume_weight
    result: dict[str, object] = {
        "time": state.time,
        "cycle": state.cycle,
        "disk_cells": int(np.count_nonzero(disk)),
        "density_max": float(np.max(state.density)),
        "internal_energy_integral": float(np.sum(state.internal * volume_weight)),
        "magnetic_energy_integral": float(
            np.sum(0.5 * state.magnetic_squared * volume_weight)
        ),
        "max_abs_nonfinite": int(
            sum(
                array.size - np.count_nonzero(np.isfinite(array))
                for array in (
                    state.density,
                    state.internal,
                    state.velocity,
                    state.magnetic,
                    state.ktot,
                    *state.kel.values(),
                )
            )
        ),
        "electron": {},
    }
    electron_result: dict[str, object] = {}
    for model in MODEL_NAMES:
        tp_over_te, theta_e, electron_internal = temperature_fields(state, model)
        fraction, rowan_valid = heating_fraction(state, model, code == "pangu")
        valid_disk = disk & rowan_valid
        entropy_integral = float(
            np.sum(state.electron_cons[model]) * state.native_cell_volume
        )
        model_result: dict[str, object] = {
            "internal_energy_integral": float(np.sum(electron_internal * volume_weight)),
            "conserved_entropy_integral": entropy_integral,
            "tp_over_te_mass_quantiles": weighted_quantile(tp_over_te[disk], mass_weight[disk]),
            "theta_e_mass_quantiles": weighted_quantile(theta_e[disk], mass_weight[disk]),
            "heating_fraction_mass_quantiles": weighted_quantile(
                fraction[valid_disk], mass_weight[valid_disk]
            ),
        }
        if model == "Rowan":
            temperature_limited = disk & (tp_over_te > 0.99 * 1.0e3)
            model_result["fit_domain_valid_cells"] = int(np.count_nonzero(valid_disk))
            model_result["fit_domain_invalid_cells"] = int(np.count_nonzero(disk & ~rowan_valid))
            model_result["fit_domain_invalid_mass_fraction"] = float(
                np.sum(mass_weight[disk & ~rowan_valid])
                / max(np.sum(mass_weight[disk]), np.finfo(np.float64).tiny)
            )
            model_result["upper_temperature_limit_cells"] = int(
                np.count_nonzero(temperature_limited)
            )
            model_result["upper_temperature_limit_mass_fraction"] = float(
                np.sum(mass_weight[temperature_limited])
                / max(np.sum(mass_weight[disk]), np.finfo(np.float64).tiny)
            )
        electron_result[model] = model_result
    result["electron"] = electron_result
    result["ktot_conserved_integral"] = float(
        np.sum(state.electron_cons["Ktot"]) * state.native_cell_volume
    )
    if state.raw_dissipation is not None:
        kawazura_fraction, _ = heating_fraction(state, "Kawazura", True)
        kawazura_tpte, _, _ = temperature_fields(state, "Kawazura")
        result["last_substep_dissipation"] = {
            "raw_volume_integral": float(np.sum(state.raw_dissipation * volume_weight)),
            "applied_volume_integral": float(
                np.sum(state.applied_dissipation * volume_weight)
            ),
            "flagged_cells": int(np.count_nonzero(state.diagnostic_flags)),
            "kawazura_fraction_relative_l1": relative_l1(
                state.diagnostic_fraction, kawazura_fraction, disk
            ),
            "kawazura_tp_over_te_relative_l1": relative_l1(
                state.diagnostic_temperature_ratio, kawazura_tpte, disk
            ),
        }
    return result


def direct_disk_comparison(left: State, right: State, disk_density: float) -> dict[str, float]:
    disk = right.density > disk_density
    comparison: dict[str, float] = {}
    fields = {
        "density": (left.density, right.density),
        "internal_energy": (left.internal, right.internal),
        "velocity_1": (left.velocity[:, 0], right.velocity[:, 0]),
        "velocity_2": (left.velocity[:, 1], right.velocity[:, 1]),
        "velocity_3": (left.velocity[:, 2], right.velocity[:, 2]),
        "magnetic_1": (left.magnetic[:, 0], right.magnetic[:, 0]),
        "magnetic_2": (left.magnetic[:, 1], right.magnetic[:, 1]),
        "magnetic_3": (left.magnetic[:, 2], right.magnetic[:, 2]),
        "Ktot": (left.ktot, right.ktot),
    }
    for name, (left_field, right_field) in fields.items():
        if float(np.max(np.abs(right_field[disk]))) < 1.0e-14:
            comparison[f"{name}_absolute_linf"] = float(
                np.max(np.abs(left_field[disk] - right_field[disk]))
            )
        else:
            comparison[name] = relative_l1(left_field, right_field, disk)
    for model in MODEL_NAMES:
        mask = disk
        if model == "Rowan" and left.time > 0.0:
            _, valid_left = heating_fraction(left, model, True)
            _, valid_right = heating_fraction(right, model, False)
            right_tpte, _, _ = temperature_fields(right, model)
            mask = disk & valid_left & valid_right & (right_tpte < 0.99 * 1.0e3)
            comparison["Kel_Rowan_full_disk"] = relative_l1(
                left.kel[model], right.kel[model], disk
            )
            comparison["Kel_Rowan_fit_domain_unclipped"] = relative_l1(
                left.kel[model], right.kel[model], mask
            )
        else:
            comparison[f"Kel_{model}"] = relative_l1(
                left.kel[model], right.kel[model], mask
            )
    return comparison


def cross_code_statistics(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "internal_energy_integral_relative_difference": relative_difference(
            float(left["internal_energy_integral"]), float(right["internal_energy_integral"])
        ),
        "magnetic_energy_integral_relative_difference": relative_difference(
            float(left["magnetic_energy_integral"]), float(right["magnetic_energy_integral"])
        ),
        "electron": {},
    }
    electron: dict[str, object] = {}
    for model in MODEL_NAMES:
        left_model = left["electron"][model]
        right_model = right["electron"][model]
        model_result: dict[str, object] = {
            "internal_energy_integral_relative_difference": relative_difference(
                float(left_model["internal_energy_integral"]),
                float(right_model["internal_energy_integral"]),
            ),
            "conserved_entropy_integral_relative_difference": relative_difference(
                float(left_model["conserved_entropy_integral"]),
                float(right_model["conserved_entropy_integral"]),
            ),
        }
        for name in (
            "tp_over_te_mass_quantiles",
            "theta_e_mass_quantiles",
            "heating_fraction_mass_quantiles",
        ):
            lq = np.asarray(left_model[name], dtype=np.float64)
            rq = np.asarray(right_model[name], dtype=np.float64)
            model_result[f"{name}_relative_difference"] = [
                relative_difference(float(a), float(b)) for a, b in zip(lq, rq)
            ]
        electron[model] = model_result
    result["electron"] = electron
    return result


def main() -> int:
    args = parse_args()
    pangu_files = snapshots(args.pangu, "gr_torus_sane.prim.*.phdf")
    kharma_files = snapshots(args.kharma, "torus.out0.*.phdf")
    if not pangu_files or not kharma_files:
        raise RuntimeError(
            f"missing snapshots: PANGU={len(pangu_files)} KHARMA={len(kharma_files)}"
        )
    pairs: list[tuple[Path, Path]] = []
    for pangu_time, pangu_path in pangu_files:
        kharma_time, kharma_path = min(
            kharma_files, key=lambda item: abs(item[0] - pangu_time)
        )
        if abs(kharma_time - pangu_time) > args.time_tolerance:
            raise RuntimeError(
                f"output-time mismatch: PANGU={pangu_time} KHARMA={kharma_time}"
            )
        pairs.append((pangu_path, kharma_path))

    records: list[dict[str, object]] = []
    initial_entropy: dict[str, dict[str, float]] | None = None
    for index, (pangu_path, kharma_path) in enumerate(pairs):
        pangu = pangu_state(pangu_path, args.spin, args.hslope)
        kharma = kharma_state(kharma_path, args.spin, args.hslope)
        pangu_diag = state_diagnostics(pangu, args.disk_density, "pangu")
        kharma_diag = state_diagnostics(kharma, args.disk_density, "kharma")
        if initial_entropy is None:
            initial_entropy = {
                "pangu": {
                    name: float(pangu_diag["electron"][name]["conserved_entropy_integral"])
                    for name in MODEL_NAMES
                },
                "kharma": {
                    name: float(kharma_diag["electron"][name]["conserved_entropy_integral"])
                    for name in MODEL_NAMES
                },
            }
        for code, diagnostic in (("pangu", pangu_diag), ("kharma", kharma_diag)):
            for model in MODEL_NAMES:
                current = float(diagnostic["electron"][model]["conserved_entropy_integral"])
                initial = initial_entropy[code][model]
                diagnostic["electron"][model]["net_entropy_change_over_initial"] = (
                    current - initial
                ) / max(abs(initial), np.finfo(np.float64).tiny)
        record: dict[str, object] = {
            "pangu_file": pangu_path.name,
            "kharma_file": kharma_path.name,
            "time_separation": abs(pangu.time - kharma.time),
            "pangu": pangu_diag,
            "kharma": kharma_diag,
            "statistical_relative_difference": cross_code_statistics(pangu_diag, kharma_diag),
        }
        if index == 0 or (len(pairs) == 2 and pangu.time <= 1.0e-4):
            record["direct_disk_relative_l1"] = direct_disk_comparison(
                pangu, kharma, args.disk_density
            )
        records.append(record)

    nonfinite = sum(
        int(record[code]["max_abs_nonfinite"])
        for record in records
        for code in ("pangu", "kharma")
    )
    accepted_models = ("Kawazura", "Werner", "Sharma")
    maxima = {
        "base_internal_energy_integral": 0.0,
        "base_magnetic_energy_integral": 0.0,
        "electron_internal_energy_integral": 0.0,
        "electron_conserved_entropy_integral": 0.0,
        "tp_over_te_quantile": 0.0,
        "theta_e_quantile": 0.0,
        "heating_fraction_quantile": 0.0,
        "net_entropy_change_over_initial_absolute": 0.0,
        "rowan_heating_fraction_quantile": 0.0,
    }
    for record in records:
        statistical = record["statistical_relative_difference"]
        maxima["base_internal_energy_integral"] = max(
            maxima["base_internal_energy_integral"],
            float(statistical["internal_energy_integral_relative_difference"]),
        )
        maxima["base_magnetic_energy_integral"] = max(
            maxima["base_magnetic_energy_integral"],
            float(statistical["magnetic_energy_integral_relative_difference"]),
        )
        for model in accepted_models:
            model_difference = statistical["electron"][model]
            maxima["electron_internal_energy_integral"] = max(
                maxima["electron_internal_energy_integral"],
                float(model_difference["internal_energy_integral_relative_difference"]),
            )
            maxima["electron_conserved_entropy_integral"] = max(
                maxima["electron_conserved_entropy_integral"],
                float(model_difference["conserved_entropy_integral_relative_difference"]),
            )
            maxima["tp_over_te_quantile"] = max(
                maxima["tp_over_te_quantile"],
                *model_difference["tp_over_te_mass_quantiles_relative_difference"],
            )
            maxima["theta_e_quantile"] = max(
                maxima["theta_e_quantile"],
                *model_difference["theta_e_mass_quantiles_relative_difference"],
            )
            maxima["heating_fraction_quantile"] = max(
                maxima["heating_fraction_quantile"],
                *model_difference["heating_fraction_mass_quantiles_relative_difference"],
            )
            pangu_change = float(
                record["pangu"]["electron"][model]["net_entropy_change_over_initial"]
            )
            kharma_change = float(
                record["kharma"]["electron"][model]["net_entropy_change_over_initial"]
            )
            maxima["net_entropy_change_over_initial_absolute"] = max(
                maxima["net_entropy_change_over_initial_absolute"],
                abs(pangu_change - kharma_change),
            )
        maxima["rowan_heating_fraction_quantile"] = max(
            maxima["rowan_heating_fraction_quantile"],
            *statistical["electron"]["Rowan"][
                "heating_fraction_mass_quantiles_relative_difference"
            ],
        )

    initial_direct = records[0]["direct_disk_relative_l1"]
    limits = {
        "initial_density_relative_l1": 2.0e-6,
        "initial_ktot_relative_l1": 2.0e-2,
        "base_internal_energy_integral": 1.0e-2,
        "base_magnetic_energy_integral": 1.0e-2,
        "electron_internal_energy_integral": 5.0e-3,
        "electron_conserved_entropy_integral": 5.0e-3,
        "tp_over_te_quantile": 8.0e-2,
        "theta_e_quantile": 5.0e-2,
        "heating_fraction_quantile": 2.0e-2,
        "net_entropy_change_over_initial_absolute": 5.0e-3,
        "rowan_heating_fraction_quantile": 2.0e-3,
    }
    actual = {
        "initial_density_relative_l1": float(initial_direct["density"]),
        "initial_ktot_relative_l1": float(initial_direct["Ktot"]),
        **maxima,
    }
    violations = {
        name: {"actual": actual[name], "limit": limit}
        for name, limit in limits.items()
        if actual[name] > limit
    }
    passed = nonfinite == 0 and not violations
    report = {
        "schema_version": 1,
        "case": "MKS/static SANE electron heating",
        "comparison": "PANGU versus KHARMA",
        "definitions": {
            "theta_e": "(m_p/m_e) Kel rho^(gamma_e-1)",
            "tp_over_te": "[(gamma_p-1) u/rho] / [Kel rho^(gamma_e-1)]",
            "electron_internal_energy": "Kel rho^gamma_e / (gamma_e-1)",
            "heating_integral_proxy": (
                "initial-subtracted domain integral of the densitized conserved electron "
                "entropy; it includes boundary transport and is not interpreted as a local source integral"
            ),
            "distribution_weighting": "proper-volume rest-mass weighting over rho > threshold",
            "rowan_policy": (
                "strict comparison uses 1-beta/beta_max >= 0; PANGU clamps the fit argument "
                "outside that domain while KHARMA falls through NaN clipping to Kel_min"
            ),
        },
        "parameters": {
            "spin": args.spin,
            "hslope": args.hslope,
            "gamma_gas": GAMMA_GAS,
            "gamma_electron": GAMMA_ELECTRON,
            "gamma_proton": GAMMA_PROTON,
            "disk_density_threshold": args.disk_density,
            "time_tolerance": args.time_tolerance,
            "mass_quantiles": QUANTILES.tolist(),
        },
        "records": records,
        "nonfinite_values": nonfinite,
        "acceptance": {
            "accepted_models": list(accepted_models),
            "rowan_scope": "heating-fraction distribution only; Kel evolution is diagnostic",
            "actual_maxima": actual,
            "limits": limits,
            "violations": violations,
            "passed": passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    initial = records[0]["direct_disk_relative_l1"]
    print(
        f"PANGU/KHARMA electron SANE comparison {'PASS' if passed else 'FAIL'}: "
        f"snapshots={len(records)}, initial rho={initial['density']:.3e}, "
        f"Ktot={initial['Ktot']:.3e}, nonfinite={nonfinite}, violations={len(violations)}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        print(f"PANGU/KHARMA electron SANE comparison FAIL: {error}")
        raise SystemExit(1)
