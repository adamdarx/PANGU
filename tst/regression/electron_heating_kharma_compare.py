#!/usr/bin/env python3
"""Compare PANGU and KHARMA EH-2 Noh evolution in self-similar coordinates."""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np


GAMMA_GAS = 5.0 / 3.0
GAMMA_E = 4.0 / 3.0
HEATING_FRACTION = 0.5


def pangu_cell(data: h5py.File, name: str) -> np.ndarray:
    array = np.asarray(data[name])
    return array[:, :, 0, 0, :].transpose(0, 2, 1).reshape(-1, array.shape[1])


def load_pangu(path: pathlib.Path) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as data:
        time = float(data["Info"].attrs["Time"])
        x = np.asarray(data["VolumeLocations/x"]).reshape(-1)
        primitive = pangu_cell(data, "mhd.prim")
        electron = pangu_cell(data, "electrons.prim")
    electron_internal = electron[:, 1] * primitive[:, 0] ** GAMMA_E / (GAMMA_E - 1.0)
    fields = {
        "density": primitive[:, 0],
        "internal_energy": primitive[:, 4],
        "spatial_four_velocity": primitive[:, 1],
        "ue_over_ug": electron_internal / np.maximum(primitive[:, 4], np.finfo(np.float64).tiny),
        "ktot": electron[:, 0],
        "kel_constant": electron[:, 1],
    }
    order = np.argsort(x)
    return time, x[order], {name: value[order] for name, value in fields.items()}


def load_kharma(path: pathlib.Path) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as data:
        time = float(data["Info"].attrs["Time"])
        x = np.asarray(data["VolumeLocations/x"]).reshape(-1)
        density = np.asarray(data["prims.rho"]).reshape(-1)
        internal = np.asarray(data["prims.u"]).reshape(-1)
        ktot = np.asarray(data["prims.Ktot"]).reshape(-1)
        kel = np.asarray(data["prims.Kel_Constant"]).reshape(-1)
        velocity = (
            np.asarray(data["prims.uvec"])
            .transpose(0, 2, 3, 4, 1)
            .reshape(-1, 3)[:, 0]
        )
    electron_internal = kel * density**GAMMA_E / (GAMMA_E - 1.0)
    fields = {
        "density": density,
        "internal_energy": internal,
        "spatial_four_velocity": velocity,
        "ue_over_ug": electron_internal / np.maximum(internal, np.finfo(np.float64).tiny),
        "ktot": ktot,
        "kel_constant": kel,
    }
    order = np.argsort(x)
    return time, x[order], {name: value[order] for name, value in fields.items()}


def analytic_postshock_ratio() -> float:
    gamma = GAMMA_GAS
    gamma_e = GAMMA_E
    return (
        HEATING_FRACTION
        / 2.0
        * (
            ((gamma + 1.0) / (gamma - 1.0)) ** gamma_e * (1.0 - gamma / gamma_e)
            + 1.0
            + gamma / gamma_e
        )
        * ((gamma * gamma - 1.0) / (gamma_e * gamma_e - 1.0))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu-directory", required=True, type=pathlib.Path)
    parser.add_argument("--kharma-directory", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    pangu_files = sorted(args.pangu_directory.glob("electron_noh.out2*.phdf"))
    kharma_files = sorted(args.kharma_directory.glob("noh.out0*.phdf"))
    if len(pangu_files) != 5 or len(kharma_files) != 5:
        raise RuntimeError(
            f"expected five files per code, found PANGU={len(pangu_files)} KHARMA={len(kharma_files)}"
        )

    analytic_ratio = analytic_postshock_ratio()
    records: list[dict[str, object]] = []
    for pangu_path, kharma_path in zip(pangu_files, kharma_files):
        pangu_time, pangu_x, pangu = load_pangu(pangu_path)
        kharma_time, kharma_x, kharma = load_kharma(kharma_path)
        record: dict[str, object] = {
            "pangu_time": pangu_time,
            "kharma_time": kharma_time,
        }
        if pangu_time == 0.0 and kharma_time == 0.0:
            record["initial_linf"] = {
                name: float(np.max(np.abs(pangu[name] - kharma[name]))) for name in pangu
            }
        else:
            pangu_similarity = pangu_x / pangu_time
            kharma_similarity = kharma_x / kharma_time
            mask = np.abs(pangu_similarity) < 1.5e-3
            l1: dict[str, float] = {}
            for name in pangu:
                reference = np.interp(pangu_similarity[mask], kharma_similarity, kharma[name])
                scale = max(float(np.mean(np.abs(reference))), 1.0e-300)
                l1[name] = float(np.mean(np.abs(pangu[name][mask] - reference)) / scale)
            pangu_shocked = pangu["density"] > 1.5
            kharma_shocked = kharma["density"] > 1.5
            pangu_shock = float(np.max(np.abs(pangu_x[pangu_shocked])))
            kharma_shock = float(np.max(np.abs(kharma_x[kharma_shocked])))
            pangu_reference = np.where(pangu_shocked, analytic_ratio, 0.0)
            kharma_reference = np.where(kharma_shocked, analytic_ratio, 0.0)
            record.update(
                {
                    "self_similar_relative_l1": l1,
                    "shock_speed": {
                        "pangu": pangu_shock / pangu_time,
                        "kharma": kharma_shock / kharma_time,
                    },
                    "analytic_ue_over_ug_l1": {
                        "pangu": float(np.mean(np.abs(pangu["ue_over_ug"] - pangu_reference))),
                        "kharma": float(np.mean(np.abs(kharma["ue_over_ug"] - kharma_reference))),
                    },
                }
            )
        records.append(record)

    final_l1 = records[-1]["self_similar_relative_l1"]
    limits = {
        "density": 2.0e-2,
        "internal_energy": 3.0e-2,
        "spatial_four_velocity": 1.0e-2,
        "ue_over_ug": 2.0e-2,
        "ktot": 3.0e-2,
        "kel_constant": 3.0e-2,
    }
    violations = {
        name: {"actual": final_l1[name], "limit": limit}
        for name, limit in limits.items()
        if final_l1[name] > limit
    }
    if violations:
        raise RuntimeError(f"final PANGU/KHARMA Noh comparison failed: {violations}")

    report = {
        "comparison": "PANGU_KHARMA_EH2_centered_Noh",
        "coordinate": "x_over_t",
        "analytic_postshock_ue_over_ug": analytic_ratio,
        "records": records,
        "final_acceptance_limits": limits,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "PANGU/KHARMA EH-2 Noh multitime PASS: "
        + ", ".join(f"{name}={value:.3e}" for name, value in final_l1.items())
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"PANGU/KHARMA EH-2 comparison FAIL: {error}")
        raise SystemExit(1)
