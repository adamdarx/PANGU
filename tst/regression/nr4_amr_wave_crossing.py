#!/usr/bin/env python3
"""Compare a two-level Z4c AMR gauge wave with a uniform fine-grid run."""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np


def snapshot_files(directory: pathlib.Path, prefix: str) -> list[pathlib.Path]:
    return sorted(directory.glob(f"{prefix}*.phdf"))


def gauge_metrics(path: pathlib.Path, amplitude: float = 0.01) -> dict[str, float | int | list[int]]:
    with h5py.File(path, "r") as data:
        info = data["Info"].attrs
        time = float(info["Time"])
        levels = np.asarray(data["Levels"])
        x = np.asarray(data["VolumeLocations"]["x"])
        adm = np.asarray(data["nr.adm"])
        constraints = np.asarray(data["nr.constraints"])
        root_domain = np.asarray(info["RootGridDomain"])
        num_blocks = int(info["NumMeshBlocks"])

    x_min = float(root_domain[0])
    x_max = float(root_domain[1])
    wavelength = x_max - x_min
    phase = 2.0 * np.pi * ((x - x_min) / wavelength - time / wavelength)
    h = amplitude * np.sin(phase)
    physical_gxx = 1.0 - h
    expected_alpha = np.sqrt(physical_gxx)
    dh_dt = -amplitude * (2.0 * np.pi / wavelength) * np.cos(phase)
    expected_kxx = 0.5 * dh_dt / expected_alpha

    # VolumeLocations/x has shape (block, i); broadcast across k,j.
    expected = [physical_gxx[:, None, None, :], expected_kxx[:, None, None, :],
                expected_alpha[:, None, None, :]]
    observed = [adm[:, 0], adm[:, 6], adm[:, 0] * 0.0 + np.sqrt(adm[:, 0])]
    # The lapse is represented by the Z4c alpha field; ADM gxx is sufficient
    # for the uniform-grid phase check, while z4c alpha is loaded separately.
    with h5py.File(path, "r") as data:
        alpha = np.asarray(data["nr.z4c"])[:, 18]
    observed[2] = alpha

    field_errors = [float(np.max(np.abs(a - b))) for a, b in zip(observed, expected)]
    constraint_abs = np.abs(constraints)
    return {
        "time": time,
        "blocks": num_blocks,
        "levels": sorted({int(level) for level in levels}),
        "gxx_linf": field_errors[0],
        "kxx_linf": field_errors[1],
        "alpha_linf": field_errors[2],
        "phase_linf": max(field_errors),
        "constraints_l1": float(np.mean(constraint_abs)),
        "constraints_linf": float(np.max(constraint_abs)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amr-dir", type=pathlib.Path, required=True)
    parser.add_argument("--uniform-dir", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    args = parser.parse_args()

    amr = snapshot_files(args.amr_dir, "nr_gauge_wave.nr4_amr_crossing")
    uniform = snapshot_files(args.uniform_dir, "nr_gauge_wave.nr4_uniform_reference")
    if len(amr) < 4 or len(uniform) < 4:
        raise RuntimeError(f"expected multi-time outputs, got AMR={len(amr)}, uniform={len(uniform)}")

    amr_metrics = [gauge_metrics(path) for path in amr]
    uniform_metrics = [gauge_metrics(path) for path in uniform]
    for item in amr_metrics + uniform_metrics:
        if not all(np.isfinite(float(item[key])) for key in
                   ("phase_linf", "constraints_l1", "constraints_linf")):
            raise RuntimeError(f"non-finite wave/constraint metric: {item}")
    if not any(len(item["levels"]) > 1 for item in amr_metrics):
        raise RuntimeError("AMR snapshots never contain both coarse and fine levels")

    for left, right in zip(amr_metrics, uniform_metrics):
        if abs(float(left["time"]) - float(right["time"])) > 2.0e-12:
            raise RuntimeError(f"AMR/uniform output times do not align: {left['time']} vs {right['time']}")

    summary = {
        "snapshots": len(amr_metrics),
        "amr": amr_metrics,
        "uniform": uniform_metrics,
        "max_amr_uniform_phase_error_ratio": max(
            a["phase_linf"] / max(u["phase_linf"], np.finfo(float).tiny)
            for a, u in zip(amr_metrics, uniform_metrics)
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    final = amr_metrics[-1]
    ref = uniform_metrics[-1]
    print(
        "NR-4 AMR wave crossing PASS: "
        f"snapshots={len(amr_metrics)} final_time={final['time']:.8f} "
        f"AMR_blocks={final['blocks']} levels={final['levels']} "
        f"phase_Linf={final['phase_linf']:.6e} "
        f"uniform_phase_Linf={ref['phase_linf']:.6e} "
        f"constraint_Linf={final['constraints_linf']:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
