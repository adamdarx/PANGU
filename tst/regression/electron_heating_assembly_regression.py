#!/usr/bin/env python3
"""Run an EH-2 GPU heating gate for one compile-time Geometry assembly."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile

import h5py
import numpy as np


def run(command: list[str], directory: pathlib.Path) -> None:
    result = subprocess.run(command, cwd=directory, text=True, capture_output=True)
    output = result.stdout + result.stderr
    (directory / "run.log").write_text(output, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed; see {directory / 'run.log'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--metric", required=True, choices=("cks", "mks"))
    parser.add_argument("--mode", required=True, choices=("dynamic", "static"))
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    executable = str(args.executable.resolve())
    input_path = str(args.input.resolve())
    args.workdir.mkdir(parents=True, exist_ok=True)
    case = pathlib.Path(tempfile.mkdtemp(prefix=f"eh2-{args.metric}-{args.mode}-", dir=args.workdir))
    enabled = case / "enabled"
    disabled = case / "disabled"
    enabled.mkdir()
    disabled.mkdir()
    common = [
        executable,
        "-i",
        input_path,
        f"geometry/expected_metric={args.metric}",
        f"geometry/expected_mode={args.mode}",
    ]
    run(
        common
        + [
            "electrons/heating=true",
            "electrons/write_diagnostics=true",
            "electrons/suppress_highb_heat=true",
            "electrons/sigma_heat_cutoff=1.0",
            "parthenon/output2/variables=mhd.prim,mhd.cons,electrons.prim,"
            "electrons.cons,electrons.diagnostics,mhd.fofc",
        ],
        enabled,
    )
    run(
        common
        + [
            "electrons/heating=false",
            "electrons/write_diagnostics=false",
            "parthenon/output2/variables=mhd.prim,mhd.cons,electrons.prim,electrons.cons,mhd.fofc",
        ],
        disabled,
    )

    enabled_path = enabled / "gr_monopole.out2.final.phdf"
    disabled_path = disabled / "gr_monopole.out2.final.phdf"
    with h5py.File(enabled_path, "r") as heated, h5py.File(disabled_path, "r") as passive:
        primitive = heated["mhd.prim"][...]
        conserved = heated["mhd.cons"][...]
        electron = heated["electrons.prim"][...]
        electron_cons = heated["electrons.cons"][...]
        diagnostics = heated["electrons.diagnostics"][...]
        if not all(
            np.isfinite(array).all()
            for array in (primitive, conserved, electron, electron_cons, diagnostics)
        ):
            raise RuntimeError("assembly heating gate contains NaN or Inf")
        scale = max(float(np.max(np.abs(electron_cons))), np.finfo(np.float64).tiny)
        u_equals_dk = float(
            np.max(np.abs(electron_cons - conserved[:, 0:1, ...] * electron)) / scale
        )
        if u_equals_dk > 5.0e-15:
            raise RuntimeError(f"assembly U_K=D*K residual is {u_equals_dk:.17e}")
        gamma = 4.0 / 3.0
        energy_entropy = (gamma - 1.0) * primitive[:, 4, ...] * primitive[:, 0, ...] ** (-gamma)
        entropy_difference = np.abs(electron[:, 0, ...] - energy_entropy)
        entropy_relative = float(
            np.max(entropy_difference / np.maximum(np.abs(energy_entropy), np.finfo(np.float64).tiny))
        )
        if entropy_relative > 5.0e-15:
            raise RuntimeError(f"assembly Ktot synchronization residual is {entropy_relative:.17e}")
        mhd_linf = {
            name: float(np.max(np.abs(heated[name][...] - passive[name][...])))
            for name in ("mhd.prim", "mhd.cons")
        }
        if any(value != 0.0 for value in mhd_linf.values()):
            raise RuntimeError(f"passive electron heating changed MHD: {mhd_linf}")
        levels = sorted(set(int(level) for level in heated["Levels"][...]))
        flags = sorted(set(int(flag) for flag in diagnostics[:, 4, ...].reshape(-1)))

    if args.metric == "cks" and args.mode == "dynamic":
        if levels != [0, 1]:
            raise RuntimeError(f"CKS/Dynamic heating did not retain SMR levels: {levels}")
    elif levels != [0]:
        raise RuntimeError(f"non-SMR assembly unexpectedly has levels {levels}")

    report = {
        "assembly": {"metric": args.metric, "mode": args.mode, "backend": "cuda"},
        "levels": levels,
        "u_equals_dk_relative": u_equals_dk,
        "ktot_energy_pointwise_relative": entropy_relative,
        "mhd_linf_heating_vs_disabled": mhd_linf,
        "heating_flags": flags,
    }
    (case / "eh2-assembly-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"EH-2 {args.metric.upper()}/{args.mode.capitalize()} heating PASS: "
        f"levels={levels}, MHD passive identity, U_K=D*K"
    )
    print(case)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"EH-2 assembly heating FAIL: {error}")
        raise SystemExit(1)
