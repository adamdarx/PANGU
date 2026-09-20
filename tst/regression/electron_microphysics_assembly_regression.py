#!/usr/bin/env python3
"""Run all kinetic electron-heating prescriptions in one GPU assembly gate."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile

import h5py
import numpy as np


KINETIC_LABELS = ["Kel_Howes", "Kel_Kawazura", "Kel_Werner", "Kel_Rowan", "Kel_Sharma"]


def run(command: list[str], directory: pathlib.Path) -> None:
    result = subprocess.run(command, cwd=directory, text=True, capture_output=True)
    (directory / "run.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed; see {directory / 'run.log'}")


def final_dump(directory: pathlib.Path) -> pathlib.Path:
    paths = sorted(directory.glob("*.out2.final.phdf"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one final PHDF in {directory}, found {len(paths)}")
    return paths[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--metric", required=True, choices=("cks", "mks"))
    parser.add_argument("--mode", required=True, choices=("dynamic", "static"))
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    case = pathlib.Path(
        tempfile.mkdtemp(prefix=f"eh3-{args.metric}-{args.mode}-", dir=args.workdir)
    )
    heated = case / "heated"
    passive = case / "passive"
    heated.mkdir()
    passive.mkdir()
    common = [
        str(args.executable.resolve()),
        "-i",
        str(args.input.resolve()),
        f"geometry/expected_metric={args.metric}",
        f"geometry/expected_mode={args.mode}",
        "parthenon/time/nlim=1",
        "electrons/constant=false",
        "electrons/howes=true",
        "electrons/kawazura=true",
        "electrons/werner=true",
        "electrons/rowan=true",
        "electrons/sharma=true",
        "electrons/suppress_highb_heat=false",
        "electrons/write_diagnostics=true",
        "parthenon/output2/variables=mhd.prim,mhd.cons,electrons.prim,"
        "electrons.cons,electrons.diagnostics,mhd.fofc",
    ]
    run(common + ["electrons/heating=true"], heated)
    run(common + ["electrons/heating=false"], passive)

    with h5py.File(final_dump(heated), "r") as active, h5py.File(
        final_dump(passive), "r"
    ) as control:
        primitive = active["mhd.prim"][...]
        conserved = active["mhd.cons"][...]
        electron = active["electrons.prim"][...]
        electron_cons = active["electrons.cons"][...]
        diagnostics = active["electrons.diagnostics"][...]
        expected_names = ["electrons.prim_Ktot"] + [
            f"electrons.prim_{label}" for label in KINETIC_LABELS
        ]
        component_names = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in active["Info"].attrs["ComponentNames"]
        ]
        missing = [name for name in expected_names if name not in component_names]
        if missing:
            raise RuntimeError(f"kinetic component labels are missing: {missing}")
        if electron.shape[1] != 1 + len(KINETIC_LABELS):
            raise RuntimeError(f"unexpected electron component count {electron.shape[1]}")
        if not all(
            np.isfinite(array).all()
            for array in (primitive, conserved, electron, electron_cons, diagnostics)
        ):
            raise RuntimeError("kinetic assembly gate contains NaN or Inf")

        scale = max(float(np.max(np.abs(electron_cons))), np.finfo(np.float64).tiny)
        u_equals_dk = float(
            np.max(np.abs(electron_cons - conserved[:, 0:1, ...] * electron)) / scale
        )
        if u_equals_dk > 5.0e-15:
            raise RuntimeError(f"kinetic U_K=D*K residual is {u_equals_dk:.17e}")

        gamma = 4.0 / 3.0
        energy_entropy = (gamma - 1.0) * primitive[:, 4, ...] * primitive[:, 0, ...] ** (-gamma)
        entropy_relative = float(
            np.max(
                np.abs(electron[:, 0, ...] - energy_entropy)
                / np.maximum(np.abs(energy_entropy), np.finfo(np.float64).tiny)
            )
        )
        if entropy_relative > 5.0e-15:
            raise RuntimeError(
                f"kinetic Ktot synchronization residual is {entropy_relative:.17e}"
            )

        mhd_linf = {
            name: float(np.max(np.abs(active[name][...] - control[name][...])))
            for name in ("mhd.prim", "mhd.cons")
        }
        if any(value != 0.0 for value in mhd_linf.values()):
            raise RuntimeError(f"kinetic electron heating changed MHD: {mhd_linf}")

        fractions = diagnostics[:, 2, ...]
        fraction_range = [float(np.min(fractions)), float(np.max(fractions))]
        if fraction_range[0] < 0.0 or fraction_range[1] > 1.0:
            raise RuntimeError(f"heating fraction left [0,1]: {fraction_range}")
        levels = sorted(set(int(level) for level in active["Levels"][...]))

    if args.metric == "cks" and args.mode == "dynamic":
        if levels != [0, 1]:
            raise RuntimeError(f"CKS/Dynamic kinetic gate did not retain SMR: {levels}")
    elif levels != [0]:
        raise RuntimeError(f"non-SMR kinetic assembly has unexpected levels: {levels}")

    report = {
        "assembly": {"metric": args.metric, "mode": args.mode, "backend": "cuda"},
        "models": KINETIC_LABELS,
        "levels": levels,
        "heating_fraction_range": fraction_range,
        "u_equals_dk_relative": u_equals_dk,
        "ktot_energy_pointwise_relative": entropy_relative,
        "mhd_linf_heating_vs_disabled": mhd_linf,
    }
    (case / "eh3-assembly-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"EH-3 {args.metric.upper()}/{args.mode.capitalize()} kinetic heating PASS: "
        f"levels={levels}, fraction={fraction_range}, MHD passive identity"
    )
    print(case)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"EH-3 kinetic assembly heating FAIL: {error}")
        raise SystemExit(1)
