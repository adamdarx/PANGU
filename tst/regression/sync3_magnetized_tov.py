#!/usr/bin/env python3
"""Audit the SYNC-3 magnetized-TOV stability gate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


def snapshot(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as stream:
        fields = {
            name: np.asarray(stream[name], dtype=np.float64)
            for name in (
                "mhd.prim", "mhd.cons", "mhd.b_cell", "mhd.b_face", "mhd.divb",
                "nr.z4c", "nr.adm", "nr.tmunu", "nr.constraints",
            )
        }
        for name, values in fields.items():
            if not np.isfinite(values).all():
                raise RuntimeError(f"nonfinite values in {path.name}:{name}")
        primitive = fields["mhd.prim"]
        speed = np.sqrt(np.sum(primitive[:, 1:4] ** 2, axis=1))
        core = primitive[:, 0] > 1.0e-5
        if not np.any(core):
            raise RuntimeError(f"empty dense-core mask in {path.name}")
        magnetic = np.sqrt(np.sum(fields["mhd.b_cell"] ** 2, axis=1))
        constraints = np.abs(fields["nr.constraints"])
        return {
            "time": float(stream["Info"].attrs["Time"]),
            "density_maximum": float(np.max(primitive[:, 0])),
            "dense_core_speed_maximum": float(np.max(speed[core])),
            "magnetic_field_maximum": float(np.max(magnetic)),
            "maximum_abs_divb": float(np.max(np.abs(fields["mhd.divb"]))),
            "combined_constraint_maximum": float(np.max(constraints[:, 0])),
            "hamiltonian_constraint_maximum": float(np.max(constraints[:, 1])),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    if (args.pangu is None) != (args.input is None):
        raise RuntimeError("--pangu and --input must be supplied together")
    if args.pangu is not None:
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.pangu.resolve()), "-i", str(args.input.resolve()),
            "parthenon/output1/dt=0.5",
        ]
        completed = subprocess.run(
            command, cwd=workdir, env=os.environ.copy(), check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (workdir / "run.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0 or "Driver completed." not in completed.stdout:
            raise RuntimeError(
                f"command failed ({completed.returncode}): {' '.join(command)}\n"
                f"{completed.stdout[-8000:]}"
            )
    paths = sorted(workdir.glob("*sync_grmhd_tov*.phdf"))
    if len(paths) < 3:
        raise RuntimeError(f"need at least three magnetized-TOV snapshots in {workdir}")
    samples = [snapshot(path) for path in paths]
    initial, final = samples[0], samples[-1]
    density_drift = abs(final["density_maximum"] - initial["density_maximum"]) / initial[
        "density_maximum"
    ]
    magnetic_drift = abs(
        final["magnetic_field_maximum"] - initial["magnetic_field_maximum"]
    ) / initial["magnetic_field_maximum"]
    if final["time"] < 2.0 - 1.0e-14:
        raise RuntimeError(f"magnetized-TOV run ended early at {final['time']}")
    if density_drift > 1.0e-2:
        raise RuntimeError(f"excessive central-density drift: {density_drift}")
    if magnetic_drift > 3.0e-2:
        raise RuntimeError(f"excessive magnetic-field drift: {magnetic_drift}")
    if final["dense_core_speed_maximum"] > 1.0e-2:
        raise RuntimeError(
            f"excessive dense-core velocity: {final['dense_core_speed_maximum']}"
        )
    if max(sample["maximum_abs_divb"] for sample in samples) > 1.0e-12:
        raise RuntimeError("magnetized-TOV divergence constraint exceeded tolerance")
    if final["combined_constraint_maximum"] > 1.05 * initial[
        "combined_constraint_maximum"
    ]:
        raise RuntimeError("magnetized-TOV maximum constraint grew unexpectedly")

    report = {
        "schema": "pangu.sync3.magnetized-tov.v2",
        "pass": True,
        "relative_central_density_drift": density_drift,
        "relative_peak_magnetic_field_drift": magnetic_drift,
        "samples": samples,
    }
    (workdir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
