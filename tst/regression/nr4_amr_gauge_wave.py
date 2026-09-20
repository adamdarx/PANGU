#!/usr/bin/env python3
"""Check device-side Z4c AMR tagging and coarse/fine state transfer."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess

import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    input_file = args.input.resolve()

    if args.workdir.exists():
        shutil.rmtree(args.workdir)
    args.workdir.mkdir(parents=True)
    result = subprocess.run(
        [str(executable), "-i", str(input_file)],
        cwd=args.workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=300,
    )
    (args.workdir / "run.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"NR-4 AMR run failed ({result.returncode})\n{result.stdout}")

    snapshots = sorted(args.workdir.glob("*.phdf"))
    if not snapshots:
        raise RuntimeError("NR-4 AMR run did not write a PHDF snapshot")
    with h5py.File(snapshots[-1], "r") as data:
        z4c = np.asarray(data["nr.z4c"])
        levels = np.asarray(data["Levels"])
        info = data["Info"].attrs
        max_level = int(np.max(levels))
        num_blocks = int(info["NumMeshBlocks"])

    if z4c.ndim != 5 or z4c.shape[1] != 22:
        raise RuntimeError(f"unexpected AMR Z4c shape: {z4c.shape}")
    if not np.isfinite(z4c).all():
        raise RuntimeError("AMR output contains non-finite Z4c values")
    if max_level < 1:
        raise RuntimeError(f"chi gradient criterion did not refine any block: levels={levels}")
    if num_blocks <= 4 or len(levels) <= 4:
        raise RuntimeError(f"AMR did not create fine blocks: NumMeshBlocks={num_blocks}")
    print(
        "NR-4 Z4c AMR PASS: "
        f"snapshots={len(snapshots)} NumMeshBlocks={num_blocks} max_level={max_level} "
        f"shape={tuple(z4c.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
