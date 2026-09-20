#!/usr/bin/env python3
"""Smoke-test a physical NR boundary and the non-physical constraint mask."""

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

    if args.workdir.exists():
        shutil.rmtree(args.workdir)
    args.workdir.mkdir(parents=True)
    command = [str(args.executable), "-i", str(args.input)]
    result = subprocess.run(
        command,
        cwd=args.workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=300,
    )
    (args.workdir / "run.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"NR physical-boundary run failed ({result.returncode})\n{result.stdout}")

    snapshots = sorted(args.workdir.glob("*.phdf"))
    if len(snapshots) < 2:
        raise RuntimeError(f"expected at least two PHDF snapshots, found {len(snapshots)}")
    with h5py.File(snapshots[-1], "r") as data:
        # Keep the block dimension: the puncture neighborhood spans the corner
        # of all eight blocks in this deliberately multi-block fixture.
        mask = np.asarray(data["nr.constraint_mask"])
        constraints = np.asarray(data["nr.constraints"])
        info = data["Info"].attrs
        cycle = int(info["NCycle"])
        time = float(info["Time"])

    if not np.all(np.isfinite(mask)) or not np.all(np.isfinite(constraints)):
        raise RuntimeError("physical-boundary run produced non-finite mask or constraints")
    if not np.all(np.isin(mask, [0.0, 1.0])):
        raise RuntimeError("constraint mask contains values other than 0 and 1")
    masked = int(np.count_nonzero(mask <= 0.5))
    active = int(np.count_nonzero(mask > 0.5))
    if masked == 0 or active == 0:
        raise RuntimeError(f"constraint mask does not split the domain: masked={masked} active={active}")

    histories = sorted(args.workdir.glob("*.hst"))
    if len(histories) != 1:
        raise RuntimeError(f"expected one history file, found {len(histories)}")
    history_text = histories[0].read_text(encoding="utf-8")
    if "nr_constraint_mask_cells_0" not in history_text or "nr_constraint_mask_cells_1" not in history_text:
        raise RuntimeError("mask-cell history columns are missing")
    history = np.loadtxt(histories[0], comments="#", ndmin=2)
    if history.size == 0 or not np.all(np.isfinite(history)):
        raise RuntimeError("constraint history contains no finite rows")

    print(
        "NR-3 physical boundary and mask PASS: "
        f"snapshots={len(snapshots)} cycle={cycle} time={time:.17e} "
        f"masked_cells={masked} active_cells={active} history_rows={len(history)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
