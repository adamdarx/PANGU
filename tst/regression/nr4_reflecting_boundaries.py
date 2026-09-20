#!/usr/bin/env python3
"""Check tensor parity in all six physical NR reflecting ghost slabs."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess

import h5py
import numpy as np


def parity(component: int, direction: int) -> float:
    # component layout is fixed by z4c/component_indices.h.
    if 1 <= component <= 6 or 8 <= component <= 13:
        local = component - (1 if component <= 6 else 8)
        pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
        first, second = pairs[local]
        return -1.0 if ((first == direction) ^ (second == direction)) else 1.0
    if 14 <= component <= 16:
        return -1.0 if component - 14 == direction else 1.0
    if 19 <= component <= 21:
        return -1.0 if component - 19 == direction else 1.0
    return 1.0


def check_face(values: np.ndarray, direction: int, inner: bool, nghost: int) -> float:
    _, ncomp, nk, nj, ni = values.shape
    sizes = (nk, nj, ni)
    nactive = sizes[direction] - 2 * nghost
    source = nghost if inner else nghost + nactive - 1
    worst = 0.0
    for depth in range(nghost):
        ghost = nghost - 1 - depth if inner else nghost + nactive + depth
        reflected = 2 * source + (-1 if inner else 1) - ghost
        for component in range(ncomp):
            expected = parity(component, direction)
            if direction == 0:
                observed = values[:, component, :, :, ghost]
                reference = values[:, component, :, :, reflected]
            elif direction == 1:
                observed = values[:, component, :, ghost, :]
                reference = values[:, component, :, reflected, :]
            else:
                observed = values[:, component, ghost, :, :]
                reference = values[:, component, reflected, :, :]
            worst = max(worst, float(np.max(np.abs(observed - expected * reference))))
    return worst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    parser.add_argument("--nghost", type=int, default=2)
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
        raise RuntimeError(f"NR-4 reflecting run failed ({result.returncode})\n{result.stdout}")

    snapshots = sorted(args.workdir.glob("*.phdf"))
    if not snapshots:
        raise RuntimeError("NR-4 reflecting run did not write a PHDF snapshot")
    with h5py.File(snapshots[-1], "r") as data:
        values = np.asarray(data["nr.z4c"])
        cycle = int(data["Info"].attrs["NCycle"])
        time = float(data["Info"].attrs["Time"])
    if values.ndim != 5 or values.shape[1] != 22:
        raise RuntimeError(f"unexpected nr.z4c shape with ghost zones: {values.shape}")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("reflecting boundary produced non-finite Z4c values")
    nghost = args.nghost
    if nghost < 1:
        raise RuntimeError("--nghost must be positive")
    if any(size <= 2 * nghost for size in values.shape[2:]):
        raise RuntimeError(f"PHDF output does not contain active cells and ghosts: {values.shape}")
    errors = [
        check_face(values, direction, inner, nghost)
        for direction in range(3)
        for inner in (True, False)
    ]
    worst = max(errors)
    tolerance = 64.0 * np.finfo(values.dtype).eps
    if worst > tolerance:
        raise RuntimeError(f"tensor-parity mismatch {worst:.6e} exceeds {tolerance:.6e}")
    print(
        "NR-4 reflecting tensor parity PASS: "
        f"snapshots={len(snapshots)} cycle={cycle} time={time:.17e} "
        f"shape={tuple(values.shape)} max_error={worst:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
