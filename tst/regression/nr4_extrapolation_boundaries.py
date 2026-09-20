#!/usr/bin/env python3
"""Check AthenaK order-2/3/4 extrapolation in all physical NR ghost slabs."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess

import h5py
import numpy as np


def extrapolate(samples: tuple[np.ndarray, ...], distance: int, order: int) -> np.ndarray:
    """Use AthenaK z4c_bcs.cpp's expression tree for one ghost depth."""
    delta = float(distance)
    f0, f1 = samples[:2]
    if order == 2:
        return f0 + delta * (f0 - f1)
    f2 = samples[2]
    if order == 3:
        return 0.5 * (
            f0 * (1.0 + delta) * (2.0 + delta)
            + delta * (f2 + delta * f2 - 2.0 * f1 * (2.0 + delta))
        )
    f3 = samples[3]
    return (
        -3.0 * f1 * delta * (2.0 + delta) * (3.0 + delta)
        + f0 * (1.0 + delta) * (2.0 + delta) * (3.0 + delta)
        + delta
        * (1.0 + delta)
        * (-f3 * (2.0 + delta) + 3.0 * f2 * (3.0 + delta))
    ) / 6.0


def check_face(
    values: np.ndarray, direction: int, inner: bool, nghost: int, order: int
) -> float:
    _, _, nk, nj, ni = values.shape
    sizes = (nk, nj, ni)
    nactive = sizes[direction] - 2 * nghost
    source = nghost if inner else nghost + nactive - 1
    worst = 0.0
    # These components are scalar under all three face reflections and are not
    # changed by the algebraic metric/A projection that follows the boundary
    # task. They provide a round-off-level observable of the extrapolation
    # operation itself; tensor parity is covered by the companion test.
    components = (0, 7, 17, 18)  # chi, Khat, Theta, alpha
    for depth in range(nghost):
        ghost = nghost - 1 - depth if inner else nghost + nactive + depth
        distance = source - ghost if inner else ghost - source
        inward = 1 if inner else -1
        if direction == 0:
            # Parthenon applies x1, then x2, then x3 physical callbacks.  The
            # transverse x1 edges/corners are consequently owned and replaced
            # by the later callbacks; only the active transverse slab retains
            # the x1 expression in the final output.
            observed = values[:, components, nghost:-nghost, nghost:-nghost, ghost]
            samples = tuple(
                values[
                    :, components, nghost:-nghost, nghost:-nghost,
                    source + offset * inward,
                ]
                for offset in range(order)
            )
        elif direction == 1:
            # x2 owns the x1/x2 edges, but x3 subsequently owns all cells in
            # the x3 ghost slab.
            observed = values[:, components, nghost:-nghost, ghost, :]
            samples = tuple(
                values[:, components, nghost:-nghost, source + offset * inward, :]
                for offset in range(order)
            )
        else:
            observed = values[:, components, ghost, :, :]
            samples = tuple(
                values[:, components, source + offset * inward, :, :]
                for offset in range(order)
            )
        expected = extrapolate(samples, distance, order)
        worst = max(worst, float(np.max(np.abs(observed - expected))))
    return worst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    parser.add_argument("--nghost", type=int, default=2)
    parser.add_argument("--order", type=int, choices=(2, 3, 4), required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    input_file = args.input.resolve()

    if args.workdir.exists():
        shutil.rmtree(args.workdir)
    args.workdir.mkdir(parents=True)
    result = subprocess.run(
        [
            str(executable), "-i", str(input_file),
            f"numerical_relativity/boundary_extrapolation_order={args.order}",
        ],
        cwd=args.workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=300,
    )
    (args.workdir / "run.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"NR-4 extrapolation run failed ({result.returncode})\n{result.stdout}")

    snapshots = sorted(args.workdir.glob("*.phdf"))
    if not snapshots:
        raise RuntimeError("NR-4 extrapolation run did not write a PHDF snapshot")
    with h5py.File(snapshots[-1], "r") as data:
        values = np.asarray(data["nr.z4c"])
        cycle = int(data["Info"].attrs["NCycle"])
        time = float(data["Info"].attrs["Time"])
    if values.ndim != 5 or values.shape[1] != 22:
        raise RuntimeError(f"unexpected nr.z4c shape with ghost zones: {values.shape}")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("extrapolation boundary produced non-finite Z4c values")
    nghost = args.nghost
    if nghost < 1:
        raise RuntimeError("--nghost must be positive")
    if any(size <= 2 * nghost for size in values.shape[2:]):
        raise RuntimeError(f"PHDF output does not contain active cells and ghosts: {values.shape}")
    errors = [
        check_face(values, direction, inner, nghost, args.order)
        for direction in range(3)
        for inner in (True, False)
    ]
    worst = max(errors)
    tolerance = 512.0 * np.finfo(values.dtype).eps
    if worst > tolerance:
        raise RuntimeError(
            f"order-{args.order} extrapolation mismatch {worst:.6e} exceeds {tolerance:.6e}"
        )
    print(
        f"NR-4 scalar AthenaK order-{args.order} extrapolation PASS: "
        f"snapshots={len(snapshots)} cycle={cycle} time={time:.17e} "
        f"shape={tuple(values.shape)} max_error={worst:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
