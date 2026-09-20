#!/usr/bin/env python3
"""Validate coarse/fine NR tensor-parity boundaries on a GPU AMR run."""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess

import h5py
import numpy as np


PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def reflection_sign(component: int, direction: int) -> int:
    if 1 <= component <= 6:
        first, second = PAIRS[component - 1]
    elif 8 <= component <= 13:
        first, second = PAIRS[component - 8]
    elif 14 <= component <= 16:
        return -1 if component - 14 == direction else 1
    elif 19 <= component <= 21:
        return -1 if component - 19 == direction else 1
    else:
        return 1
    return -1 if (first == direction) ^ (second == direction) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--mpi-launcher", default="mpirun")
    args = parser.parse_args()
    executable = args.executable.resolve()
    input_file = args.input.resolve()

    if args.workdir.exists():
        shutil.rmtree(args.workdir)
    args.workdir.mkdir(parents=True)
    command = [str(executable), "-i", str(input_file)]
    if args.nproc > 1:
        command = [args.mpi_launcher]
        if os.geteuid() == 0:
            command.append("--allow-run-as-root")
        command += ["--mca", "opal_cuda_support", "1", "-np", str(args.nproc)]
        command += [str(executable), "-i", str(input_file)]
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
        raise RuntimeError(f"NR-4 AMR reflecting run failed ({result.returncode})\n{result.stdout}")

    snapshots = sorted(args.workdir.glob("*.phdf"))
    if not snapshots:
        raise RuntimeError("NR-4 AMR reflecting run did not write a PHDF snapshot")
    with h5py.File(snapshots[-1], "r") as data:
        z4c = np.asarray(data["nr.z4c"])
        levels = np.asarray(data["Levels"])
        locations = np.asarray(data["LogicalLocations"])
        num_blocks = int(data["Info"].attrs["NumMeshBlocks"])

    if z4c.ndim != 5 or z4c.shape[1] != 22 or not np.isfinite(z4c).all():
        raise RuntimeError(f"unexpected or non-finite AMR Z4c output: {z4c.shape}")
    if int(levels.max()) < 1 or num_blocks <= 4:
        raise RuntimeError(f"AMR did not create fine blocks: levels={levels}, blocks={num_blocks}")

    nghost = 2
    max_locations = locations.max(axis=0)
    max_error = 0.0
    comparisons = 0
    for block, (x, y, z) in enumerate(locations):
        faces = []
        if x == 0:
            faces.append((0, True))
        if x == max_locations[0]:
            faces.append((0, False))
        if y == 0:
            faces.append((1, True))
        if y == max_locations[1]:
            faces.append((1, False))
        if z == 0:
            faces.append((2, True))
        if z == max_locations[2]:
            faces.append((2, False))
        for direction, inner in faces:
            for component in range(22):
                sign = reflection_sign(component, direction)
                for layer in range(nghost):
                    if direction == 0:
                        ghost = layer if inner else z4c.shape[-1] - nghost + layer
                        mirror = 2 * nghost - 1 - layer if inner else z4c.shape[-1] - nghost - 1 - layer
                        actual = z4c[block, component, nghost:-nghost, nghost:-nghost, ghost]
                        expected = sign * z4c[block, component, nghost:-nghost, nghost:-nghost, mirror]
                    elif direction == 1:
                        ghost = layer if inner else z4c.shape[-2] - nghost + layer
                        mirror = 2 * nghost - 1 - layer if inner else z4c.shape[-2] - nghost - 1 - layer
                        actual = z4c[block, component, nghost:-nghost, ghost, nghost:-nghost]
                        expected = sign * z4c[block, component, nghost:-nghost, mirror, nghost:-nghost]
                    else:
                        ghost = layer if inner else z4c.shape[-3] - nghost + layer
                        mirror = 2 * nghost - 1 - layer if inner else z4c.shape[-3] - nghost - 1 - layer
                        actual = z4c[block, component, ghost, nghost:-nghost, nghost:-nghost]
                        expected = sign * z4c[block, component, mirror, nghost:-nghost, nghost:-nghost]
                    max_error = max(max_error, float(np.max(np.abs(actual - expected))))
                    comparisons += actual.size

    if max_error > 64 * np.finfo(z4c.dtype).eps:
        raise RuntimeError(f"tensor-parity residual too large: {max_error:.6e}")
    print(
        "NR-4 coarse/fine reflecting AMR PASS: "
        f"blocks={num_blocks} max_level={int(levels.max())} "
        f"comparisons={comparisons} max_error={max_error:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
