#!/usr/bin/env python3
"""GPU evolution and convergence gates for the NR-2 vacuum Z4c stage."""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import re
import shutil
import subprocess


def run_case(
    executable: pathlib.Path,
    input_file: pathlib.Path,
    workdir: pathlib.Path,
    overrides: list[str],
    pattern: str,
) -> tuple[float, str]:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    command = [str(executable), "-i", str(input_file), *overrides]
    environment = dict(os.environ)
    environment["CUDA_LAUNCH_BLOCKING"] = "1"
    result = subprocess.run(
        command,
        cwd=workdir,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"NR-2 run failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    match = re.search(pattern, result.stdout)
    if match is None:
        raise RuntimeError(f"NR-2 diagnostic was not found in output:\n{result.stdout}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise RuntimeError(f"NR-2 diagnostic is non-finite: {value}")
    return value, result.stdout


def common_output_overrides() -> list[str]:
    return ["parthenon/output1/dt=-1", "parthenon/output2/dt=-1"]


def observed_orders(errors: list[float]) -> list[float]:
    return [math.log(errors[index] / errors[index + 1], 2.0) for index in range(len(errors) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=pathlib.Path, required=True)
    parser.add_argument("--linear-input", type=pathlib.Path, required=True)
    parser.add_argument("--gauge-input", type=pathlib.Path, required=True)
    parser.add_argument("--robust-input", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    resolutions = (32, 64, 128)
    linear_errors: list[float] = []
    gauge_errors: list[float] = []
    one_period = 1.0 / math.sqrt(3.0)
    for resolution in resolutions:
        linear, _ = run_case(
            args.executable,
            args.linear_input,
            args.workdir / f"linear-o2-{resolution}",
            [
                f"parthenon/mesh/nx1={resolution}",
                f"parthenon/mesh/nx2={resolution}",
                f"parthenon/mesh/nx3={resolution}",
                f"parthenon/meshblock/nx1={resolution}",
                f"parthenon/meshblock/nx2={resolution}",
                f"parthenon/meshblock/nx3={resolution}",
                "parthenon/mesh/nghost=2",
                "parthenon/time/integrator=rk2",
                f"parthenon/time/tlim={one_period:.17g}",
                "numerical_relativity/finite_difference_order=2",
                "problem/kx1=1",
                "problem/kx2=1",
                "problem/kx3=1",
                *common_output_overrides(),
            ],
            r"NR-2 linear wave:.*L1_RMS=([0-9.eE+-]+)",
        )
        linear_errors.append(linear)

        gauge, _ = run_case(
            args.executable,
            args.gauge_input,
            args.workdir / f"gauge-o2-{resolution}",
            [
                f"parthenon/mesh/nx1={resolution}",
                f"parthenon/meshblock/nx1={resolution}",
                "parthenon/mesh/nghost=2",
                "parthenon/time/integrator=rk4",
                "numerical_relativity/finite_difference_order=2",
                *common_output_overrides(),
            ],
            r"NR-2 gauge wave:.*L1=([0-9.eE+-]+)",
        )
        gauge_errors.append(gauge)

    linear_orders = observed_orders(linear_errors)
    gauge_orders = observed_orders(gauge_errors)
    if min(linear_orders) < 1.8:
        raise RuntimeError(f"linear-wave convergence is too slow: {linear_orders}")
    if min(gauge_orders) < 1.8:
        raise RuntimeError(f"gauge-wave convergence is too slow: {gauge_orders}")
    if linear_errors[1] > 3.5e-11:
        raise RuntimeError(f"64^3 second-order linear-wave error is too large: {linear_errors[1]}")
    if linear_errors[1] / linear_errors[0] > 0.25:
        raise RuntimeError(
            "32^3->64^3 second-order linear-wave error ratio is too large: "
            f"{linear_errors[1] / linear_errors[0]}"
        )

    rk3, _ = run_case(
        args.executable,
        args.linear_input,
        args.workdir / "linear-rk3-32",
        [
            "parthenon/mesh/nx1=32",
            "parthenon/mesh/nx2=32",
            "parthenon/mesh/nx3=32",
            "parthenon/meshblock/nx1=32",
            "parthenon/meshblock/nx2=32",
            "parthenon/meshblock/nx3=32",
            "parthenon/mesh/nghost=2",
            "parthenon/time/integrator=rk3",
            "parthenon/time/tlim=1.0",
            "numerical_relativity/finite_difference_order=2",
            "problem/kx1=1",
            "problem/kx2=0",
            "problem/kx3=0",
            *common_output_overrides(),
        ],
        r"NR-2 linear wave:.*L1_RMS=([0-9.eE+-]+)",
    )
    if rk3 > 2.0e-10:
        raise RuntimeError(f"32^3 RK3 linear-wave error is too large: {rk3}")

    sixth_order, _ = run_case(
        args.executable,
        args.linear_input,
        args.workdir / "linear-o6-64",
        [
            "parthenon/mesh/nx1=64",
            "parthenon/mesh/nx2=64",
            "parthenon/mesh/nx3=64",
            "parthenon/meshblock/nx1=64",
            "parthenon/meshblock/nx2=64",
            "parthenon/meshblock/nx3=64",
            "parthenon/mesh/nghost=4",
            "parthenon/time/integrator=rk4",
            f"parthenon/time/tlim={one_period:.17g}",
            "numerical_relativity/finite_difference_order=6",
            "problem/kx1=1",
            "problem/kx2=1",
            "problem/kx3=1",
            *common_output_overrides(),
        ],
        r"NR-2 linear wave:.*L1_RMS=([0-9.eE+-]+)",
    )
    if sixth_order > 6.0e-12:
        raise RuntimeError(f"64^3 sixth-order linear-wave error is too large: {sixth_order}")

    robust, _ = run_case(
        args.executable,
        args.robust_input,
        args.workdir / "robust-1000-steps",
        [*common_output_overrides()],
        r"NR-2 robust stability:.*RMS=([0-9.eE+-]+)",
    )
    if robust > 1.0e-6:
        raise RuntimeError(f"robust-stability RMS grew beyond the acceptance envelope: {robust}")

    print(
        "NR-2 evolution PASS: "
        f"linear={linear_errors} orders={linear_orders}; "
        f"gauge={gauge_errors} orders={gauge_orders}; "
        f"linear_rk3_32={rk3}; linear_o6_64={sixth_order}; robust_rms={robust}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
