#!/usr/bin/env python3
"""Measure compile-time and active-cooling overhead on one CUDA GRMHD problem."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import subprocess
import tempfile


PERFORMANCE = re.compile(
    r"cycle=(?P<cycle>\d+).*zone-cycles/wsec_step=(?P<throughput>[0-9.eE+-]+)"
    r".*wsec_total=(?P<total>[0-9.eE+-]+).*wsec_step=(?P<interval>[0-9.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--none-executable", required=True, type=pathlib.Path)
    parser.add_argument("--cooling-executable", required=True, type=pathlib.Path)
    parser.add_argument("--none-input", required=True, type=pathlib.Path)
    parser.add_argument("--cooling-input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--nlim", type=int, default=300)
    parser.add_argument("--warmup-nlim", type=int, default=300)
    parser.add_argument("--report-interval", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--maximum-strong-overhead", type=float, default=0.05)
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args()


def run(
    executable: pathlib.Path,
    input_path: pathlib.Path,
    directory: pathlib.Path,
    nlim: int,
    report_interval: int,
    radiation_overrides: tuple[str, ...],
) -> tuple[list[dict[str, float | int]], str]:
    command = [
        str(executable.resolve()),
        "-i",
        str(input_path.resolve()),
        f"parthenon/time/nlim={nlim}",
        f"parthenon/time/ncycle_out={report_interval}",
        "parthenon/time/ncycle_out_mesh=0",
        "problem/pert_amp=0.0",
        "parthenon/output1/dt=1.0e9",
        "parthenon/output2/dt=1.0e9",
        "parthenon/output3/dt=1.0e9",
        *radiation_overrides,
    ]
    completed = subprocess.run(
        command,
        cwd=directory,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed with {completed.returncode}\n{completed.stdout[-8000:]}"
        )
    samples: list[dict[str, float | int]] = []
    for match in PERFORMANCE.finditer(completed.stdout):
        samples.append(
            {
                "cycle": int(match.group("cycle")),
                "throughput_zone_cycles_per_s": float(match.group("throughput")),
                "elapsed_total_s": float(match.group("total")),
                "elapsed_interval_s": float(match.group("interval")),
            }
        )
    steady = [sample for sample in samples if int(sample["cycle"]) >= 2 * report_interval]
    if len(steady) < 2:
        raise RuntimeError(f"insufficient steady performance samples: {samples}")
    return steady, completed.stdout


def main() -> int:
    args = parse_args()
    if args.nlim < 4 * args.report_interval:
        raise ValueError("--nlim must span at least four reporting intervals")
    if args.warmup_nlim < 4 * args.report_interval:
        raise ValueError("--warmup-nlim must span at least four reporting intervals")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    cases = {
        "none": (args.none_executable, args.none_input, ()),
        "weak_cooling": (
            args.cooling_executable,
            args.cooling_input,
            (
                "radiation/start_time=0.0",
                "radiation/ramp_time=0.0",
                "radiation/beta_cool=6.283185307179586",
            ),
        ),
        "strong_cooling": (
            args.cooling_executable,
            args.cooling_input,
            (
                "radiation/start_time=0.0",
                "radiation/ramp_time=0.0",
                "radiation/beta_cool=0.6283185307179586",
            ),
        ),
    }
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    repeat_records_by_label: dict[str, list[dict[str, object]]] = {
        label: [] for label in cases
    }
    labels = list(cases)
    with tempfile.TemporaryDirectory(prefix="rc6-performance-", dir=workdir) as temporary:
        root = pathlib.Path(temporary)
        # Precondition the GPU before either measured endpoint.  Without this
        # sweep, the first no-radiation case can run at a transient cold-GPU
        # boost clock and bias the inferred radiation overhead upward.
        warmup_dir = root / "gpu_warmup"
        warmup_dir.mkdir()
        warmup_samples, warmup_log = run(
            args.cooling_executable,
            args.cooling_input,
            warmup_dir,
            args.warmup_nlim,
            args.report_interval,
            (
                "radiation/start_time=0.0",
                "radiation/ramp_time=0.0",
                "radiation/beta_cool=0.6283185307179586",
            ),
        )
        for repeat in range(args.repeats):
            # Reverse every other sweep so thermal drift cannot systematically
            # favour the no-radiation baseline or penalize strong cooling.
            order = labels if repeat % 2 == 0 else list(reversed(labels))
            for label in order:
                executable, input_path, overrides = cases[label]
                directory = root / f"{label}_{repeat:02d}"
                directory.mkdir()
                samples, log = run(
                    executable,
                    input_path,
                    directory,
                    args.nlim,
                    args.report_interval,
                    overrides,
                )
                repeat_records_by_label[label].append(
                    {
                        "repeat": repeat,
                        "samples": samples,
                        "median_throughput_zone_cycles_per_s": statistics.median(
                            float(sample["throughput_zone_cycles_per_s"]) for sample in samples
                        ),
                        "log": log,
                    }
                )
        for label in labels:
            repeat_records = repeat_records_by_label[label]
            records[label] = {
                "repeats": repeat_records,
                "median_throughput_zone_cycles_per_s": statistics.median(
                    float(record["median_throughput_zone_cycles_per_s"])
                    for record in repeat_records
                ),
            }

    baseline = float(records["none"]["median_throughput_zone_cycles_per_s"])
    overheads = {
        label: 1.0 - float(record["median_throughput_zone_cycles_per_s"]) / baseline
        for label, record in records.items()
        if label != "none"
    }
    strong_pass = overheads["strong_cooling"] <= args.maximum_strong_overhead
    report = {
        "schema_version": 1,
        "status": "pass" if strong_pass or not args.enforce else "fail",
        "enforced": args.enforce,
        "nlim": args.nlim,
        "warmup_nlim": args.warmup_nlim,
        "warmup": {"samples": warmup_samples, "log": warmup_log},
        "report_interval": args.report_interval,
        "repeats": args.repeats,
        "cases": records,
        "overhead_relative_to_none": overheads,
        "maximum_strong_overhead": args.maximum_strong_overhead,
        "strong_overhead_target_met": strong_pass,
    }
    output = (args.output or workdir / "radiation_performance_gate.json").resolve()
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"PANGU radiation RC-6 performance {report['status'].upper()}: "
        f"strong_overhead={overheads['strong_cooling']:.3%} report={output}"
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
