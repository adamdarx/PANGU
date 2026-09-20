#!/usr/bin/env python3
"""Interleaved CUDA wall-time gate for the SYNC-GRMHD magnetized TOV test.

The benchmark measures cycles 20--100 from each application's own elapsed-time
counter.  Initialization, the first 20 warm-up cycles, the final ten cycles,
and all simulation output are excluded from the primary comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import shutil
import statistics
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
ACTIVE_CELLS = 64**3
WINDOW_BEGIN = 20
WINDOW_END = 100
NVIDIA_SMI = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(command: list[str], cwd: pathlib.Path = ROOT) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip()


def pangu_command(executable: pathlib.Path, input_file: pathlib.Path) -> list[str]:
    return [
        str(executable),
        "-i",
        str(input_file),
        "parthenon/time/nlim=110",
        "parthenon/time/tlim=1000000",
        "parthenon/time/ncycle_out=10",
        "parthenon/time/ncycle_out_mesh=1000000",
        "parthenon/output1/dt=-1",
        "parthenon/output2/dt=-1",
        "problem/write_final_csv=false",
    ]


def athenak_command(executable: pathlib.Path, input_file: pathlib.Path) -> list[str]:
    command = [
        str(executable),
        "-i",
        str(input_file),
        "time/nlim=110",
        "time/tlim=1000000",
        "time/ndiag=10",
    ]
    command.extend(f"output{index}/dt=-1" for index in range(1, 7))
    return command


def parse_measured_window(code: str, log: str) -> tuple[float, int]:
    elapsed: dict[int, float] = {}
    if code == "PANGU":
        pattern = r"cycle=(\d+)\s+time=\S+\s+dt=\S+.*?wsec_total=(\S+)"
        for match in re.finditer(pattern, log):
            elapsed[int(match.group(1))] = float(match.group(2))
    else:
        pattern = r"elapsed=(\S+)\s+cycle=(\d+)"
        for match in re.finditer(pattern, log):
            elapsed[int(match.group(2))] = float(match.group(1))
    missing = {WINDOW_BEGIN, WINDOW_END}.difference(elapsed)
    if missing:
        raise RuntimeError(f"{code} timing markers missing: {sorted(missing)}")
    return elapsed[WINDOW_END] - elapsed[WINDOW_BEGIN], WINDOW_END - WINDOW_BEGIN


def gpu_process_memory_mib(pid: int) -> int:
    output = capture(
        [
            NVIDIA_SMI,
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    peak = 0
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            if int(fields[0]) == pid:
                peak = max(peak, int(fields[1]))
        except ValueError:
            continue
    return peak


def gpu_total_memory_mib() -> int:
    output = capture(
        [
            NVIDIA_SMI,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        return int(output.splitlines()[0].strip())
    except (IndexError, ValueError):
        return 0


def run_once(
    code: str,
    command: list[str],
    run_dir: pathlib.Path,
    environment: dict[str, str],
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=False)
    baseline_total_memory_mib = gpu_total_memory_mib()
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=run_dir,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    peak_memory_mib = 0
    peak_total_memory_mib = baseline_total_memory_mib
    while process.poll() is None:
        peak_memory_mib = max(peak_memory_mib, gpu_process_memory_mib(process.pid))
        peak_total_memory_mib = max(peak_total_memory_mib, gpu_total_memory_mib())
        time.sleep(0.05)
    stdout, _ = process.communicate()
    total_wall_seconds = time.perf_counter() - started
    (run_dir / "run.log").write_text(stdout, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"{code} failed with {process.returncode}\n{stdout[-8000:]}")
    measured_seconds, measured_cycles = parse_measured_window(code, stdout)
    return {
        "code": code,
        "command": command,
        "measured_window": [WINDOW_BEGIN, WINDOW_END],
        "measured_cycles": measured_cycles,
        "measured_seconds": measured_seconds,
        "cycles_per_second": measured_cycles / measured_seconds,
        "cell_updates_per_second": ACTIVE_CELLS * measured_cycles / measured_seconds,
        "total_wall_seconds": total_wall_seconds,
        "peak_process_gpu_memory_mib": peak_memory_mib,
        "baseline_total_gpu_memory_mib": baseline_total_memory_mib,
        "peak_total_gpu_memory_mib": peak_total_memory_mib,
        "peak_total_gpu_memory_delta_mib": max(
            0, peak_total_memory_mib - baseline_total_memory_mib
        ),
        "log": str(run_dir / "run.log"),
    }


def summary(records: list[dict[str, object]], code: str) -> dict[str, float]:
    selected = [record for record in records if record["code"] == code]
    measured = [float(record["measured_seconds"]) for record in selected]
    throughput = [float(record["cell_updates_per_second"]) for record in selected]
    total = [float(record["total_wall_seconds"]) for record in selected]
    memory = [float(record["peak_process_gpu_memory_mib"]) for record in selected]
    memory_delta = [float(record["peak_total_gpu_memory_delta_mib"]) for record in selected]
    quartiles = (
        statistics.quantiles(measured, n=4, method="inclusive")
        if len(measured) > 1
        else [measured[0], measured[0], measured[0]]
    )
    return {
        "median_measured_seconds": statistics.median(measured),
        "iqr_measured_seconds": quartiles[2] - quartiles[0],
        "min_measured_seconds": min(measured),
        "median_cell_updates_per_second": statistics.median(throughput),
        "median_total_wall_seconds": statistics.median(total),
        "max_peak_process_gpu_memory_mib": max(memory),
        "max_peak_total_gpu_memory_delta_mib": max(memory_delta),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", type=pathlib.Path, required=True)
    parser.add_argument("--athenak", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    arguments = parser.parse_args()

    pangu = arguments.pangu.resolve()
    athenak = arguments.athenak.resolve()
    pangu_input = ROOT / "input/numerical_relativity/sync_grmhd_tov.in"
    athenak_input = ROOT / "input/numerical_relativity/athenak_sync_grmhd_tov.athinput"
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    allowed_cores = sorted(os.sched_getaffinity(0))
    selected_core = allowed_cores[0]
    commands = {
        "PANGU": ["taskset", "-c", str(selected_core), *pangu_command(pangu, pangu_input)],
        "AthenaK": [
            "taskset",
            "-c",
            str(selected_core),
            *athenak_command(athenak, athenak_input),
        ],
    }
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        }
    )

    jobs: list[tuple[int, str]] = []
    generator = random.Random(20260828)
    for repeat in range(arguments.repeats):
        pair = ["PANGU", "AthenaK"]
        generator.shuffle(pair)
        jobs.extend((repeat, code) for code in pair)

    records: list[dict[str, object]] = []
    for order, (repeat, code) in enumerate(jobs, start=1):
        print(f"[{order}/{len(jobs)}] repeat={repeat} code={code}", flush=True)
        record = run_once(
            code,
            commands[code],
            output / f"{repeat:02d}_{code.lower()}",
            environment,
        )
        record.update({"order": order, "repeat": repeat})
        records.append(record)
        print(
            f"  measured={record['measured_seconds']:.6f}s "
            f"throughput={record['cell_updates_per_second']:.6e} cell-updates/s "
            f"peak={record['peak_process_gpu_memory_mib']} MiB",
            flush=True,
        )

    summaries = {code: summary(records, code) for code in commands}
    ratio = (
        summaries["PANGU"]["median_measured_seconds"]
        / summaries["AthenaK"]["median_measured_seconds"]
    )
    result = {
        "benchmark": "SYNC-GRMHD magnetized TOV single-GPU wall time",
        "grid": [64, 64, 64],
        "meshblock": [32, 32, 32],
        "active_cells": ACTIVE_CELLS,
        "warmup_cycles": WINDOW_BEGIN,
        "measured_window": [WINDOW_BEGIN, WINDOW_END],
        "repeats": arguments.repeats,
        "interleaving_seed": 20260828,
        "selected_cpu_core": selected_core,
        "timing_definition": (
            "application-internal elapsed-time difference from cycle 20 to 100; "
            "initialization, warm-up, final cycles, diagnostics, and all file output excluded"
        ),
        "acceptance": {
            "target_pangu_over_athenak": "<= 1.10",
            "failure_pangu_over_athenak": "> 1.20",
            "observed_pangu_over_athenak": ratio,
            "target_passed": ratio <= 1.10,
        },
        "summaries": summaries,
        "records": records,
        "provenance": {
            "pangu_git": capture(["git", "rev-parse", "HEAD"]),
            "athenak_git": capture(["git", "-C", str(ROOT / "athenak"), "rev-parse", "HEAD"]),
            "pangu_binary": {"path": str(pangu), "sha256": sha256(pangu)},
            "athenak_binary": {"path": str(athenak), "sha256": sha256(athenak)},
            "pangu_input": {"path": str(pangu_input), "sha256": sha256(pangu_input)},
            "athenak_input": {"path": str(athenak_input), "sha256": sha256(athenak_input)},
            "nvidia_smi_before": capture(
                [
                    NVIDIA_SMI,
                    "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,memory.total",
                    "--format=csv,noheader",
                ]
            ),
            "nvcc": capture(["nvcc", "--version"]),
            "mpirun": capture(["mpirun", "--version"]),
            "environment": {
                name: environment.get(name, "")
                for name in (
                    "PATH",
                    "LD_LIBRARY_PATH",
                    "OMP_NUM_THREADS",
                    "OMP_PROC_BIND",
                    "OMP_PLACES",
                    "OMPI_MCA_opal_cuda_support",
                )
            },
        },
    }
    result["provenance"]["nvidia_smi_after"] = capture(
        [
            NVIDIA_SMI,
            "--query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used",
            "--format=csv,noheader",
        ]
    )
    report = output / "benchmark.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "acceptance": result["acceptance"]}, indent=2))
    return 0 if result["acceptance"]["target_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
