#!/usr/bin/env python3
"""Check byte-exact HDF5 restart continuity for the stage-3 CT state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import subprocess
import tempfile


def run(command: list[str], directory: pathlib.Path, log_name: str) -> str:
    completed = subprocess.run(command, cwd=directory, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (directory / log_name).write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n"
                           f"{completed.stdout[-8000:]}")
    return completed.stdout


def profile(path: pathlib.Path) -> tuple[bytes, float, int]:
    payload = path.read_bytes()
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty final profile: {path}")
    max_divb = max(abs(float(row["divB"])) for row in rows)
    return payload, max_divb, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    case = pathlib.Path(tempfile.mkdtemp(prefix="mhd-restart-", dir=args.workdir))
    continuous = case / "continuous"
    split = case / "split"
    continuous.mkdir()
    split.mkdir()
    executable = str(args.executable.resolve())
    input_path = str(args.input.resolve())
    common = [
        executable, "-i", input_path,
        "parthenon/mesh/nx1=32", "parthenon/mesh/nx2=16",
        "parthenon/meshblock/nx1=32", "parthenon/meshblock/nx2=16",
        "parthenon/time/nlim=100", "parthenon/time/dt_force=0.005",
        "parthenon/output1/dt=1.0", "parthenon/output2/file_type=rst",
        "parthenon/output2/dt=0.005",
    ]
    run([*common, "parthenon/time/tlim=0.01"], continuous, "run.log")
    run([*common, "parthenon/time/tlim=0.005"], split, "initial.log")
    restart_files = sorted(split.glob("*.rhdf"))
    if not restart_files:
        raise RuntimeError("split run did not write a restart file")
    restart_log = run([executable, "-r", str(restart_files[-1].resolve()),
                       "parthenon/time/tlim=0.01"], split, "restart.log")
    if "Var: mhd.b_face:3" not in restart_log or "Var: mhd.cons:5" not in restart_log:
        raise RuntimeError("restart did not restore both independent CT/MHD variables")

    baseline, baseline_divb, row_count = profile(continuous / "pangu-mhd-final.csv")
    restarted, restarted_divb, restarted_rows = profile(split / "pangu-mhd-final.csv")
    if baseline != restarted or row_count != restarted_rows:
        raise RuntimeError("continuous and restarted MHD profiles are not byte-identical")
    max_divb = max(baseline_divb, restarted_divb)
    if max_divb > 1.0e-13:
        raise RuntimeError(f"restart divB={max_divb:.17e} exceeds gate")

    digest = hashlib.sha256(baseline).hexdigest()
    print(f"PANGU MHD restart PASS: rows={row_count} max_abs=0 "
          f"max_divB={max_divb:.17e} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
