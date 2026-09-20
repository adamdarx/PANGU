#!/usr/bin/env python3
"""Run the stage-1 advection problem and optionally prove HDF5 restart."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile


def run(command: list[str], cwd: pathlib.Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--workdir", required=True, type=pathlib.Path)
    parser.add_argument("--require-restart", action="store_true")
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    case_dir = pathlib.Path(tempfile.mkdtemp(prefix="case-", dir=args.workdir))
    initial_log = run([str(args.executable.resolve()), "-i", str(args.input.resolve())],
                      case_dir)
    (case_dir / "initial.log").write_text(initial_log, encoding="utf-8")

    error_file = case_dir / "pangu-advection-errors.dat"
    if not error_file.is_file() or len(error_file.read_text(encoding="utf-8").splitlines()) < 2:
        raise RuntimeError("advection error diagnostic was not produced")

    history = list(case_dir.glob("*.hst"))
    if not history:
        raise RuntimeError("history output was not produced")

    if args.require_restart:
        outputs = list(case_dir.glob("*.phdf"))
        restarts = sorted(case_dir.glob("*.rhdf"))
        if not outputs or not restarts:
            raise RuntimeError("HDF5 field output or restart output was not produced")
        restart_log = run(
            [str(args.executable.resolve()), "-r", str(restarts[-1].resolve()),
             "parthenon/time/tlim=0.08"],
            case_dir,
        )
        (case_dir / "restart.log").write_text(restart_log, encoding="utf-8")
        if "restart" not in restart_log.lower():
            raise RuntimeError("restart run did not identify itself as a restart")

    print(f"stage-1 smoke PASS: {case_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - a test harness should print one failure
        print(f"stage-1 smoke FAIL: {error}", file=sys.stderr)
        raise
