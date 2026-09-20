#!/usr/bin/env python3
"""Pass only when a command fails with the requested diagnostic."""

from __future__ import annotations

import argparse
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--needle", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode == 0:
        raise RuntimeError("command unexpectedly succeeded")
    if args.needle not in completed.stdout:
        raise RuntimeError(
            f"failure did not contain {args.needle!r}:\n{completed.stdout}"
        )
    print(f"expected failure observed: {args.needle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
