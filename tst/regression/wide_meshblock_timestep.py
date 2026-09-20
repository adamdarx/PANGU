#!/usr/bin/env python3
"""Check the unforced first timestep on MeshBlocks 512 and 1024 cells wide.

Device reductions over such blocks once returned the reducer identity (512) or
aborted (1024), so the first cycle is run without dt_init or dt_force.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import tempfile
from pathlib import Path

CYCLE_ZERO = re.compile(r"^cycle=0 time=\S+ dt=(\S+)", re.MULTILINE)


def first_timestep(executable: Path, deck: Path, width: int, workdir: Path) -> float:
    command = [
        str(executable), "-i", str(deck),
        "parthenon/time/nlim=1",
        "parthenon/time/ncycle_out=1",
        f"parthenon/mesh/nx1={width}",
        f"parthenon/meshblock/nx1={width}",
        "parthenon/output1/dt=1.0e9",
        "parthenon/output2/dt=1.0e9",
    ]
    completed = subprocess.run(command, cwd=workdir, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"nx1={width} run failed:\n{completed.stdout[-4000:]}")
    match = CYCLE_ZERO.search(completed.stdout)
    if match is None:
        raise RuntimeError(f"nx1={width} printed no cycle-0 timestep")
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    # Drop the conservative startup step so the estimator alone sets cycle 0.
    text = args.input.read_text(encoding="utf-8")
    text = re.sub(r"^\s*dt_init\s*=.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*dt_factor\s*=.*$", "", text, flags=re.MULTILINE)
    with tempfile.TemporaryDirectory(dir=args.workdir) as scratch:
        workdir = Path(scratch)
        deck = workdir / "deck.in"
        deck.write_text(text, encoding="utf-8")
        steps = {width: first_timestep(args.executable, deck, width, workdir)
                 for width in (256, 512, 1024)}

    for width, dt in steps.items():
        print(f"nx1={width}: first dt = {dt:.16e}")
        if not (math.isfinite(dt) and 0.0 < dt < 1.0):
            raise RuntimeError(f"nx1={width} first timestep {dt} is not a CFL-limited value")
    # Halving the radial spacing must reduce the step; an identity result would not.
    if not steps[1024] < steps[512] < steps[256]:
        raise RuntimeError(f"first timesteps do not decrease with resolution: {steps}")
    print("wide MeshBlock first timesteps PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
