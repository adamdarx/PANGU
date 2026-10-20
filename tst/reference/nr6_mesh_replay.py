#!/usr/bin/env python3
"""Replay actual tracker centers through both native mesh trees; compare ALL leaves.

This is a geometry correctness test, not an evolution/timing benchmark. It does
not prescribe tracker coordinates to either production solver.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--snapshots", type=Path)
    parser.add_argument("--tracker", type=Path)
    parser.add_argument("--self-test", action="store_true",
                        help="Generate a moving-sphere mesh correctness test")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--pangu-trace", type=Path,
                        help="Optional single-rank, single-tracker production trace")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        if args.snapshots or args.tracker or args.pangu_trace:
            parser.error("--self-test cannot use external evidence")
        leaves = {(1, i, j, k) for i in range(2) for j in range(2) for k in range(2)}
        for level in range(1, 5):
            width = math.ldexp(64., -level)
            refine = [loc for loc in leaves if loc[0] == level and
                      all(-32. + q * width < 4. and -32. + (q + 1) * width > -4.
                          for q in loc[1:])]
            for loc in refine:
                leaves.remove(loc)
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            leaves.add((level + 1, 2 * loc[1] + i,
                                        2 * loc[2] + j, 2 * loc[3] + k))
        assert len(leaves) == 232
        args.snapshots = args.workdir / "synthetic-initial.json"
        args.snapshots.write_text(json.dumps([dict(cycle=0, blocks=[
            dict(logical=[i, j, k, level - 1]) for level, i, j, k in sorted(leaves)])]))
        args.tracker = args.workdir / "synthetic-tracker.txt"
        # Resolve face/edge/corner crossings and multi-level coarsening, then
        # return to the origin. No solver or prescribed production motion.
        positions = [(0., 0., 0.)]
        positions += [(2.3e-16 * sign, 0., 0.) for sign in (-1, 1)]
        positions += [(12. * math.sin(n / 20.), 8. * math.sin(n / 31.),
                       6. * math.sin(n / 43.)) for n in range(160)]
        positions += [(0., 0., 0.)] * 16
        args.tracker.write_text("".join(
            f"{n} {n * .025:.17g} {x:.17g} {y:.17g} {z:.17g}\n"
            for n, (x, y, z) in enumerate(positions)))
    elif not args.snapshots or not args.tracker:
        parser.error("provide --snapshots and --tracker, or --self-test")
    snapshots = json.loads(args.snapshots.read_text())
    initial, = [s for s in snapshots if s["cycle"] == 0]
    # AthenaK bin logical[3] is PHYSICAL refinement level, not logical level.
    def topology(snapshot):
        return sorted((b["logical"][3] + 1, *b["logical"][:3])
                      for b in snapshot["blocks"])
    seed = args.workdir / "seed.txt"
    seed.write_text("".join(" ".join(map(str, loc)) + "\n" for loc in topology(initial)))
    rows = [line.split() for line in args.tracker.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    if [int(row[0]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("Tracker records must contain every consecutive cycle from zero")
    centers = args.workdir / "centers.txt"
    centers.write_text("".join(" ".join(row[2:5]) + "\n" for row in rows))
    output = args.workdir / "topology.txt"
    run = subprocess.run([str(args.replay.resolve()), str(seed.resolve()),
                          str(centers.resolve()), str(output.resolve())],
                         text=True, capture_output=True)
    (args.workdir / "replay.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)
    replay = {}
    for line in output.read_text().splitlines():
        cycle, count, *blocks = line.split()
        values = sorted(tuple(map(int, item.split(","))) for item in blocks)
        assert len(values) == int(count)
        replay[int(cycle)] = values
    evidence = []
    for snapshot in snapshots:
        cycle = snapshot["cycle"]
        # Athena outputs after evolution but BEFORE that step's mesh update.
        before_amr = max(0, cycle - 1)
        expected = topology(snapshot)
        actual = replay[before_amr]
        if expected != actual:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise RuntimeError(f"Saved AthenaK snapshot mismatch cycle={cycle}: "
                               f"missing={missing[:4]}, extra={extra[:4]}")
        evidence.append(dict(cycle=cycle, blocks=len(actual), matched=True))
    block_cycles = sum(len(replay[n]) for n in range(len(rows)))
    summary = dict(updates=len(rows), every_native_tree_leaf_set_matched=True,
                   meshblock_cycles=block_cycles, snapshots=evidence,
                   tracker_sha256=hashlib.sha256(args.tracker.read_bytes()).hexdigest(),
                   snapshots_sha256=hashlib.sha256(args.snapshots.read_bytes()).hexdigest(),
                   limitation="Shared-trajectory tree replay, not independent evolution.")
    if args.pangu_trace:
        first_mesh_difference = first_center_difference = None
        traced_steps = 0
        for line in args.pangu_trace.read_text().splitlines():
            step, x, y, z, count, *blocks = line.split()
            step = int(step)
            if step > len(rows):
                raise RuntimeError("PANGU trace exceeds reference trajectory")
            locations = sorted(tuple(map(int, item.split(","))) for item in blocks)
            assert len(locations) == int(count)
            if locations != replay[step - 1] and first_mesh_difference is None:
                first_mesh_difference = step
            if [float(x), float(y), float(z)] != list(map(float, rows[step - 1][2:5])) \
                    and first_center_difference is None:
                first_center_difference = step
            traced_steps += 1
        summary["independent_evolution"] = dict(
            traced_steps=traced_steps, first_mesh_difference=first_mesh_difference,
            first_center_difference=first_center_difference,
            complete=len(rows) == traced_steps,
            passed=traced_steps == len(rows) and first_mesh_difference is None)
    (args.workdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(run.stdout.strip())
    label = "synthetic seed" if args.self_test else "real AthenaK snapshot"
    print(f"Matched {len(evidence)} {label} full leaf sets; "
          f"block-cycles={block_cycles}")
    if args.pangu_trace:
        independent = summary["independent_evolution"]
        print("Independent evolution:", independent)
        if not independent["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
