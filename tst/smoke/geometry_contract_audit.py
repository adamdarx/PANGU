#!/usr/bin/env python3
"""Verify that every geometry consumer uses the reviewed Geometry interface."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys


LEGACY_PATTERN = re.compile(r"geometry::(Evaluate|Derivatives|KerrSchild)\s*\(")
ACCESS_PATTERN = re.compile(r"\.(MetricAt|DerivativesAt)\s*\(")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = manifest["geometry_access"]["by_file"]

    observed: dict[str, dict[str, int]] = {}
    legacy: dict[str, list[str]] = {}
    production_roots = (root / "src", root / "plugins")
    for source in sorted(path for tree in production_roots for path in tree.rglob("*")):
        if source.suffix not in {".cc", ".h"}:
            continue
        contents = source.read_text(encoding="utf-8")
        legacy_symbols = [match.group(1) for match in LEGACY_PATTERN.finditer(contents)]
        if legacy_symbols:
            legacy[source.relative_to(root).as_posix()] = legacy_symbols
        counts = {"MetricAt": 0, "DerivativesAt": 0}
        for match in ACCESS_PATTERN.finditer(contents):
            counts[match.group(1)] += 1
        if any(counts.values()):
            observed[source.relative_to(root).as_posix()] = counts

    if legacy:
        print("Legacy free-function geometry access remains.", file=sys.stderr)
        print(json.dumps(legacy, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    if observed != expected:
        print("Geometry access differs from the reviewed GEO-1 inventory.", file=sys.stderr)
        print("Expected:", json.dumps(expected, indent=2, sort_keys=True), file=sys.stderr)
        print("Observed:", json.dumps(observed, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    metric_total = sum(counts["MetricAt"] for counts in observed.values())
    derivative_total = sum(counts["DerivativesAt"] for counts in observed.values())
    declared = manifest["geometry_access"]
    if metric_total != declared["metric_evaluations"]:
        print("Metric-evaluation total disagrees with manifest", file=sys.stderr)
        return 1
    if derivative_total != declared["derivative_evaluations"]:
        print("Derivative-evaluation total disagrees with manifest", file=sys.stderr)
        return 1
    if metric_total + derivative_total != declared["total"]:
        print("Raw-access total disagrees with manifest", file=sys.stderr)
        return 1

    print(
        f"GEO-1 geometry interface is complete: {metric_total} metric and "
        f"{derivative_total} derivative evaluations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
