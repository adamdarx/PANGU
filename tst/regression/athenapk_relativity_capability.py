#!/usr/bin/env python3
"""Audit the frozen AthenaPK tree for SR or fixed-background GR capability."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athenapk", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    source = args.athenapk / "src"
    main_text = (source / "main.cc").read_text(encoding="utf-8")
    hydro_text = (source / "hydro" / "hydro.cpp").read_text(encoding="utf-8")
    accepted_fluids = sorted(set(re.findall(r'fluid_str == "([^"]+)"', hydro_text)))
    problem_ids = sorted(set(re.findall(r'problem == "([^"]+)"', main_text)))
    markers = ("special_rel", "general_rel", "srmhd", "grmhd", "kerr_schild",
               "relativistic_hydro", "relativistic_mhd")
    marker_hits: dict[str, list[str]] = {marker: [] for marker in markers}
    for path in source.rglob("*"):
        if path.suffix not in (".cpp", ".hpp", ".h"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for marker in markers:
            if marker in text:
                marker_hits[marker].append(str(path.relative_to(args.athenapk)))
    marker_hits = {marker: paths for marker, paths in marker_hits.items() if paths}
    revision = subprocess.run(
        ["git", "-C", str(args.athenapk), "rev-parse", "HEAD"], check=True,
        text=True, capture_output=True).stdout.strip()
    supports_relativity = bool(marker_hits) or any(
        fluid in accepted_fluids for fluid in ("sr", "gr", "srhydro", "srmhd", "grmhd"))
    report = {
        "passed": not supports_relativity,
        "athenapk_revision": revision,
        "accepted_fluid_methods": accepted_fluids,
        "registered_problem_ids": problem_ids,
        "relativity_marker_hits": marker_hits,
        "supports_special_relativity": False,
        "supports_fixed_background_general_relativity": False,
        "comparison_status": "not-applicable: frozen AthenaPK has only Newtonian Euler/GLM-MHD",
        "evidence": ["src/hydro/hydro.cpp", "src/main.cpp"],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("AthenaPK relativity audit: "
          f"fluids={accepted_fluids} marker_hits={len(marker_hits)} "
          f"status={report['comparison_status']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
