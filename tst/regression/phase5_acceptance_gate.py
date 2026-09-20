#!/usr/bin/env python3
"""Audit the compact phase-5 records before cleanup and release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(directory: Path, name: str) -> dict:
    with (directory / name).open(encoding="utf-8") as stream:
        record = json.load(stream)
    if not record.get("passed", False):
        raise RuntimeError(f"{name}: producer did not mark the record passed")
    return record


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records", type=Path, default=Path("docs/comparisons"), help="comparison directory"
    )
    args = parser.parse_args()

    shocks = load(args.records, "phase-05-sr-shocks.json")
    waves = load(args.records, "phase-05-sr-linear-waves.json")
    bondi = load(args.records, "phase-05-gr-bondi.json")
    monopole = load(args.records, "phase-05-gr-monopole.json")
    torus = load(args.records, "phase-05-gr-torus.json")
    convergence = load(args.records, "phase-05-relativity-convergence.json")
    capability = load(args.records, "phase-05-athenapk-capability.json")
    aggregate = load(args.records, "phase-05-relativity.json")

    shock_counts = {
        case: sum(record["case"] == case for record in shocks["records"])
        for case in ("hydro", "mhd")
    }
    wave_counts = {
        case: sum(record["case"] == case for record in waves["records"])
        for case in ("hydro", "mhd")
    }
    require(min(shock_counts.values()) >= 5, f"shock snapshots insufficient: {shock_counts}")
    require(min(wave_counts.values()) >= 5, f"wave snapshots insufficient: {wave_counts}")
    require(bondi["snapshot_pairs"] >= 5, "Bondi must compare at least five snapshots")
    require(monopole["snapshot_pairs"] >= 5, "monopole must compare at least five snapshots")
    require(torus["snapshot_pairs"] >= 5, "torus must compare at least five snapshots")

    require(shocks["maximum_linf"] <= shocks["absolute_tolerance"], "SR shock parity failed")
    require(waves["maximum_linf"] <= waves["absolute_tolerance"], "SR wave parity failed")
    require(bondi["maximum_linf"] <= bondi["absolute_tolerance"], "Bondi parity failed")
    require(monopole["maximum_linf"] <= monopole["absolute_tolerance"], "monopole parity failed")
    require(monopole["bitwise_unequal"] == 0, "monopole is not bitwise identical")
    require(
        torus["high_precision_line_maximum_linf"]
        <= torus["high_precision_line_tolerance"],
        "torus double-precision line parity failed",
    )
    require(
        torus["full_grid_maximum_linf"] <= torus["full_grid_tolerance"],
        "torus float32 full-grid parity failed",
    )
    require(
        convergence["minimum_observed_order_l1"]
        >= convergence["minimum_required_order_l1"],
        "relativity convergence order failed",
    )

    require(not capability["supports_special_relativity"], "AthenaPK SR audit changed")
    require(
        not capability["supports_fixed_background_general_relativity"],
        "AthenaPK fixed-GR audit changed",
    )
    require(aggregate["framework_gates"]["cpu_ctest_passed"] == 18, "CPU CTest gate missing")
    require(aggregate["framework_gates"]["cuda_relativity_unit"], "CUDA unit gate missing")
    require(aggregate["framework_gates"]["mpi_amr_all_finite"], "MPI+AMR gate missing")
    require(aggregate["framework_gates"]["srmhd_restart_linf"] == 0.0, "restart gate failed")

    print(
        "PANGU phase-5 acceptance PASS: "
        f"shock={shock_counts}, wave={wave_counts}, Bondi={bondi['snapshot_pairs']}, "
        f"monopole={monopole['snapshot_pairs']}, torus={torus['snapshot_pairs']}, "
        f"max_double_linf={max(shocks['maximum_linf'], waves['maximum_linf'], bondi['maximum_linf'], monopole['maximum_linf'], torus['high_precision_line_maximum_linf']):.17g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
