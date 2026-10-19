#!/usr/bin/env python3
"""Check that a PANGU field dump contains current (not stale) NR diagnostics.

Requires h5py and numpy. Intended for stationary_multilevel_pangu.in with
nr.z4c and nr.constraints dumped together after a nontrivial evolution.
"""
import argparse

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="PANGU final PHDF field dump")
    parser.add_argument("--require-evolved-theta", action="store_true")
    args = parser.parse_args()
    with h5py.File(args.file, "r") as data:
        # See Z4cComponent::theta and ConstraintComponent::theta.
        theta = data["nr.z4c"][:, 17, ...]
        diagnostic_theta = data["nr.constraints"][:, 7, ...]
        if not np.isfinite(theta).all() or not np.isfinite(diagnostic_theta).all():
            raise SystemExit("FAIL: non-finite theta")
        error = float(np.max(np.abs(theta - diagnostic_theta)))
        if not np.array_equal(theta, diagnostic_theta):
            raise SystemExit(f"FAIL: stale/inconsistent constraint theta; max error={error:.17e}")
        if args.require_evolved_theta and float(np.max(np.abs(theta))) < 1.0e-6:
            raise SystemExit("FAIL: theta has not evolved enough to detect a stale diagnostic")
        print(f"PASS: constraint theta matches evolved Z4c theta exactly; max error={error}")


if __name__ == "__main__":
    main()
