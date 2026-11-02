#!/usr/bin/env python3
"""Verify test-fluid gravity against the initial Schwarzschild acceleration.

A uniform, constant-pressure, initially resting, unmagnetized fluid must obey
d(W v^i)/dt = -gamma^{ij} partial_j alpha, even with nr.tmunu == 0.
The existing plasma plugin supplies Schwarzschild puncture data.  We evolve
only an exterior Cartesian patch, exclude boundary stencils, and compare two
spacings and a half-time control.  This is a short initial-acceleration test,
not a stationary accretion or outer-boundary benchmark.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
import resource
import subprocess
import time

import h5py
import numpy as np


def load_helper(repo):
    spec = importlib.util.spec_from_file_location(
        "plasma_gravity_input", repo / "tst/regression/compact_object_plasma.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def measure(directory, final_time):
    initial = next(p for p in directory.glob("*.phdf") if ".00000." in p.name)
    final = next(directory.glob("*.final.phdf"))
    with h5py.File(initial) as start, h5py.File(final) as end:
        assert abs(float(end["Info"].attrs["Time"]) - final_time) < 1e-12
        for name in ("mhd.prim", "mhd.cons", "mhd.b_cell", "mhd.divb", "nr.z4c", "nr.adm"):
            assert np.isfinite(start[name][:]).all() and np.isfinite(end[name][:]).all(), name
        assert np.max(abs(start["nr.tmunu"][:])) == 0
        assert np.max(abs(end["nr.tmunu"][:])) == 0
        assert np.max(abs(end["mhd.b_cell"][:])) == 0
        assert np.max(abs(end["mhd.divb"][:])) == 0
        reference, measured = [], []
        metric_error = 0.
        counts = 0
        for block in range(int(start["Info"].attrs["NumMeshBlocks"])):
            x, y, z = [start["VolumeLocations/" + axis][block] for axis in "xyz"]
            zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
            position = np.stack((xx, yy, zz), axis=0)
            radius = np.sqrt(np.sum(position**2, axis=0))
            psi = 1 + .5 / radius
            mask = ((xx > .8) & (xx < 1.2) & (abs(yy) < .2) & (abs(zz) < .2))
            if not np.any(mask):
                continue
            metric_error = max(metric_error, float(np.max(abs(
                start["nr.z4c"][block, 18][mask] - psi[mask]**-2))))
            assert np.max(abs(start["mhd.prim"][block, 1:4][:, mask])) == 0
            # M=1, alpha=psi^-2, gamma_ij=psi^4 delta_ij.
            acceleration = -position / (radius**3 * psi**7)[None]
            actual = end["mhd.prim"][block, 1:4] / final_time
            reference.append(acceleration[:, mask])
            measured.append(actual[:, mask])
            counts += int(mask.sum())
        expected = np.concatenate(reference, axis=1)
        actual = np.concatenate(measured, axis=1)
        error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        radial_cosine = float(np.min(np.sum(actual * expected, axis=0) /
                                     (np.linalg.norm(actual, axis=0) * np.linalg.norm(expected, axis=0))))
        scale_ratio = float(np.sum(actual * expected) / np.sum(expected**2))
        assert counts > 0 and metric_error < 1e-12
        return dict(relative_acceleration_l2=error, minimum_inward_cosine=radial_cosine,
                    fitted_acceleration_ratio=scale_ratio, sampled_cells=counts,
                    initial_lapse_error=metric_error, zero_backreaction=True,
                    final_time=final_time), actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pangu", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    repo = Path(__file__).resolve().parents[2]
    binary, root = args.pangu.resolve(), args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    helper = load_helper(repo)
    text = (repo / "plugins/compact_objects/examples/single_plasma_admission.in").read_text()
    text = re.sub(r"(?ms)^<parthenon/static_refinement\d+>.*?(?=^<|\Z)", "", text)
    report = dict(status="running", schema="pangu.test-fluid-gravity.v1", runs={},
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  analytic_initial_acceleration="-x^i/[r^3 (1+1/(2r))^7] for M=1",
                  acceptance=dict(finest_relative_l2=.02, minimum_inward_cosine=.999,
                                  minimum_spatial_order=1.5, half_time_relative_change=.01))

    def save():
        temporary = root / "summary.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(root / "summary.json")

    save()
    try:
        velocities = {}
        for name, n, duration in (("n16", 16, .001), ("n32", 32, .001),
                                  ("n32_half_time", 32, .0005)):
            directory = root / name
            directory.mkdir()
            changes = {("parthenon/mesh", "refinement"): "none",
                       ("parthenon/time", "tlim"): duration,
                       ("parthenon/time", "ncycle_out"): 1,
                       ("parthenon/output1", "dt"): 1,
                       ("parthenon/output2", "dt"): 1,
                       ("numerical_relativity", "dissipation"): 0,
                       ("plugin/compact_objects", "btilde3"): 0}
            for axis in (1, 2, 3):
                changes["parthenon/mesh", f"nx{axis}"] = n
                changes["parthenon/meshblock", f"nx{axis}"] = 16
                changes["parthenon/mesh", f"x{axis}min"] = .5 if axis == 1 else -.5
                changes["parthenon/mesh", f"x{axis}max"] = 1.5 if axis == 1 else .5
            deck = directory / "input.in"
            deck.write_text(helper.edit(text, changes))
            command = [str(binary), "-i", str(deck)]
            start = time.monotonic()
            with (directory / "run.log").open("w") as log:
                completed = subprocess.run(command, cwd=directory, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=args.timeout)
            entry = report["runs"][name] = dict(exit_code=completed.returncode,
                                                wall_seconds=time.monotonic() - start)
            save()
            assert completed.returncode == 0 and "Driver completed." in (directory / "run.log").read_text()
            result, velocity = measure(directory, duration)
            entry.update(result)
            velocities[name] = velocity
            save()
            print(name, json.dumps(entry), flush=True)
        coarse, fine, half = [report["runs"][name] for name in ("n16", "n32", "n32_half_time")]
        report["spatial_order"] = float(np.log2(coarse["relative_acceleration_l2"] /
                                                  fine["relative_acceleration_l2"]))
        report["half_time_relative_change"] = float(np.linalg.norm(
            velocities["n32"] - velocities["n32_half_time"]) / np.linalg.norm(velocities["n32_half_time"]))
        assert fine["relative_acceleration_l2"] < .02, "incorrect gravitational acceleration magnitude"
        assert min(r["minimum_inward_cosine"] for r in (coarse, fine, half)) > .999, "incorrect fall direction"
        assert report["spatial_order"] > 1.5, "acceleration did not converge with spatial refinement"
        assert report["half_time_relative_change"] < .01, "time truncation dominates the acceleration test"
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", failure=repr(exc))
        save()
        raise
    save()


if __name__ == "__main__":
    main()
